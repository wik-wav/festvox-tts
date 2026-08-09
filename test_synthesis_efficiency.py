import array
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from cache_support import BoundedMemoryCache, FileIdentityCache
import synth_diphone as sd


def _write_fixture_voice(root: Path) -> Path:
    (root / "dic").mkdir(parents=True)
    (root / "wav").mkdir()
    rate = 16000
    time = np.arange(rate // 2, dtype=np.float64) / rate
    samples = np.rint(np.sin(2.0 * np.pi * 180.0 * time) * 12000.0)
    pcm = samples.astype("<i2")
    with wave.open(str(root / "wav" / "fixture.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())
    index = {
        "name": "fixture",
        "average_pitch_hz": 180.0,
        "index": {
            "pau-a": ["fixture.wav", 0.00, 0.10, 0.24],
            "a-pau": ["fixture.wav", 0.24, 0.34, 0.49],
        },
        "alternatives": {},
    }
    (root / "dic" / "diphone_index.json").write_text(
        json.dumps(index), encoding="utf-8")
    return root


class FileIdentityCacheTests(unittest.TestCase):
    def test_memory_lru_enforces_entry_and_byte_limits(self):
        cache = BoundedMemoryCache(
            "fixture-memory", max_entries=2, max_bytes=5,
            size_func=lambda value: len(value))
        cache["a"] = b"aa"
        cache["b"] = b"bb"
        self.assertEqual(cache["a"], b"aa")
        cache["c"] = b"ccc"

        self.assertIn("a", cache)
        self.assertNotIn("b", cache)
        self.assertIn("c", cache)
        self.assertLessEqual(cache.info()["bytes"], 5)
        self.assertLessEqual(len(cache), 2)

    def test_file_change_invalidates_one_cached_value(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "value.json"
            source.write_text('{"value": 1}', encoding="utf-8")
            calls = []
            cache = FileIdentityCache("fixture", max_entries=2,
                                      max_bytes=1024 * 1024)

            def load(path):
                calls.append(path)
                return json.loads(path.read_text(encoding="utf-8"))

            first = cache.get(source, load)
            self.assertIs(cache.get(source, load), first)
            source.write_text('{"value": 200}', encoding="utf-8")
            changed = cache.get(source, load)

            self.assertEqual(changed["value"], 200)
            self.assertEqual(len(calls), 2)
            self.assertEqual(cache.info()["entries"], 1)
            cache.clear()
            self.assertEqual(cache.info()["bytes"], 0)

    def test_same_size_restored_timestamp_still_invalidates(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "value.json"
            source.write_text('{"value":1}', encoding="utf-8")
            original = source.stat()
            cache = FileIdentityCache(
                "fixture", max_entries=2, max_bytes=1024 * 1024)
            self.assertEqual(cache.get(
                source, lambda path: path.read_text(encoding="utf-8")),
                '{"value":1}')

            source.write_text('{"value":2}', encoding="utf-8")
            os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))

            self.assertEqual(cache.get(
                source, lambda path: path.read_text(encoding="utf-8")),
                '{"value":2}')
            self.assertEqual(cache.info()["misses"], 2)


class DiphoneRuntimeCacheTests(unittest.TestCase):
    def setUp(self):
        sd.clear_synth_text_cache()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = _write_fixture_voice(Path(self.temporary.name) / "voice")

    def tearDown(self):
        sd.clear_synth_text_cache()
        self.temporary.cleanup()

    def test_slice_cache_is_bounded_and_thread_safe(self):
        database = sd.DiphoneDB(
            self.root,
            cache_max_files=1,
            cache_max_bytes=1024 * 1024,
            slice_cache_max_entries=2,
            slice_cache_max_bytes=1024 * 1024,
        )
        first = database.slice_info("pau-a", 100.0)
        second = database.slice_info("pau-a", 100.0)
        self.assertIsNot(first[1], second[1])
        original = second[1][20]
        first[1][20] = original + 1000
        self.assertEqual(
            database.slice_info("pau-a", 100.0)[1][20], original)

        with ThreadPoolExecutor(max_workers=6) as pool:
            rows = list(pool.map(
                lambda _index: database.slice_info("a-pau", 90.0),
                range(24),
            ))
        self.assertTrue(all(len(row[1]) for row in rows))
        database.slice_info("pau-a", 80.0)

        info = database.cache_info()
        self.assertLessEqual(info["files"], 1)
        self.assertLessEqual(info["slices"], 2)
        self.assertGreater(info["slice_hits"], 1)
        database.clear_cache()
        self.assertEqual(database.cache_info()["total_bytes"], 0)

    def test_replaced_source_wav_invalidates_decoded_and_slice_caches(self):
        database = sd.DiphoneDB(self.root)
        first = database.slice_info("pau-a", 100.0)[1]
        target = self.root / "wav" / "fixture.wav"
        original = target.stat()
        replacement = target.with_suffix(".replacement.wav")
        with wave.open(str(replacement), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(array.array("h", [7000] * 8000).tobytes())
        os.utime(replacement,
                 ns=(original.st_atime_ns, original.st_mtime_ns))
        os.replace(replacement, target)

        changed = database.slice_info("pau-a", 100.0)[1]

        self.assertNotEqual(first.tobytes(), changed.tobytes())
        self.assertTrue(all(value == 7000 for value in changed))
        self.assertGreaterEqual(
            database.cache_info()["stale_invalidations"], 1)

    def test_in_place_source_wav_rewrite_invalidates_with_restored_metadata(self):
        database = sd.DiphoneDB(self.root)
        first = database.slice_info("pau-a", 100.0)[1]
        target = self.root / "wav" / "fixture.wav"
        original = target.stat()
        original_inode = int(getattr(original, "st_ino", 0))
        with wave.open(str(target), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(array.array("h", [3000] * 8000).tobytes())
        os.utime(target,
                 ns=(original.st_atime_ns, original.st_mtime_ns))

        changed = database.slice_info("pau-a", 100.0)[1]

        self.assertEqual(int(getattr(target.stat(), "st_ino", 0)),
                         original_inode)
        self.assertEqual(target.stat().st_size, original.st_size)
        self.assertEqual(target.stat().st_mtime_ns, original.st_mtime_ns)
        self.assertNotEqual(first.tobytes(), changed.tobytes())
        self.assertTrue(all(value == 3000 for value in changed))
        self.assertGreaterEqual(
            database.cache_info()["stale_invalidations"], 1)

    def test_direct_pcm_matches_default_wav_output(self):
        database = sd.DiphoneDB(self.root)
        encoded = sd.render(database, ["a"], encode_wav=True)
        direct = sd.render(
            database, ["a"], encode_wav=False, return_pcm=True)

        with wave.open(io.BytesIO(encoded["wav"]), "rb") as wav:
            encoded_pcm = array.array("h")
            encoded_pcm.frombytes(wav.readframes(wav.getnframes()))
        self.assertIsNone(direct["wav"])
        self.assertEqual(encoded_pcm.tobytes(), direct["pcm16"].tobytes())
        self.assertEqual(encoded["segments"], direct["segments"])
        self.assertEqual(encoded["splice_records"], direct["splice_records"])

    def test_legacy_join_fault_does_not_change_unit_selection(self):
        database = sd.DiphoneDB(self.root)

        measured = sd.render(database, ["a"], return_pcm=True,
                             encode_wav=False)
        legacy = sd.render(database, ["a"], return_pcm=True,
                           encode_wav=False, legacy_joins=True)

        self.assertEqual(measured["diphones"], legacy["diphones"])
        self.assertEqual(measured["selected_units"],
                         legacy["selected_units"])
        self.assertEqual(measured["join_mode"], "measured")
        self.assertEqual(legacy["join_mode"], "legacy")
        self.assertTrue(all(
            row["position_source"] == "python-measured-crossfade"
            for row in measured["splice_records"]))
        self.assertTrue(all(
            row["position_source"] == "python-legacy-linear-crossfade"
            for row in legacy["splice_records"]))

    def test_one_call_synthesis_reuses_bounded_voice_database(self):
        config = {"festvox_db": [str(self.root)], "synth_speed": 1.0}

        first = sd.synth_text(config, "a", lang="asaxi")
        second = sd.synth_text(config, "a", lang="asaxi")

        self.assertEqual(first["wav"], second["wav"])
        info = sd.synth_text_cache_info()
        self.assertEqual(info["voices"], 1)
        self.assertLessEqual(info["voices"], info["max_voices"])
        self.assertGreater(info["databases"][0]["slice_hits"], 0)

    def test_one_call_synthesis_routes_legacy_join_mode(self):
        config = {"festvox_db": [str(self.root)], "synth_speed": 1.0}

        result = sd.synth_text(
            config, "a", lang="asaxi", legacy_joins=True)

        self.assertEqual(result["join_mode"], "legacy")
        self.assertTrue(all(
            row["position_source"] == "python-legacy-linear-crossfade"
            for row in result["splice_records"]))


if __name__ == "__main__":
    unittest.main()
