from pathlib import Path
import tempfile
import unittest

from arpasing_profile import (
    DEFAULT_PHONEME_MAP_PATH,
    load_arpasing_profile,
    map_japanese_mora,
)
from japanese_frontend import analyze_japanese
from japanese_synthesis import create_synthesis_plan


class ArpasingProfileTests(unittest.TestCase):
    def test_bundled_default_is_the_supplied_mapping(self):
        profile = load_arpasing_profile()
        self.assertEqual(
            profile.source_sha256,
            "6356B50F3C25417797130F94F47E9D52C3B4A96B7DC5FFDB511A18358C517A99",
        )
        self.assertEqual(profile.source_name,
                         "bundled:profiles/en-jap-mapping.yaml")
        self.assertGreater(len(profile.entries), 1500)
        self.assertEqual(profile.resolve("か").phonemes, ("k", "a"))
        self.assertEqual(profile.resolve("きゃ").phonemes, ("ky", "a"))

    def test_conflicting_mapping_is_deterministic_and_inspectable(self):
        profile = load_arpasing_profile()
        resolved = profile.resolve("ヴァ1")
        self.assertEqual(resolved.phonemes, ("v", "a"))
        self.assertIn(("v",), resolved.alternatives)
        self.assertTrue(any(
            item.code == "conflicting_grapheme_mapping"
            and "ヴァ1" in item.message
            for item in profile.diagnostics
        ))

    def test_moraic_nasal_routes_are_profile_defined(self):
        runtime = load_arpasing_profile().runtime_map()
        cases = {
            "m": "mm",
            "k": "nng",
            "s": "xn",
            None: "xn",
            "t": "nn",
            "n": "nn",
        }
        for following, expected in cases.items():
            with self.subTest(following=following):
                mapped, _reason = map_japanese_mora(
                    "ん", ("N",), runtime,
                    following_phone=following,
                )
                self.assertEqual(mapped, (expected,))

    def test_label_derived_liquids_recover_profile_tap_mapping(self):
        profile = load_arpasing_profile()
        runtime = profile.runtime_map()
        cases = (
            ("ra", ("r", "a"), ("dx", "a")),
            ("ri", ("r", "i"), ("dxy", "i")),
            ("ru", ("r", "u"), ("dx", "u")),
            ("re", ("r", "e"), ("dx", "e")),
            ("ro", ("r", "o"), ("dx", "o")),
        )
        for reading, canonical, expected in cases:
            with self.subTest(reading=reading):
                mapped, reason = map_japanese_mora(
                    reading,
                    canonical,
                    runtime,
                    available_phones=tuple(profile.symbols),
                )
                self.assertEqual(mapped, expected)
                self.assertEqual(reason, "profile_canonical_mora_map")

    def test_japanese_only_plan_keeps_canonical_liquid(self):
        utterance = analyze_japanese("\u3089\u308a", mode="kana")
        plan = create_synthesis_plan(utterance)
        spoken = [
            item.phone for item in plan.segments if item.phone != "pau"
        ]
        self.assertEqual(spoken, ["r", "a", "r", "i"])

    def test_custom_profile_parses_without_yaml_dependency(self):
        payload = """%YAML 1.2
---
symbols:
  - {symbol: a, type: vowel}
  - {symbol: k, type: stop}
timings:
  - {symbol: k, value: 1.4}
replacements:
  - {from: [old], to: [a]}
entries:
  - grapheme: か
    phonemes: [k, a]
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.yaml"
            path.write_text(payload, encoding="utf-8")
            first = load_arpasing_profile(path)
            second = load_arpasing_profile(path)
        self.assertEqual(first.runtime_map(), second.runtime_map())
        self.assertEqual(first.resolve("か").phonemes, ("k", "a"))

    def test_japanese_planner_uses_profile_phone_namespace(self):
        profile = load_arpasing_profile()
        runtime = {
            "language": "en",
            "supported_languages": ["en", "asaxi", "ja"],
            "voice_entry_points": {"ja": "voice_test_ja"},
            "japanese_phoneme_map": profile.runtime_map(),
            "phones": list(profile.symbols),
            "average_pitch_hz": 180.0,
        }
        utterance = analyze_japanese("かんた", mode="kana")
        plan = create_synthesis_plan(
            utterance, runtime_metadata=runtime, base_pitch_hz=180.0
        )
        spoken = [item.phone for item in plan.segments
                  if item.phone != "pau"]
        self.assertEqual(spoken, ["k", "a", "nn", "t", "a"])


if __name__ == "__main__":
    unittest.main()
