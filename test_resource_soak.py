import json
from pathlib import Path
import tempfile
import time
import tracemalloc
import unittest

import resource_soak as soak


def snapshot(label, cycle, elapsed, rss=100, private=80, python=10,
             cpu=0.0, threads=2, handles=10, children=0, temp_files=0,
             temp_bytes=0, logical=None):
    return soak.ResourceSnapshot(
        label=label, cycle=cycle, elapsed_seconds=elapsed,
        rss_bytes=rss, private_bytes=private, virtual_bytes=None,
        cpu_seconds=cpu, thread_count=threads, python_thread_count=1,
        handle_count=handles, child_process_count=children,
        temp_file_count=temp_files, temp_bytes=temp_bytes,
        python_allocated_bytes=python, python_peak_bytes=python,
        qt_widget_count=3, gpu_memory_bytes=None,
        gpu_status="not tested", logical_resources=dict(logical or {}),
    )


class ResourceSoakTests(unittest.TestCase):
    def test_phone_sequence_enumeration_honors_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "dic").mkdir()
            index = {
                "index": {
                    "pau-a": [], "a-pau": [], "a-b": [], "b-pau": [],
                    "a-c": [], "c-pau": [], "a-d": [], "d-pau": [],
                }
            }
            (root / "dic" / "diphone_index.json").write_text(
                json.dumps(index), encoding="utf-8")

            rows = soak._soak_phone_sequences(root, limit=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(set(rows)), 2)

    def test_stable_snapshots_pass_and_include_idle_cpu(self):
        rows = [snapshot("startup", -1, 0.0, cpu=0.0)]
        for cycle in range(6):
            rows.append(snapshot(
                "cycle", cycle, cycle + 1.0, rss=100 + cycle,
                private=80 + cycle, python=10 + cycle, cpu=.1 * cycle,
                logical={"playback_timer_active": False,
                         "playback_finish_timer_active": False}))
        rows.extend([
            snapshot("idle_start", 6, 8.0, cpu=.6),
            snapshot("idle_end", 6, 10.0, cpu=.62),
        ])

        result = soak.analyze_snapshots(rows, warmup=2)

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["idle_cpu_percent"], 1.0)
        self.assertEqual(result["failed_checks"], [])

    def test_growth_and_pending_timer_are_reported(self):
        mib = soak.MIB
        rows = [snapshot("startup", -1, 0.0)]
        for cycle in range(6):
            rows.append(snapshot(
                "cycle", cycle, cycle + 1.0,
                rss=100 + cycle * 2 * mib,
                private=80 + cycle * 2 * mib,
                python=10 + cycle * mib,
                logical={"playback_timer_active": cycle == 5}))
        rows.extend([
            snapshot("idle_start", 6, 8.0, cpu=0.0),
            snapshot("idle_end", 6, 9.0, cpu=.2),
        ])

        result = soak.analyze_snapshots(rows, warmup=1)

        self.assertFalse(result["passed"])
        self.assertIn("rss_slope_mib_per_cycle", result["failed_checks"])
        self.assertIn("stopped_playback_timers", result["failed_checks"])
        self.assertIn("idle_cpu_percent", result["failed_checks"])

    def test_one_time_handle_setup_is_separate_from_steady_growth(self):
        rows = [snapshot("startup", -1, 0.0, handles=10)]
        for cycle in range(6):
            rows.append(snapshot(
                "cycle", cycle, cycle + 1.0, handles=18,
                logical={"playback_timer_active": False}))
        rows.extend([
            snapshot("idle_start", 6, 8.0, handles=17),
            snapshot("idle_end", 6, 9.0, handles=17),
        ])

        result = soak.analyze_snapshots(rows, warmup=1)

        self.assertTrue(result["passed"])
        self.assertEqual(result["deltas"]["handle_count"], 7)
        self.assertEqual(result["steady_state_deltas"]["handle_count"], 0)

    def test_per_cycle_handle_growth_fails(self):
        rows = [snapshot("startup", -1, 0.0, handles=10)]
        for cycle in range(6):
            rows.append(snapshot(
                "cycle", cycle, cycle + 1.0, handles=20 + cycle,
                logical={"playback_timer_active": False}))
        rows.extend([
            snapshot("idle_start", 6, 8.0, handles=25),
            snapshot("idle_end", 6, 9.0, handles=25),
        ])

        result = soak.analyze_snapshots(rows, warmup=1)

        self.assertFalse(result["passed"])
        self.assertIn("handle_growth", result["failed_checks"])

    def test_current_process_snapshot_is_finite_and_non_destructive(self):
        tracemalloc.stop()
        tracemalloc.start()
        self.addCleanup(tracemalloc.stop)

        row = soak.take_snapshot("test", 0, time.monotonic())

        self.assertGreaterEqual(row.cpu_seconds, 0.0)
        self.assertGreaterEqual(row.thread_count, 1)
        self.assertGreaterEqual(row.temp_file_count, 0)
        self.assertGreaterEqual(row.python_allocated_bytes, 0)

    def test_untraced_snapshot_marks_python_heap_unavailable(self):
        tracemalloc.stop()

        row = soak.take_snapshot("test", 0, time.monotonic())

        self.assertIsNone(row.python_allocated_bytes)
        self.assertIsNone(row.python_peak_bytes)


if __name__ == "__main__":
    unittest.main()
