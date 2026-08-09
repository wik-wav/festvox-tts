import unittest
import math
import statistics

from japanese_frontend import analyze_japanese
from japanese_prosody_ab import build_pitch_systems
from japanese_synthesis import create_synthesis_plan


class JapaneseProsodyABTests(unittest.TestCase):
    def test_repeated_phrase_ab_changes_pitch_only(self):
        utterance = analyze_japanese(
            "kore wa tesuto desu. kore wa tesuto desu.", mode="kana")
        plan = create_synthesis_plan(utterance, base_pitch_hz=165.0)
        systems = build_pitch_systems(
            utterance, plan, base_pitch_hz=165.0, fall_percent=18.0)

        self.assertEqual(set(systems), {"legacy_pitch", "contextual_pitch"})
        self.assertNotEqual(
            systems["legacy_pitch"]["pitch_model_id"],
            systems["contextual_pitch"]["pitch_model_id"],
        )
        self.assertEqual(
            [row["start"] for row in
             systems["legacy_pitch"]["intonation_blocks"]],
            [row["start"] for row in
             systems["contextual_pitch"]["intonation_blocks"]],
        )
        legacy = systems["legacy_pitch"]["raw_targets"]
        contextual = systems["contextual_pitch"]["raw_targets"]
        self.assertEqual([time for time, _f0 in legacy],
                         [time for time, _f0 in contextual])
        self.assertNotEqual([f0 for _time, f0 in legacy],
                            [f0 for _time, f0 in contextual])

        by_phrase = {}
        for target, (_time, f0) in zip(plan.f0_targets, contextual):
            by_phrase.setdefault(target.phrase_index, []).append(f0)
        first, second = by_phrase[0], by_phrase[1]
        self.assertEqual(len(first), len(second))
        differences = [
            12.0 * math.log2(left / right)
            for left, right in zip(first, second)
        ]
        # Phrase-position context changes contour shape, not register.
        self.assertLess(abs(statistics.mean(differences)), 0.1)
        self.assertGreater(max(differences) - min(differences), 0.7)


if __name__ == "__main__":
    unittest.main()
