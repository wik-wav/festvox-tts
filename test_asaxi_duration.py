from __future__ import annotations

import math
import unittest

import asaxi_duration as duration
import asaxi_prosody


class AsaxiDurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dictionary = asaxi_prosody.load_dictionary()

    def plan(self, text: str, *, speed: float = 1.0):
        linguistic = asaxi_prosody.analyze_utterance(
            text, self.dictionary
        )
        entries = [("pau", 0.0, 0.08)]
        cursor = 0.08
        for phone in linguistic.phones:
            entries.append((phone, cursor, cursor + 0.10))
            cursor += 0.10
        entries.append(("pau", cursor, cursor + 0.09))
        return linguistic, duration.plan_durations(
            linguistic, entries, speed=speed
        )

    @staticmethod
    def spoken(plan):
        return [
            (phone, value)
            for phone, value in plan.entries
            if phone not in duration.PAUSE_PHONES
        ]

    def test_cv_phone_durations_are_not_equal(self) -> None:
        _linguistic, planned = self.plan("ka")
        spoken = dict(self.spoken(planned))

        self.assertLess(spoken["k"], spoken["a"])
        self.assertGreater(spoken["k"], 0.025)
        self.assertLess(sum(spoken.values()), 0.18)

    def test_longer_fricative_partly_shortens_following_vowel(self) -> None:
        _stop_plan, stop = self.plan("ka")
        _fricative_plan, fricative = self.plan("sa")
        stop_rows = self.spoken(stop)
        fricative_rows = self.spoken(fricative)

        self.assertGreater(fricative_rows[0][1], stop_rows[0][1])
        self.assertLess(fricative_rows[1][1], stop_rows[1][1])
        self.assertLess(
            sum(value for _phone, value in fricative_rows)
            - sum(value for _phone, value in stop_rows),
            0.020,
        )

    def test_voiced_stop_is_shorter_than_voiceless_stop(self) -> None:
        _voiceless_plan, voiceless = self.plan("ka")
        _voiced_plan, voiced = self.plan("ba")

        self.assertLess(
            self.spoken(voiced)[0][1],
            self.spoken(voiceless)[0][1],
        )

    def test_closed_mora_is_bounded_and_coda_is_not_a_full_beat(self) -> None:
        linguistic, planned = self.plan("kem.ma")
        rows = {row.segment_index: row for row in planned.phones}
        first = linguistic.moras[0]
        first_rows = [
            row for row in rows.values()
            if row.mora_index == first.index
        ]

        self.assertEqual(
            [row.role for row in first_rows],
            ["onset", "nucleus", "coda"],
        )
        self.assertLess(first_rows[-1].duration_seconds, 0.060)
        self.assertLess(
            sum(row.duration_seconds for row in first_rows),
            duration.DEFAULT_CONFIG.maximum_mora_seconds,
        )

    def test_syllabic_nasal_occupies_one_mora_as_a_nucleus(self) -> None:
        linguistic, planned = self.plan("mmba")
        nasal = next(
            row for row in planned.phones if row.phone == "mm"
        )

        self.assertEqual(linguistic.moras[0].kind, "syllabic_nasal")
        self.assertEqual(nasal.role, "nucleus")
        self.assertGreater(nasal.duration_seconds, 0.10)
        self.assertLess(nasal.duration_seconds, 0.14)

    def test_stop_geminate_has_one_structural_hold_mora(self) -> None:
        linguistic, planned = self.plan("tte")
        hold = next(row for row in planned.phones if row.phone == "cl")

        self.assertEqual(linguistic.moras[0].kind, "geminate")
        self.assertEqual(hold.role, "geminate_hold")
        self.assertGreater(hold.duration_seconds, 0.09)

    def test_continuant_geminate_extends_one_following_phone(self) -> None:
        linguistic, planned = self.plan("sse")
        sibilant = next(row for row in planned.phones if row.phone == "s")

        self.assertEqual(linguistic.moras[0].kind, "geminate")
        self.assertIn(
            "continuant_geminate_hold", sibilant.modifiers
        )
        self.assertEqual(sibilant.absorbed_mora_indices, (0,))
        self.assertGreater(sibilant.duration_seconds, 0.15)

    def test_diphthong_is_one_mora_not_two_beats(self) -> None:
        linguistic, planned = self.plan("\u0103")

        self.assertEqual(len(linguistic.moras), 1)
        self.assertEqual(linguistic.phones, ("a", "y"))
        self.assertLess(
            sum(value for _phone, value in self.spoken(planned)),
            0.16,
        )

    def test_doubled_vowel_retains_two_morae(self) -> None:
        linguistic, planned = self.plan("aa")

        self.assertEqual(len(linguistic.moras), 2)
        self.assertEqual(len(self.spoken(planned)), 2)
        self.assertGreater(
            sum(value for _phone, value in self.spoken(planned)),
            0.20,
        )

    def test_glottal_coda_does_not_add_a_mora(self) -> None:
        linguistic, planned = self.plan("a'")
        rows = self.spoken(planned)

        self.assertEqual(len(linguistic.moras), 1)
        self.assertEqual([phone for phone, _value in rows], ["a", "q"])
        self.assertLess(rows[1][1], 0.05)
        self.assertLess(sum(value for _phone, value in rows), 0.16)

    def test_phrase_final_lengthening_is_local(self) -> None:
        linguistic, planned = self.plan("ka ka.")
        mora_durations = []
        for mora in linguistic.moras:
            mora_durations.append(sum(
                row.duration_seconds
                for row in planned.phones
                if row.mora_index == mora.index
            ))

        self.assertEqual(len(mora_durations), 2)
        self.assertGreater(mora_durations[1], mora_durations[0])
        self.assertIn(
            "phrase_final_lengthening",
            planned.phones[-1].modifiers,
        )

    def test_speed_scales_only_modeled_phones(self) -> None:
        _normal_linguistic, normal = self.plan("kem.ma", speed=1.0)
        _fast_linguistic, fast = self.plan("kem.ma", speed=2.0)

        for (_phone_a, normal_value), (_phone_b, fast_value) in zip(
            self.spoken(normal), self.spoken(fast)
        ):
            self.assertAlmostEqual(fast_value, normal_value / 2.0, places=6)
        self.assertEqual(normal.entries[0][1], fast.entries[0][1])
        self.assertEqual(normal.entries[-1][1], fast.entries[-1][1])

    def test_metadata_is_finite_and_deterministic(self) -> None:
        _linguistic, first = self.plan("kem.ma")
        _linguistic, second = self.plan("kem.ma")

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(all(
            math.isfinite(value) and value >= 0.0
            for _phone, value in first.entries
        ))
        self.assertEqual(first.model_id, duration.ASAXI_DURATION_MODEL_ID)


if __name__ == "__main__":
    unittest.main()
