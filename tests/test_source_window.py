import unittest

from source_window import (
    DEFAULT_SOURCE_WINDOW_MS,
    build_source_window_plan,
    effective_oto_overlap_ms,
    source_window_variant_names,
)


class SourceWindowTests(unittest.TestCase):
    def test_default_half_window_matches_validated_normal_phone_cap(self):
        self.assertEqual(DEFAULT_SOURCE_WINDOW_MS, 60.0)

    def test_zero_overlap_guard_is_bounded_and_reversible(self):
        self.assertEqual(effective_oto_overlap_ms(100, 0), 0.0)
        self.assertEqual(
            effective_oto_overlap_ms(100, 0, zero_overlap_guard_ms=12),
            12.0,
        )
        self.assertEqual(
            effective_oto_overlap_ms(20, 0, zero_overlap_guard_ms=12),
            5.0,
        )
        self.assertEqual(effective_oto_overlap_ms(100, 30), 30.0)
        self.assertEqual(
            effective_oto_overlap_ms(100, 0),
            0.0,
        )

    def test_adaptive_window_bounds_normal_phone_but_retains_full_geometry(self):
        plan = build_source_window_plan(
            0.10, 0.30, 0.80, mode="adaptive", half_window_ms=120)

        self.assertEqual(plan.geometry("base"), (0.18, 0.30, 0.42))
        self.assertEqual(plan.geometry("both"), (0.10, 0.30, 0.80))
        self.assertAlmostEqual(plan.left_activation_duration, 0.40)
        self.assertAlmostEqual(plan.right_activation_duration, 1.00)

    def test_adaptive_window_uses_only_the_stretched_side(self):
        plan = build_source_window_plan(
            0.10, 0.30, 0.80, mode="adaptive", half_window_ms=120)

        self.assertEqual(plan.variant_kind(0.10, 0.10), "base")
        self.assertEqual(plan.variant_kind(0.45, 0.10), "left")
        self.assertEqual(plan.variant_kind(0.10, 1.10), "right")
        self.assertEqual(plan.variant_kind(0.45, 1.10), "both")

    def test_bounded_and_full_modes_are_explicitly_reversible(self):
        bounded = build_source_window_plan(
            0.10, 0.30, 0.80, mode="bounded", half_window_ms=120)
        full = build_source_window_plan(
            0.10, 0.30, 0.80, mode="full", half_window_ms=120)

        self.assertEqual(bounded.geometry(), (0.18, 0.30, 0.42))
        self.assertEqual(bounded.geometry("both"), bounded.geometry("base"))
        self.assertEqual(
            bounded.to_dict()["full"],
            {"start": 0.1, "phone_boundary": 0.3, "end": 0.8},
        )
        self.assertEqual(
            len(set(source_window_variant_names("a", bounded).values())), 1)
        self.assertEqual(bounded.variant_kind(9.0, 9.0), "base")
        self.assertEqual(full.geometry(), (0.10, 0.30, 0.80))
        self.assertEqual(full.to_dict()["mode"], "full")

    def test_variant_names_do_not_duplicate_an_unbounded_side(self):
        plan = build_source_window_plan(
            0.20, 0.30, 0.80, mode="adaptive", half_window_ms=120)
        names = source_window_variant_names("a", plan)

        self.assertEqual(names["left"], "a")
        self.assertEqual(names["right"], "a__wr")
        self.assertEqual(names["both"], "a__wr")


if __name__ == "__main__":
    unittest.main()
