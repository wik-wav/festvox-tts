import unittest

import special_phones as sp


class SpecialPhoneRealizationTests(unittest.TestCase):
    def test_generated_cl_anticipates_following_consonant(self):
        resolution = sp.resolve_special_phone_sequence(
            ["pau", "i", "cl", "s", "o", "pau"],
            metadata={"context_model": "oto_directional_v1"},
            available_diphones={
                "pau-i", "i-s", "s-s", "s-o", "o-pau",
            },
        )

        self.assertEqual(
            resolution.render_phones,
            ("pau", "i", "s", "s", "o", "pau"),
        )
        closure = resolution.realizations[0]
        self.assertEqual(len(resolution.realizations), 1)
        self.assertEqual(closure.phone, "cl")
        self.assertEqual(closure.source_phone, "s")
        self.assertEqual(closure.status, "resolved")
        self.assertEqual(closure.required_diphones, ("i-s", "s-s"))

    def test_resolution_is_language_neutral(self):
        metadata = {"context_model": "oto_directional_v1"}
        inventory = {"a-t", "t-t", "t-a"}

        first = sp.resolve_special_phone_sequence(
            ["a", "cl", "t", "a"],
            metadata=metadata,
            available_diphones=inventory,
        )
        second = sp.resolve_special_phone_sequence(
            ["a", "cl", "t", "a"],
            metadata=metadata,
            available_diphones=inventory,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.render_phones, ("a", "t", "t", "a"))
        closure = first.realizations[0]
        self.assertEqual(closure.required_diphones, ("a-t", "t-t"))

    def test_missing_hold_is_visible_and_does_not_claim_resolution(self):
        result = sp.resolve_special_phone_sequence(
            ["i", "cl", "s", "o"],
            metadata={"context_model": "oto_directional_v1"},
            available_diphones={"i-s", "s-o", "i-cl", "cl-s"},
        )

        self.assertEqual(result.render_phones, ("i", "cl", "s", "o"))
        self.assertEqual(result.unresolved[0].status,
                         "missing_source_diphones")
        self.assertEqual(result.unresolved[0].missing_diphones, ("s-s",))

    def test_orphan_cl_sources_silence_not_literal_oto(self):
        result = sp.resolve_special_phone_sequence(
            ["a", "cl", "pau"],
            metadata={"context_model": "oto_directional_v1"},
            available_diphones={"a-pau", "pau-pau"},
        )

        self.assertEqual(result.render_phones, ("a", "pau", "pau"))
        self.assertEqual(result.realizations[-1].status,
                         "resolved_silence_fallback")

    def test_creator_declared_literal_cl_coexists_with_structural_cl(self):
        policy = sp.generated_voice_policy(
            literal_phone_mappings={"cl_literal": "cl"}
        )
        metadata = {
            "context_model": "oto_directional_v1",
            "special_phone_realizations": policy,
        }
        literal = sp.resolve_special_phone_sequence(
            ["a", "cl_literal", "t"],
            metadata=metadata,
            available_diphones={"a-cl", "cl-t", "a-t", "t-t"},
        )
        structural = sp.resolve_special_phone_sequence(
            ["a", "cl", "t"],
            metadata=metadata,
            available_diphones={"a-cl", "cl-t", "a-t", "t-t"},
        )

        self.assertEqual(literal.render_phones, ("a", "cl", "t"))
        self.assertEqual(literal.realizations[0].mode,
                         sp.LITERAL_ALIAS_MODE)
        self.assertEqual(
            structural.render_phones, ("a", "t", "t")
        )
        self.assertEqual(structural.realizations[0].mode,
                         sp.STRUCTURAL_CL_MODE)

    def test_old_literal_policy_is_upgraded_to_coexisting_alias(self):
        metadata = {
            "context_model": "oto_directional_v1",
            "special_phone_realizations": {
                "schema_version": 1,
                "phones": {"cl": {"mode": "literal"}},
            },
        }

        structural = sp.resolve_special_phone_sequence(
            ["a", "cl", "t"],
            metadata=metadata,
            available_diphones={"a-t", "t-t", "a-cl", "cl-t"},
        )
        literal = sp.resolve_special_phone_sequence(
            ["a", "cl_literal", "t"],
            metadata=metadata,
            available_diphones={"a-t", "t-t", "a-cl", "cl-t"},
        )

        self.assertEqual(
            structural.render_phones, ("a", "t", "t")
        )
        self.assertEqual(literal.render_phones, ("a", "cl", "t"))

    def test_external_voice_and_q_remain_literal(self):
        result = sp.resolve_special_phone_sequence(
            ["a", "q", "a", "cl", "t"],
            metadata={},
        )

        self.assertEqual(result.render_phones, ("a", "q", "a", "cl", "t"))

    def test_generic_third_party_metadata_does_not_enable_structural_cl(self):
        result = sp.resolve_special_phone_sequence(
            ["a", "cl", "t", "a"],
            metadata={
                "builder_version": "third-party-7",
                "configuration_id": "external-profile",
            },
        )

        self.assertEqual(result.render_phones, ("a", "cl", "t", "a"))
        self.assertEqual(result.realizations[0].status, "literal")

    def test_generated_structural_cl_requires_readable_inventory(self):
        result = sp.resolve_special_phone_sequence(
            ["a", "cl", "t", "a"],
            metadata={"context_model": "oto_directional_v1"},
        )

        self.assertEqual(result.render_phones, ("a", "cl", "t", "a"))
        self.assertEqual(result.unresolved[0].status,
                         "inventory_unavailable")

    def test_builder_policy_requires_an_explicit_literal_opt_in(self):
        default = sp.generated_voice_policy()
        literal = sp.generated_voice_policy(
            sp.parse_special_phone_mode_specs(["cl=literal"])
        )

        self.assertEqual(
            default["phones"]["cl"]["mode"],
            sp.STRUCTURAL_CL_MODE,
        )
        self.assertEqual(
            literal["phones"]["cl"]["mode"],
            sp.STRUCTURAL_CL_MODE,
        )
        self.assertEqual(
            literal["literal_phone_mappings"]["cl_literal"]
            ["source_phone"],
            "cl",
        )
        explicit = sp.generated_voice_policy(
            literal_phone_mappings={"literal_cl": "cl"}
        )
        self.assertEqual(
            explicit["literal_phone_mappings"]["literal_cl"]
            ["source_phone"],
            "cl",
        )
        self.assertEqual(
            sp.parse_literal_phone_map_specs(["literal_cl=cl"]),
            {"literal_cl": "cl"},
        )
        with self.assertRaisesRegex(ValueError, "PHONE=MODE"):
            sp.parse_special_phone_mode_specs(["literal"])
        with self.assertRaisesRegex(ValueError, "unsupported realization"):
            sp.parse_special_phone_mode_specs(["q=anticipatory_consonant"])
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            sp.parse_literal_phone_map_specs(["cl=cl"])


if __name__ == "__main__":
    unittest.main()
