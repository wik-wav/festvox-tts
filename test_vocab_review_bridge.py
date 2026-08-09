# -*- coding: utf-8 -*-
"""Headless Vocab Forge bridge project-row tests."""

from __future__ import annotations

import io
import unittest

import numpy as np

import vocab_review_bridge as bridge

fc = bridge.fc


class VocabReviewBridgeTests(unittest.TestCase):
    def test_request_reader_decodes_asaxi_as_utf8_not_console_encoding(self):
        payload = bridge._read_request(io.BytesIO(
            '{"text":"xi fůjå ma."}'.encode("utf-8")
        ))

        self.assertEqual(payload["text"], "xi fůjå ma.")

    def test_project_row_preserves_asaxi_prosody_and_idiom_words(self):
        syn = fc.Synthesis(
            np.zeros(160, np.float32),
            16000,
            [fc.Segment("g", 0.0, 0.01)],
            text="ga vi",
            lang="asaxi",
            voicebank="fixture",
            phones=["g"],
            render_phones=["g"],
            asaxi_prosody={
                "schema_version": 1,
                "word_count": 2,
                "phrases": [{
                    "words": [{
                        "surface": "ga",
                        "phrase_expression": "ga vi",
                    }, {
                        "surface": "vi",
                        "phrase_expression": "ga vi",
                    }],
                }],
            },
        )
        row = bridge._project_row(syn, {
            "text": "ga vi",
            "voicebank": "fixture",
            "speed": 1.0,
            "pitch_hz": 164.0,
            "fall_pct": 18.0,
        })

        self.assertEqual(row["lang_code"], "asaxi")
        self.assertEqual(row["asaxi_prosody"]["word_count"], 2)
        self.assertEqual(
            row["asaxi_prosody"]["phrases"][0]["words"][1][
                "phrase_expression"
            ],
            "ga vi",
        )
        self.assertEqual(row["cache_wav"], "cache/sentence_0001.wav")


if __name__ == "__main__":
    unittest.main()
