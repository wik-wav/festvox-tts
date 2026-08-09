# Japanese Support Release Checklist

## Automated Gate

1. Run every repository test program.
2. Run `japanese_release.py` against the repository root and
   `JAPANESE_DEPENDENCIES_AND_LICENSES.md`.
3. Generate candidate/runtime metadata twice and compare bytes.
4. Verify all source-bank manifests before and after tests.
5. Confirm no source bank, generated WAV, cache, dictionary, model, private
   absolute path, or listening output is staged.

## Licensing Gate

1. Choose and record a project license compatible with the selected PyQt5
   terms, or obtain commercial PyQt terms.
2. Capture the exact Python wheel/sdist notices for every packaged dependency.
3. If pyopenjtalk assets are redistributed, include the exact dictionary
   `COPYING`, Open JTalk notice, and HTS Voice Mei attribution as applicable.
4. Record each source UTAU bank's license and derivative/redistribution terms.
5. Do not ship a generated voice whose source terms are missing or ambiguous.

## Acoustic Gate

1. Render the ignored Phase 5 listening corpus.
2. Review every `poor` and `review` join diagnostic against the source audio.
3. Listen to ordinary, long-vowel, nasal, geminate, palatalized, phrase,
   question, multipitch, and voice-color examples.
4. Verify manual unit and continuous F0 edits remain final.
5. Record the human reviewer and result; automated tests do not establish
   naturalness.

## Packaging Gate

1. Test a clean machine with pyopenjtalk absent.
2. Test an installation with pyopenjtalk present and its dictionary initialized.
3. Test Festival/WSL absence and report it without corrupting projects.
4. Keep all generated voices, listening WAVs, and quality caches outside the
   package or in ignored output directories.
5. Regenerate the dependency/version inventory for the actual release build.
