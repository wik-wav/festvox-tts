# Japanese Support: Dependencies and Licensing

This inventory records the Phase 5 development environment and release
boundary. It is not legal advice. The application remains local-only and does
not bundle an Open JTalk dictionary, an HTS model, Festival, or UTAU voicebanks.

## Verified Local Components

| Component | Verified version | Use | License/notice | Distribution policy |
| --- | --- | --- | --- | --- |
| Python | 3.14 development runtime | application runtime | PSF License | Install separately. |
| NumPy | 2.4.2 | waveform/DSP arrays and bank-scale phone-center join conditioning | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` in installed metadata | Normal Python dependency; preserve notices when packaging. |
| PyQt5 | 5.15.11 | GUI | GPL v3 or commercial license | A release must choose GPL-compatible application terms or obtain commercial terms. |
| PyQt5-Qt5 | 5.15.2 | Qt runtime | LGPL v3 in installed metadata | Preserve Qt notices and LGPL obligations. |
| PyQt5-sip | 12.18.0 | Python/Qt bindings | BSD-2-Clause | Preserve notice. |
| pyqtgraph | 0.14.0 | waveform and editor plots | MIT | Preserve notice. |
| sounddevice | 0.5.5 | optional playback | MIT | Optional; absence must remain graceful. |
| pyopenjtalk | 0.4.1 | optional Japanese linguistic labels | MIT (Expat) | User-installed optional dependency; the app imports it lazily. |
| pyworld | 0.3.5 | FRQ-less source-speech F0 analysis for every builder | MIT wrapper around WORLD | User-installed build dependency; imported lazily only when FRQ is unavailable. |
| WORLD | pyworld 0.3.5 bundled core | Harvest/DIO F0 estimation and StoneMask refinement | Modified BSD; upstream states its algorithms are not patented | Preserve the WORLD notice with any packaged binary. |
| setuptools | 80.10.2 | compatibility provider for pyworld 0.3.5 `pkg_resources` import | MIT | Pin below 81 until pyworld removes that import. |
| Open JTalk | component in pyopenjtalk | pronunciation/full-context labels | Modified BSD notice | The app uses labels only and does not redistribute the component. |
| open_jtalk_dic_utf_8 | 1.11, installed with pyopenjtalk | morphology and pronunciation dictionary | Installed `COPYING` includes separate NAIST, UniDic Consortium, and Open JTalk BSD notices | **not bundled**; review and reproduce the exact archive notices before any redistribution. |
| HTS Voice Mei | installed inside pyopenjtalk | not used by this application | Creative Commons Attribution 3.0 | **not bundled** and never used as the speaker waveform. |
| Festival | 2.5.0-13 in WSL | UniSyn/TD-PSOLA rendering | University of Edinburgh permissive notice, with separately noted files/modules | User-installed external runtime; preserve package notices if redistributed. |
| Speech Tools | 2.5.0-14build1 in WSL | EST indexes and pitchmarks | University of Edinburgh notice plus listed third-party files | User-installed external runtime; preserve complete package notices. |
| festvox-kallpc16k | 2.5.0-1 in WSL | English reference voice | voice-specific terms in the installed Festival package | Never infer that another voice has the same terms. |

Primary origins: [pyopenjtalk](https://github.com/r9y9/pyopenjtalk),
[pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder),
[WORLD](https://github.com/mmorise/World),
[Open JTalk](https://open-jtalk.sourceforge.net/),
[Festival](https://www.cstr.ed.ac.uk/projects/festival/),
[NumPy](https://numpy.org/),
[PyQt](https://www.riverbankcomputing.com/software/pyqt/),
[pyqtgraph](https://pyqtgraph.readthedocs.io/), and
[sounddevice](https://python-sounddevice.readthedocs.io/).

The direct NumPy project is distributed under the modified BSD license; the
additional license identifiers above come from bundled or vendored components
reported by the installed wheel. The shared acoustic requirements explicitly
declare `numpy>=1.26` rather than relying on pyworld's transitive dependency.

## UTAU Voicebanks

UTAU voicebanks are user-supplied works with independent terms. Source banks
remain read-only and are never staged. Generated slices and Festival voices may
still be derivative material; they are not redistributable until the applicable
source-bank license explicitly permits that use. Unknown or absent bank terms
are a release blocker, not implied permission.

## Optional Baselines

- `openjtalk_labels` requests pronunciation and full-context labels only. It
  never calls `pyopenjtalk.tts`, `pyopenjtalk.synthesize`, or an HTS waveform.
- `external_hts` reads user-supplied phone durations and F0 targets from JSON.
  It neither loads nor copies a model or waveform.
- The selected UTAU/Festival waveform remains the only speaker waveform.

## Source F0 Estimation

Source pitchmark generation is language-independent. A valid UTAU FRQ contour
is authoritative. A recording without FRQ uses WORLD `Harvest` by default or
the explicitly selected `DIO`; both are refined with StoneMask and converted
to phase-aligned UniSyn epochs. WORLD analyzes the user's UTAU waveform only;
it does not synthesize or substitute a speaker waveform.

The exposed options are intentionally bounded:

- `Harvest` is the quality default for speech and the pyworld project
  recommends it over DIO when signal-to-noise ratio is low.
- `DIO` is the faster speech estimator and remains useful for large clean
  banks.
- pYIN is a strong probabilistic monophonic tracker, but enabling it would add
  the librosa/SciPy dependency stack and a different voiced-state model.
- CREPE is a neural monophonic tracker with pretrained weights and a larger
  TensorFlow/model packaging contract.
- RMVPE is designed for vocal pitch in polyphonic music, not clean isolated
  source speech, and likewise requires external neural weights/runtime.

pYIN, CREPE, and RMVPE are therefore documented research candidates, not
pretend menu choices. They may be added later only with deterministic output,
offline model provenance, licensing inventory, and regression fixtures.

## Current Release Blockers

1. The local vault has no declared project license.
2. PyQt5 release terms have not been selected.
3. No source-bank license has been approved for redistribution.
4. Dictionary/model notices have been inventoried but are not packaged.
5. Acoustic naturalness has not received human listening approval.

`japanese_release.py` checks that this inventory remains present and that no
Open JTalk dictionary directory or `.htsvoice` file has entered the repository.
Its implementation check can pass while `redistribution_ready` correctly stays
false.
