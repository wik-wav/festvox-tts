# Japanese UTAU Integration: Phase 5

> **Stable-workflow correction:** pitch-bank and voice-color metadata and
> routing APIs remain for experiments, but their GUI controls were removed.
> `create_edited_plan()` ignores stored routing settings unless the caller
> explicitly passes `allow_experimental_routing=True`. Normal use builds each
> pitch or color as a separate generated voice configuration.
>
> **Prosody correction:** remediation plan schema 2 supersedes the original
> fixed phone-duration and multiplier contour. It uses mora-first timing with
> OTO safety diagnostics and a speaker-relative phrase/accent contour. Question
> rise is supplied by general Intonation blocks, never by this structural plan.

Phase 5 adds conservative quality and source-routing refinements around the
committed Japanese Festival path. The default output remains the Phase 3
structural plan. Every experiment is optional, manual occurrence choices stay
final, and the English path is untouched.

## Acoustic Diagnostics

`japanese_quality.py` reads short windows from generated voice copies. For each
adjacent selected-unit join it reports:

- endpoint amplitude discontinuity;
- RMS-level mismatch in dB;
- normalized first-difference (roughness) mismatch;
- zero-crossing-rate mismatch;
- a voiced-only period mismatch in cents;
- clipping ratio;
- a transparent bounded risk score and `good`, `review`, or `poor` rating.

The score is a triage signal, not a naturalness score. Missing geometry and
measurement failures stay visible as diagnostics. Reports contain only paths
relative to the generated voice. `JapaneseQualityCache` stores timestamp-free,
content-addressed JSON outside source banks and reuses unchanged measurements.

## Optional Baselines

`japanese_refinements.py` defines a small `JapaneseBaselineProvider` interface:

- `structural` is the normal deterministic accent/duration model;
- `openjtalk_labels` reanalyzes full-context labels when pyopenjtalk is present,
  but does not call waveform synthesis;
- `external_hts` reads an explicit Japanese phone/duration/F0 JSON trajectory.

Unavailable providers return diagnostics and leave the structural baseline
active. Phone-sequence mismatches are rejected. Manual accent/phrase edits keep
their structural F0 over an optional provider, per-mora offsets are applied
afterward, and the continuous Pitch curve remains the final authority.
Provider durations apply to a fresh Generate. Re-render keeps manually edited
phone boundaries while still applying the selected provider's F0 baseline.

## Multipitch and Voice Color

Generated runtime metadata now advertises profile subbanks, tone ranges,
available colors, and the selected build color. Older Phase 3 metadata remains
valid and simply has no routable subbanks.

`route_dynamic_candidates()` evaluates only candidates already compiled for the
target diphone. Dynamic pitch uses the final generated F0 at that edge and the
nearest declared source note/tone-range center. Voice color is an exact subbank
filter. Ties use selection cost, stable candidate ID, and left-unit name. A
missing requested color retains ordinary candidates and emits a warning; it
never drops an alias. Existing `unit_overrides` are skipped unconditionally, so
manual per-occurrence choices remain final.

The Japanese Speech parameter page exposes **Baseline**, **Dynamic pitch bank**,
and **Voice color**. Pitch/color controls enable only when the selected generated
voice advertises matching metadata. The settings persist in `japanese_state`,
participate in undo/redo, and require re-render rather than a source-bank write.

## Listening and Release Tools

`japanese_listening_set.py` now contains 16 cases, adding stressed joins, low and
high dynamic-pitch routing, and declared voice color. Each rendered row includes
join-diagnostic counts; WAVs, manifests, and quality caches remain ignored test
output. Acoustic naturalness is explicitly unverified.

The corrective Stage 6 corpus keeps 16 stable-workflow cases but replaces the
three experimental routing rows with devoiced vowels, multiple-accent-phrase
downstep, and long-phrase declination. It records the contour model,
speaker range, normalized Intonation blocks, and complete mora timing rows.

`japanese_release.py` inventories installed Python dependencies, checks the
license document, and rejects bundled Open JTalk dictionary/HTS voice assets.
It distinguishes passing implementation checks from legal redistribution
readiness. See `JAPANESE_DEPENDENCIES_AND_LICENSES.md` and
`JAPANESE_RELEASE_CHECKLIST.md`.

Typical path-neutral checks are:

```text
py -3.14 japanese_quality.py <generated-voice> <plan.json> --output <report.json>
py -3.14 japanese_release.py . JAPANESE_DEPENDENCIES_AND_LICENSES.md
```

## Safety and Compatibility

- No analyzer, metric, provider, or router writes inside a source UTAU bank.
- No Open JTalk/HTS waveform replaces the selected UTAU speaker.
- No dictionary, model, voice, cache, or listening WAV is committed.
- Optional dependencies degrade gracefully.
- Runtime extensions are additive; old Japanese voices and projects load.
- CMU, ARPAbet, ARPAsing, English OTO conversion, English phonesets, and English
  Festival entry points are unchanged.

## Verification

Phase 5 adds deterministic suites for join scoring/cache behavior, optional
provider fallback and no-waveform guarantees, dynamic routing/manual authority,
release-report determinism, runtime metadata, project state, undo/redo, and GUI
control enablement. The final complete repository gate contains 223 passing
tests with no failures or skips.

A real representative CVVC voice compiled 2,001 edge units and rendered all 16
listening examples twice. The complete output trees were byte-identical. The
automated join report found no `poor` joins, but this does not certify acoustic
naturalness. Full commands, counts, hashes, and visible fallback diagnostics are
recorded in `JAPANESE_IMPLEMENTATION_REPORT.md`.

Human listening remains the final quality gate. Automated metrics can locate
risky joins, but they cannot verify Japanese naturalness or speaker identity.
