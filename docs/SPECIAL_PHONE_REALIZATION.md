# Special-phone realization

## Purpose

Text frontends and the Phonemes input mode use one canonical phone sequence.
Some canonical symbols own editable timing without naming the source-bank
recording that should be selected for that interval. The runtime therefore
keeps two aligned views:

- `display_phones`: linguistic and editable phones shown by the GUI;
- `render_phones`: source-selection phones sent to Festival/UniSyn or the
  bundled Python renderer.

The views always have the same number of phones. Segment timing, F0, voicing,
vocal-tract controls, persistence, and Undo/Redo stay attached to the display
view. Recording selection and diphone lookup use the render view.

## Structural `cl`

Generated UTAU voices use `cl=anticipatory_consonant` by default in every
language. For example:

```text
display: i cl s o
render:  i s  s o
```

The first `s` anticipates the following consonant and gives the editable `cl`
interval a real VC/pre-consonant source. The generated `s-s` unit is a bounded
hold cut from the consonant portion of a normal `s-V` recording. It excludes
the vowel release; the following `s-o` transition remains the only release.
This prevents `cl` from becoming either digital silence or a doubled complete
consonant.

The same resolver is used for Text, direct Phonemes input, Generate, Re-render,
project reload, English, Asaxi, and Japanese. A manually entered `cl` therefore
has exactly the same behavior as one produced by a language frontend.

An orphan `cl` with no following consonant keeps its editable interval but
uses silence as a visible fallback. A generated bank that lacks the required
`V-C` and `C-C` source edges is rejected with a rebuild diagnostic; the runtime
does not silently fall back to an OTO alias named `cl`.

## Explicit literal `/cl/`

An OTO entry named `cl`, `V cl`, or `cl V` is inventory evidence only. Its
presence never changes the default structural behavior. This remains true even
when a bank contains complete incoming and outgoing `cl` aliases.

A voice creator must deliberately expose a genuine linguistic `/cl/` under a
distinct canonical token:

```powershell
py -3.14 build_festival_voice.py ... `
  --literal-phone-map cl_literal=cl
```

Both meanings are then usable:

```text
cl          structural anticipatory hold
cl_literal  authored source-bank /cl/
```

The mapping option is repeatable. `--special-phone-mode cl=literal` remains a
compatibility shorthand for `--literal-phone-map cl_literal=cl`; it does not
turn off structural `cl`. The declaration is recorded in:

- `dic/diphone_index.json`;
- `dic/unit_alternatives.json`;
- `dic/voice_manifest.json`.

For `cl_literal=cl`, the builder requires at least one authored, non-silence
incoming `X-cl` unit and one authored, non-silence outgoing `cl-X` unit. It
stops with an actionable error when either side is missing or when the chosen
display token collides with another source phone. Merely naming a
silence/calibration recording `cl` does not satisfy the check.

Built-in Festival voices and unknown third-party voices retain their own
literal behavior unless they explicitly publish this generated-voice policy.
Older voices made by this builder inherit structural `cl` only when their
metadata contains an unambiguous converter marker; if their required `C-C`
hold is absent, the GUI asks for a rebuild instead of guessing.

## Other special phones

The policy registry is language-neutral:

- `pau`, `sil`, and `sp` are literal boundary/silence symbols;
- `q` is currently literal;
- `cl` is structural by default for generated UTAU voices.

Future structural phones must be added to this registry and resolved through
the same aligned display/render contract. A coincidental OTO alias must never
activate structural or literal semantics automatically.

## Selection and editing safety

Automatic contextual selection runs against the resolved render pairs.
Manual per-occurrence recording choices remain final. When a phone edit changes
a structural source pair, overrides are remapped by unchanged source-diphone
identity; a stale choice is dropped rather than applied to a different
consonant. The GUI relabels returned source segments back to canonical display
phones and fails visibly if Festival returns a sequence that cannot be aligned.

Builders only read source UTAU banks. Structural holds, policy metadata, WAV
copies, indexes, and manifests are written under the generated output.
