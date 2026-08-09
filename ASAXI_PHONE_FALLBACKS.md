# Asaxi inventory-aware phone fallbacks

## Purpose

Asaxi's canonical G2P uses compact palatalized phones such as `hy`, `ky`, and
`ny`. Integrated ARPAsing banks often contain those phones before `a e i o u`
but not before ARPAsing-compatible Asaxi nuclei such as `ao`, `ax`, or `ih`.
The canonical G2P remains unchanged because a native compact diphone is the
preferred realization whenever the selected voice provides it.

Immediately before Festival receives utterance-local pronunciations,
`asaxi_phone_fallback.py` reads the selected generated voice's
`dic/diphone_index.json`. For a missing transition involving a compact
palatalized phone, it tries:

```text
Cy V  ->  C y y V
```

For example, canonical `b o w hy ao` becomes `b o w h y y ao` only when
`hy-ao` is absent and all affected replacement transitions are present.

## Safety rules

- The runtime uses exact diphone inventory evidence, not a bank-name or
  language-profile guess.
- The canonical sequence is retained when its transitions exist.
- A repair is accepted only when the preceding context and every replacement
  transition exist. Partial repairs are never emitted.
- All compact palatalized Asaxi onset classes use the same mechanism.
- Explicit project/user pronunciation overrides are final and are not
  rewritten.
- The adapter changes only the phone realization. It does not choose a take,
  alter contextual scoring, write to a source bank, or modify generated-bank
  files.
- If neither the canonical transition nor a complete fallback exists, the
  canonical phones remain visible and synthesis metadata carries an
  `asaxi_phone_fallback_unavailable` warning.

## Metadata

`Synthesis.asaxi_prosody` records:

- `phone_fallback_model_id`;
- every canonical phone expansion;
- the missing canonical diphone;
- every replacement diphone validated against the bank;
- an informational or warning diagnostic.

The current model identifier is `asaxi-inventory-fallback-v1`.

## Tests

`test_asaxi_phone_fallback.py` covers compact-transition preservation,
successful and incomplete fallbacks, explicit-override protection, metadata
parsing, and every declared palatalized onset class. The Festival core test
also verifies that the adapted sequence reaches utterance-local addenda,
duration planning, and F0 alignment.

