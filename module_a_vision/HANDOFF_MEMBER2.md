# Handoff note: Member 1 (Vision) → Member 2 (Edge SoC & LLM)

Date: 2026-11-22 · Branch: `feat/m1-vision-obb`

## TL;DR

The next vision model (`vision-v2`, training now) adds placement defects. Member 1
added deterministic fallback rules for them in **your** file
`edge_llm_storage/dual_advisor.py`. Please review that change; nothing else in
your territory was touched, and no schema/config changed.

## 1. New defect `type` strings you will start receiving

All arrive with `category: "SMT_PLACEMENT"` (an existing enum value, so the
schema is unchanged). `type` is free text in the schema.

| `type` | Severity | Meaning | New? |
|---|---|---|---|
| `TOMBSTONE` | HIGH | Chip lifted onto one end (unequal pad wetting) | new from the model (was only in the stub/synth generator) |
| `ORIENTATION_ERROR` | **CRITICAL** | Flipped chip (underside up) or reversed polarity (diode/cathode) | **brand new string** |
| `SHIFTED_COMPONENT` | HIGH | Part translated or rotated off its pads | existing, now far more frequent (adds a new data source) |

Existing `vision-v1` types are unchanged.

## 2. What changed in `edge_llm_storage/dual_advisor.py`

Only new `elif` branches were added after the existing ones; existing rules and
their order are untouched. They only affect the heuristic fallback (no LLM loaded).

| Branch | `TOMBSTONE` | `ORIENTATION_ERROR` | `SHIFTED_COMPONENT` |
|---|---|---|---|
| `_heuristic_dfm` | equal land patterns, equalize copper / thermal relief | pin-1 / cathode silkscreen marking, check CAD footprint rotation | (no DFM rule, it's a process issue) |
| `_heuristic_assembler` | longer reflow soak, balanced paste volume | feeder tape orientation, part rotation/polarity in placement program, quarantine reel | re-teach P&P vision offsets, placement pressure, board support |

All messages are ≤ 38 words (`_enforce_word_limit`). `pytest tests/` passes.

## 3. What we'd like from you

1. **Review/own the new advisor rules.** Reword or move them as you like.
2. **LLM prompts:** `configs/prompts.yaml` passes defects generically via
   `{current_defects}`, so the LLM path works without changes. Optionally add
   tombstone / polarity examples to the prompt hints, but `configs/` is
   schema-locked, so that needs the usual 3-member agreement.
3. **Telemetry / `recurring_defect_counts`:** expect these keys to start appearing.
4. **Severity:** `ORIENTATION_ERROR` is `CRITICAL`. If your gating treats
   `CRITICAL` as an automatic line stop, confirm that's what you want for flipped
   resistors (functionally harmless) as well as reversed diodes (functional failure).

## 4. Member 3 (UI), for reference

The Stage-2 placement gate in `module_b_guidance/app_ui.py` already catches
these via `category == "SMT_PLACEMENT"`; no UI change is required.
