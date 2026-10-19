# E1 protocol: reference-content control experiment

Status: EXPLORATORY (designed after the primary results were known). Fixed on 2026-10-20 (KST),
before any provider call. Nothing in this file, in `scripts/e1_build_requests.py`,
`scripts/e1_analyze.py` or in the request manifest and shards changes after the first provider call.

Base commit of the kit: `071848881349c8675bb4e15193a747fa02679986` (`hb-fy24-final`). Branch:
`ablation-20261102`. All E1 files live under `ablation_20261102/E1/`. `frozen/` is not modified.

## 1. Purpose

Contrast A3 (C6-E minus C4R) mixes two things: showing a reference program at all, and the learned,
firm-specific action content of that reference. E1 separates them by adding a condition that shows a
fixed, unlearned action (OE) in exactly the same format.

## 2. Cell

Free-text action space (free8), information condition IC-b, reasoning regime High, budget B1,
replicate 1, Strict ITT. All 575 firms (fiscal year 2024), both models, three conditions:
575 x 2 x 3 = 3,450 requests.

## 3. Conditions

| Condition | Prompt | Reference action |
| --- | --- | --- |
| E1-C4R | the September C4R request, unchanged | none |
| E1-RL | the September C6-E request, unchanged | the firm's own C3-E action |
| E1-OE | the September C6-E request with only the reference action JSON replaced | OE for every firm |

All three receive the same frozen C4 draft (High, replicate 1) of the same firm and model. The
instructions, the reference-block wording and the output format are identical. No condition says
where the reference comes from. E1-C4R and E1-RL differ from the September bodies only in the request
id (equal as canonical JSON). For the 232 firms whose own C3-E action is OE, the E1-OE and E1-RL
prompts are identical character for character (464 prompts); these form the placebo set.

C4 is a fixed input and is not rerun. C4R and C6-E are rerun in the same batch as E1-OE, so all three
conditions share run time and model version. This is the one exception to the rule that new
experiments rerun C4 as well, and it is recorded here.

## 4. Models and settings

Copied from the September High bodies (`HIGH_REASONING_PROTOCOL.json`), not rebuilt from the Baseline
provider code:

- OpenAI `gpt-5.4-mini-2026-03-17`, Responses Batch API, `reasoning.effort = high`, 32,768 output tokens.
- Google `gemini-3.1-flash-lite`, Gemini Batch API, `thinkingLevel = high`, 32,768 output tokens.

A model that is unavailable, or a returned model version that differs from
`expected_returned_model_version`, stops the run. No model is substituted.

## 5. Requests, manifest, shards

- Builder: `scripts/e1_build_requests.py`. Output: `requests/E1_request_manifest.csv`,
  `requests/E1_build_report.json`, `requests/batches/*.jsonl`.
- Manifest SHA-256: `1f12775443db3a929cddf7a619a27882ad49fe5b189da23572f908854064357a`.
- Order: shuffled with seed 20261020; shards of at most 500 per provider; each shard mixes the three
  conditions. Four shards per provider, eight in total.

| File | Requests | SHA-256 |
| --- | --- | --- |
| google_e1_shard_00001.jsonl | 500 | `b925540c3a8f4425b8f0e532c07333e8fdefbe972ffcd2f78f35c5f9e80a8307` |
| google_e1_shard_00002.jsonl | 500 | `d592fcfca9cd2eda4ded5990589973112fe9b418f5e1c3087bbbf198047c2a88` |
| google_e1_shard_00003.jsonl | 500 | `5df43212bbc6d3731724fad0338acb5d42860dcb507df2ab93ed453ef3630ef6` |
| google_e1_shard_00004.jsonl | 225 | `8d08eca2d894f290a425121d84e5c614b5a2d38983bbdf89c9644077a2d8f329` |
| openai_e1_shard_00001.jsonl | 500 | `a94813354990649f4cbb54a7857ec4a1abe45b71110c9e9111ccde6e6a85f35b` |
| openai_e1_shard_00002.jsonl | 500 | `3cf4742bc5d51b7a3ed46b04fa07874757b5471e2e2b7304d6d9333ddb6ca58c` |
| openai_e1_shard_00003.jsonl | 500 | `14cf26f2557e49062f3003ce02f8efe8ab489f839e6270c2b37a9f0d2d5d3301` |
| openai_e1_shard_00004.jsonl | 225 | `79a3074bee040d799aa14095ff480f1e318ab6153206d62ee5e76e7dc666d167` |

Build checks (all PASS in `E1_build_report.json`): cell inventory; own C3-E action per firm (575/575,
OE 232, RF 155, WC1 54, CX 41, WC2 39, MX2 36, MX1 18); the 2,300 September C4R/C6-E bodies equal the
ledger payloads; the renderer reproduces all 2,300 September prompts; E1-OE differs only in the
reference JSON; settings equal the High protocol; `frozen/` untouched. Python 3.12.10 with the kit's
pinned package versions.

## 6. Strict ITT and collection

- Strict ITT exactly as the kit's `apply_itt_rule`: a completed response that fails validation is run as
  A0; an infrastructure failure (no completed response) is retried until completion and is never
  turned into A0.
- Before every submission or resubmission, existing provider jobs are listed by display name
  (`e1-<provider>-shard<k>-attempt<n>`), so no shard runs twice.
- At most 3 attempts per request. The lowest-numbered delivered attempt is used; later duplicates are
  logged, not used. Resubmissions contain only undelivered requests, with identical bodies.
- Gemini batch creation may return 429 (quota); the same shard is resubmitted after a wait. A Gemini
  job pending or running for more than 48 hours expires without results; an OpenAI batch past its
  24-hour window returns completed work and cancels the rest. Every batch id is logged in
  `runs/submissions.json`; collection events are logged in `runs/collector_log.jsonl`.
- Smoke test: at most 2 firms x 3 conditions x 2 models, synchronous calls, to check access, parsing and
  returned model. Logged under `smoke/`, never scored or analysed; the batch regenerates those firms.
- No score is looked at until every request is collected.

## 7. Gates before any E1 score is computed

1. Parse replay: the September outputs of the same cell (C4R and C6-E, both models; 2,300 rows) pass
   through the normalize, parse and `apply_itt_rule` path used for E1; `strict_usable` and
   `strict_itt_action_json` equal the frozen observations for all 2,300 rows. Synthetic refusal,
   truncated, empty and error responses map as `apply_itt_rule` specifies.
2. Scoring path: the September C4, C4R and C6-E rows of the cell (both models, 3,450 rows) pushed through
   the E1 scoring path reproduce every frozen `delta_R_score_alpha/beta/gamma` (atol = rtol = 1e-9).
3. `returned_model` equals `expected_returned_model_version` for every response.

## 8. Analysis contract

Script: `scripts/e1_analyze.py` (self-test PASS before registration). It stops if the firm set differs
from the manifest. Estimates are means of paired per-firm differences; bootstrap by firm, 10,000 draws,
percentile interval, p = min(1, 2 x (min(#draws <= 0, #draws >= 0) + 1) / 10,001), kit
`paired_bootstrap` with seed = `pcg64_seed(canonical_key_json(key))`. The key has the 12 registry
fields (action_space FREE8, analysis_id, budget B1, execution_policy STRICT, information_condition
IC-b, lhs, model GPT/GEMINI, oracle, population ITT, reasoning_regime HIGH, replicate RUN_E1, rhs).
analysis_id: `ABL_E1` (575 firms), `ABL_E1_343` (firms whose C3-E action is not OE), `ABL_E1_232`
(firms whose C3-E action is OE). E1 is never pooled with the frozen registry.

| Contrast | Definition | Firms | Test | Holm family |
| --- | --- | --- | --- | --- |
| E1a primary | E1-RL minus E1-OE | 343 | yes | PRIMARY, 6 (3 Oracles x 2 models) |
| E1b | E1-OE minus E1-C4R | 575 | yes | SECONDARY, 12 (E1b and E1c) |
| E1c | E1-RL minus E1-C4R | 575 | yes | SECONDARY, 12 |
| E1a placebo | E1-RL minus E1-OE | 232 | yes | PLACEBO GATE, 6 |
| E1a, all firms | E1-RL minus E1-OE | 575 | no (descriptive) | none |

Per firm, E1c = E1a + E1b. E1c is reported next to the frozen A3 values but not compared with them by
a test (the run date differs). Output: `results/T5_E1.csv` (30 rows), `results/E1_wording.json`.

Wording for E1a in the 343 firms, applied per model and fixed now (the Korean sentences in `e1_analyze.py` are the operative text; the table gives their meaning):

| Pattern of the 343-firm E1a | Sentence |
| --- | --- |
| alpha, beta, gamma all positive, all Holm p < .05 | The learned action content produced higher policy value than the fixed OE on all three evaluation models. |
| all three negative, all Holm p < .05 | The fixed OE produced higher policy value than the learned action content on all three evaluation models. |
| only alpha Holm p < .05 (positive), beta and gamma same sign | The gain from the learned action content was confirmed on alpha; beta and gamma agree in sign only. Alpha is the evaluation model used to select C3-E. |
| only alpha Holm p < .05 (positive), beta or gamma opposite sign | Confirmed on alpha only; it cannot be separated from the effect of selecting C3-E. |
| all Holm p >= .05 | No additional gain from the learned action content was found. |
| any other combination | Report each evaluation model as it is and state that results differed across evaluation models. |

## 9. Placebo gate and its resolution

The 232 firms whose C3-E action is OE receive identical E1-RL and E1-OE prompts, so E1a in this set
must show no difference. The gate passes if all six Holm-adjusted p values are >= .05. The script
reports `FAIL: HOLD` automatically if any is below .05. That flag is resolved as follows.

If any Holm p < .05 in the placebo family (E1-RL vs E1-OE, 232 firms), run this audit before
interpreting anything:

1. the 464 placebo request bodies and generation settings are hash-identical;
2. every response maps to its request id;
3. returned model versions;
4. shard and attempt distribution of RL vs OE within the 232 firms;
5. no duplicate attempt was used.

If the audit finds a defect: fix it and rerun the analysis; never change prompts or populations.
If the audit is clean: report E1a (343 firms) as descriptive only, labelled "위약 진단에서 차이 관찰",
with no confirmatory claim; E1b and E1c are reported unchanged. The audit is recorded in
`QA_REPORT.md` under Q19.

## 10. What E1 cannot answer

The contribution of the RL training method itself, and the effect of having selected C3-E on the 2024
Oracle-alpha evaluation. For this reason the verdict looks at the direction on beta and gamma as well
as alpha. E1 is not extended with further control conditions, and nothing is re-selected after seeing
its results.

## 11. Deadlines and stop rule

Analysis by Oct 25, 2026. If the data are not fully collected and quality-checked by the end of
Oct 26, 2026, E1 is left out of the materials for Nov 2 and finished afterwards. Results are
labelled EXPLORATORY and stay DRAFT until the quality checks pass.

## 12. No changes after the first provider call

This protocol, the two scripts, the manifest, the build report and the eight shard files are committed
and pushed before the first provider call and are not edited afterwards. Collection and scoring code
under `scripts/` added later does not alter requests or the analysis contract.

