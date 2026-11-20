# RL–LLM interaction-aware adaptive guidance

Primary gates are 5-fold OOF Ridge models trained on Alpha only. Beta/Gamma reuse the same Alpha-trained gate decisions.

| Model | Oracle | Gate | Gate − always | 95% CI | Holm p (Alpha) | Reference rate |
|---|---|---|---:|---|---:|---:|
| google_gemini31flashlite | alpha | rl_only | +0.0025 | [-0.0012, +0.0081] | 1.0000 | 90.6% |
| google_gemini31flashlite | beta | rl_only | +0.0018 | [-0.0007, +0.0061] | — | 90.6% |
| google_gemini31flashlite | gamma | rl_only | +0.0014 | [-0.0010, +0.0039] | — | 90.6% |
| google_gemini31flashlite | alpha | llm_only | +0.0015 | [-0.0043, +0.0089] | 1.0000 | 96.5% |
| google_gemini31flashlite | beta | llm_only | -0.0008 | [-0.0036, +0.0013] | — | 96.5% |
| google_gemini31flashlite | gamma | llm_only | +0.0017 | [+0.0001, +0.0048] | — | 96.5% |
| google_gemini31flashlite | alpha | rl_llm_interaction | -0.0161 | [-0.0411, +0.0055] | 0.9720 | 75.7% |
| google_gemini31flashlite | beta | rl_llm_interaction | +0.0003 | [-0.0038, +0.0054] | — | 75.7% |
| google_gemini31flashlite | gamma | rl_llm_interaction | -0.0060 | [-0.0206, +0.0046] | — | 75.7% |
| openai_gpt54mini | alpha | rl_only | -0.0028 | [-0.0091, +0.0008] | 1.0000 | 93.6% |
| openai_gpt54mini | beta | rl_only | -0.0050 | [-0.0154, +0.0006] | — | 93.6% |
| openai_gpt54mini | gamma | rl_only | +0.0004 | [-0.0013, +0.0021] | — | 93.6% |
| openai_gpt54mini | alpha | llm_only | +0.0117 | [-0.0189, +0.0511] | 1.0000 | 79.7% |
| openai_gpt54mini | beta | llm_only | +0.0007 | [-0.0008, +0.0023] | — | 79.7% |
| openai_gpt54mini | gamma | llm_only | +0.0084 | [-0.0007, +0.0215] | — | 79.7% |
| openai_gpt54mini | alpha | rl_llm_interaction | +0.0029 | [-0.0226, +0.0284] | 1.0000 | 74.8% |
| openai_gpt54mini | beta | rl_llm_interaction | +0.0046 | [-0.0019, +0.0158] | — | 74.8% |
| openai_gpt54mini | gamma | rl_llm_interaction | +0.0028 | [-0.0028, +0.0106] | — | 74.8% |

A positive point estimate alone is insufficient: the primary criterion is a positive paired OOF CI and a multiplicity-adjusted result, with matching directions in Beta/Gamma.

## Final judgment

1. **Action difference is descriptively related to uplift, but not monotonically or consistently.** Gemini's largest-distance quintile has a smaller Alpha mean increment (+0.0357) than intermediate quintiles; GPT's near-match group has a smaller mean increment (+0.0057) than the mismatch group (+0.1607). This model-specific pattern does not establish a portable mechanism.
2. **RL confidence does not add demonstrated deployable value once C4 action information is available.** The combined interaction gate is worse for Gemini Alpha (−0.0161) and inconclusive for GPT Alpha (+0.0029, CI crossing zero).
3. **The interaction gate does not accurately select reference provision.** No Alpha primary comparison has a positive multiplicity-adjusted result; the interaction gate's reference rate also drops to about 75%, without policy-value support.
4. **Always Reference is not beaten.** The selected Alpha-trained gates have inconsistent Beta/Gamma signs and all relevant paired confidence intervals include zero.
5. **This is not a new AI technical contribution.** The study is a negative result: C4/RL geometric interaction features do not improve an already strong always-C6-E policy in OOF evaluation.

For both LLMs, approximately 88% of Alpha increments are exactly zero. The action tables still differ within those ties, so this reflects the current evaluator's score resolution rather than action identity. Value-network Advantage remains a separate unavailable auxiliary analysis, not a reason to change this conclusion.
