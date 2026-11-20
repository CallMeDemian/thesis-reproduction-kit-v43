# RL internal-signal adaptive-reference gate

This is an exploratory, firm-level 5-fold nested-CV analysis. The gate uses only frozen RL features; C4R/C6-E scores are outcomes, never inputs.
Q values are z-standardized within each member and firm before aggregation. No V(s) was retained, so no Q−V/Advantage claim is made.

| Model | Oracle | Gate | Features | Always RL increment | Gate − always | 95% CI | Holm p (alpha) | Reference share |
|---|---|---|---|---:|---:|---|---:|---:|
| google_gemini31flashlite | alpha | threshold:entropy | actor | +0.0897 | -0.0152 | [-0.0354, +0.0016] | 1.0000 | 92.9% |
| google_gemini31flashlite | alpha | threshold:agreement | actor | +0.0897 | -0.0054 | [-0.0241, +0.0062] | 1.0000 | 92.2% |
| google_gemini31flashlite | alpha | threshold:q_margin | q | +0.0897 | -0.0059 | [-0.0220, +0.0077] | 1.0000 | 91.1% |
| google_gemini31flashlite | alpha | ridge | actor | +0.0897 | -0.0114 | [-0.0330, +0.0037] | 1.0000 | 90.6% |
| google_gemini31flashlite | alpha | ridge | q | +0.0897 | +0.0026 | [+0.0000, +0.0078] | 1.0000 | 96.3% |
| google_gemini31flashlite | alpha | ridge | combined | +0.0897 | +0.0015 | [-0.0028, +0.0074] | 1.0000 | 89.2% |
| google_gemini31flashlite | alpha | logistic | actor | +0.0897 | -0.0897 | [-0.1411, -0.0422] | 0.0000 | 0.0% |
| google_gemini31flashlite | alpha | logistic | q | +0.0897 | -0.0897 | [-0.1411, -0.0422] | 0.0000 | 0.0% |
| google_gemini31flashlite | alpha | logistic | combined | +0.0897 | -0.0897 | [-0.1411, -0.0422] | 0.0000 | 0.0% |
| google_gemini31flashlite | beta | threshold:entropy | actor | +0.0156 | -0.0013 | [-0.0038, +0.0008] | — | 79.0% |
| google_gemini31flashlite | beta | threshold:agreement | actor | +0.0156 | -0.0011 | [-0.0032, +0.0009] | — | 67.8% |
| google_gemini31flashlite | beta | threshold:q_margin | q | +0.0156 | -0.0017 | [-0.0035, -0.0004] | — | 91.5% |
| google_gemini31flashlite | beta | ridge | actor | +0.0156 | -0.0008 | [-0.0024, +0.0007] | — | 81.2% |
| google_gemini31flashlite | beta | ridge | q | +0.0156 | +0.0006 | [-0.0028, +0.0051] | — | 86.1% |
| google_gemini31flashlite | beta | ridge | combined | +0.0156 | +0.0006 | [-0.0029, +0.0052] | — | 80.0% |
| google_gemini31flashlite | beta | logistic | actor | +0.0156 | -0.0040 | [-0.0093, +0.0018] | — | 35.3% |
| google_gemini31flashlite | beta | logistic | q | +0.0156 | -0.0046 | [-0.0101, +0.0014] | — | 33.7% |
| google_gemini31flashlite | beta | logistic | combined | +0.0156 | -0.0049 | [-0.0108, +0.0012] | — | 34.6% |
| google_gemini31flashlite | gamma | threshold:entropy | actor | +0.0222 | +0.0011 | [-0.0049, +0.0074] | — | 93.7% |
| google_gemini31flashlite | gamma | threshold:agreement | actor | +0.0222 | -0.0068 | [-0.0152, +0.0011] | — | 85.4% |
| google_gemini31flashlite | gamma | threshold:q_margin | q | +0.0222 | +0.0007 | [-0.0017, +0.0034] | — | 91.8% |
| google_gemini31flashlite | gamma | ridge | actor | +0.0222 | -0.0029 | [-0.0074, +0.0000] | — | 96.9% |
| google_gemini31flashlite | gamma | ridge | q | +0.0222 | -0.0001 | [-0.0015, +0.0007] | — | 95.1% |
| google_gemini31flashlite | gamma | ridge | combined | +0.0222 | -0.0056 | [-0.0125, -0.0001] | — | 91.7% |
| google_gemini31flashlite | gamma | logistic | actor | +0.0222 | -0.0222 | [-0.0441, -0.0019] | — | 0.0% |
| google_gemini31flashlite | gamma | logistic | q | +0.0222 | -0.0222 | [-0.0441, -0.0019] | — | 0.0% |
| google_gemini31flashlite | gamma | logistic | combined | +0.0222 | -0.0222 | [-0.0441, -0.0019] | — | 0.0% |
| openai_gpt54mini | alpha | threshold:entropy | actor | +0.1003 | -0.0001 | [-0.0007, +0.0004] | 1.0000 | 93.2% |
| openai_gpt54mini | alpha | threshold:agreement | actor | +0.1003 | -0.0147 | [-0.0475, +0.0243] | 1.0000 | 87.1% |
| openai_gpt54mini | alpha | threshold:q_margin | q | +0.1003 | -0.0013 | [-0.0038, +0.0000] | 1.0000 | 94.3% |
| openai_gpt54mini | alpha | ridge | actor | +0.1003 | -0.0030 | [-0.0091, +0.0000] | 1.0000 | 96.7% |
| openai_gpt54mini | alpha | ridge | q | +0.1003 | +0.0007 | [-0.0150, +0.0172] | 1.0000 | 93.6% |
| openai_gpt54mini | alpha | ridge | combined | +0.1003 | +0.0029 | [-0.0086, +0.0172] | 1.0000 | 93.0% |
| openai_gpt54mini | alpha | logistic | actor | +0.1003 | -0.1003 | [-0.1617, -0.0366] | 0.0480 | 0.0% |
| openai_gpt54mini | alpha | logistic | q | +0.1003 | -0.1003 | [-0.1617, -0.0366] | 0.0480 | 0.0% |
| openai_gpt54mini | alpha | logistic | combined | +0.1003 | -0.1003 | [-0.1617, -0.0366] | 0.0480 | 0.0% |
| openai_gpt54mini | beta | threshold:entropy | actor | +0.0231 | -0.0113 | [-0.0268, -0.0007] | — | 83.5% |
| openai_gpt54mini | beta | threshold:agreement | actor | +0.0231 | -0.0098 | [-0.0225, -0.0027] | — | 61.2% |
| openai_gpt54mini | beta | threshold:q_margin | q | +0.0231 | -0.0063 | [-0.0101, -0.0025] | — | 56.3% |
| openai_gpt54mini | beta | ridge | actor | +0.0231 | -0.0067 | [-0.0175, -0.0006] | — | 85.0% |
| openai_gpt54mini | beta | ridge | q | +0.0231 | -0.0021 | [-0.0148, +0.0076] | — | 84.9% |
| openai_gpt54mini | beta | ridge | combined | +0.0231 | -0.0055 | [-0.0166, +0.0013] | — | 80.0% |
| openai_gpt54mini | beta | logistic | actor | +0.0231 | -0.0231 | [-0.0472, -0.0008] | — | 0.0% |
| openai_gpt54mini | beta | logistic | q | +0.0231 | -0.0231 | [-0.0472, -0.0008] | — | 0.0% |
| openai_gpt54mini | beta | logistic | combined | +0.0231 | -0.0231 | [-0.0472, -0.0008] | — | 0.0% |
| openai_gpt54mini | gamma | threshold:entropy | actor | +0.0196 | -0.0012 | [-0.0029, +0.0001] | — | 93.2% |
| openai_gpt54mini | gamma | threshold:agreement | actor | +0.0196 | -0.0025 | [-0.0081, +0.0022] | — | 87.7% |
| openai_gpt54mini | gamma | threshold:q_margin | q | +0.0196 | +0.0000 | [-0.0009, +0.0013] | — | 94.1% |
| openai_gpt54mini | gamma | ridge | actor | +0.0196 | -0.0069 | [-0.0167, -0.0007] | — | 83.7% |
| openai_gpt54mini | gamma | ridge | q | +0.0196 | +0.0014 | [-0.0007, +0.0044] | — | 89.4% |
| openai_gpt54mini | gamma | ridge | combined | +0.0196 | -0.0004 | [-0.0047, +0.0039] | — | 77.2% |
| openai_gpt54mini | gamma | logistic | actor | +0.0196 | -0.0196 | [-0.0431, +0.0042] | — | 0.0% |
| openai_gpt54mini | gamma | logistic | q | +0.0196 | -0.0196 | [-0.0431, +0.0042] | — | 0.0% |
| openai_gpt54mini | gamma | logistic | combined | +0.0196 | -0.0196 | [-0.0431, +0.0042] | — | 0.0% |

Primary inference is the adaptive-minus-always paired, firm-resampled confidence interval. Beta and gamma are robustness analyses; alpha p-values are Holm-adjusted across all alpha gate comparisons.
A positive raw association alone is not evidence of a deployable gate; it must improve OOF policy value over always providing C3-E.

## Interpretation

The evidence does **not** support deploying an adaptive reference gate. The best Alpha point estimates were small (Gemini Q-only Ridge: +0.0026; GPT combined Ridge: +0.0029), but their family-adjusted inference did not support an improvement and the GPT interval crosses zero. Results are not robust across Beta and Gamma; several gates reduce policy value.

The largest unadjusted feature associations occur for Beta (`q_selected_relative`: Spearman 0.159 for Gemini and 0.164 for GPT), but these are explanatory signals rather than successful out-of-fold routing evidence. Entropy, vote agreement, and Q features therefore provide at most partial descriptive support for H1, not H2–H4.

Direct answers: (Q1) not demonstrated; (Q2) relative Q features have the strongest descriptive relationship, while Advantage is unavailable; (Q3) no reliable gain over always providing C3-E; (Q4) no—effects do not persist convincingly across Oracles; (Q5) this is not ready as a new technical contribution. It should be reported as a negative/exploratory result, with a checkpoint-compatible V(s) extraction and an independent-year validation required before revisiting the method.
