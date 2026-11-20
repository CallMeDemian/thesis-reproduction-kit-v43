# RL internal-signal audit

## Evidence used

- 28 frozen `actor_critic_outputs.parquet` files: 4 configurations × 7 seeds.
- Each member contained exactly 575 firms and aligned nine-action actor probabilities and Q values. All probability vectors summed to one within tolerance.
- The primary outcome table was reconstructed from the frozen HIGH-run raw responses using the fixed Stage-6 Alpha/Beta/Gamma evaluator, in an analysis-only runtime. The joined primary sample is exactly 575 firms for each LLM model and Oracle.

## Feature construction

- Actor features: mean member entropy, ensemble predictive entropy, Jensen–Shannon disagreement, vote entropy, consensus, maximum probability, and top-1/top-2 probability margin.
- Critic features: member-and-firm standardized Q values, selected-action Q, selected-vs-best Q, top-two Q margin, and actor–critic agreement. Raw Q values were not averaged across reward configurations.
- `V(s)` was absent from all 28 retained diagnostic files. The checkpoints are present, but the model-runtime package needed for a checkpoint-compatible value extraction is not part of this standalone bundle. No synthetic value or proxy Advantage was used.

## Scope and integrity notes

- Outcome: `g = delta_R_score(C6-E) − delta_R_score(C4R)`.
- Primary filter: HIGH release, STRICT ITT, FREE8, IC-b, B1, Run1, separately by GPT/Gemini and Alpha/Beta/Gamma.
- The official evaluator's registry status tag was stale despite `final_result_allowed: true`; only an analysis-copy metadata tag was normalized. Oracle parameters, fitted artifacts, frozen responses, actions, and source evidence were not changed.
- Gamma scoring was resumed with Python 3.12 and scikit-learn 1.8, matching the serialized model's version, after Python 3.14 could not load its private `_loss` module.
