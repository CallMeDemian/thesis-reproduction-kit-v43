# LLM final Plan-3 contract

The active models are GPT-5.4-mini and Gemini 3.1 Flash-Lite. Each model has 15
main and 6 stability cells, yielding 42 logical cells and 24,150 planned calls. Active
conditions are C4, C5, C4R, C6-E, and C6-EX. No superseded experimental conditions are part of the runtime.
C4R/C6-E/C6-EX branch from the same-run C4; replicate 2 uses a fresh C4.

C6-EX uses one frozen PCG64(seed=20260912) firm derangement shared by models and
budgets. It has zero self-donor collisions; action-label collisions were not optimized.
Free8 B1 permits normalized intensity at most 1+1e-9. BINF keeps axis bounds without a
total cap. DL and RF are mutually exclusive. Invalid completed outputs enter ITT as A0;
transport/provider infrastructure failures remain unresolved for retry. No clipping,
rescaling, projection, or action rewrite is allowed.

Primary contrasts are C5−C4, C4R−C4, C6-E−C4R, and C6-E−C6-EX, analyzed separately
by model and budget. Bootstrap uses 10,000 PCG64 resamples with seed 20260912.
