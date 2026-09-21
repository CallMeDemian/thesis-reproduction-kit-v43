# LLM reproduction

The final Plan-3 design contains 42 matrix cells for each of two reasoning
arms and 575 firms per arm:

- BASELINE: 24,150 logical requests.
- HIGH: 24,150 logical requests.
- Total: 48,300 logical requests.

Provider execution is disabled by default. A paid run must explicitly use
`python -m thesis_repro reproduce --full --live-llm`; `--resume` continues the
same provider jobs by their persisted provider identities. A live run may
return `EXTERNAL_WAIT` while an asynchronous provider job remains pending.
Final PASS requires the authoritative generation ledger to report
`GENERATION_COMPLETE` independently for both BASELINE and HIGH.

The exact frozen C6-EX donor permutation is bundled in the certified
distribution and restored automatically after SHA-256 validation. Fresh
C6-EX materialization combines that historical donor relation with fresh C3-E
action labels; it does not copy historical C6-EX results.

Strict and Repaired are evaluation policies over the generation evidence, not
independent provider-generation campaigns. Provider credentials, paid calls,
and asynchronous completion remain external requirements.
