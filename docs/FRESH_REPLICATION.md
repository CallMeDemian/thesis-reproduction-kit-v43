# Fresh replication

Fresh runs are namespaced by `runs/<run_id>`. The orchestration DAG is explicit, stage manifests record parent hashes, and `--resume` reuses only matching contract-complete stages.

