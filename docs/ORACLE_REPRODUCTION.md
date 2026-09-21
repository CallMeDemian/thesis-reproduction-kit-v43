# Oracle reproduction

The fresh Oracle path is connected to the production Stage0/Stage1
implementation. Its same-run manifests, artifact hashes, contracts, and
recursive lineage are required before downstream consumers can run.

Authorized raw inputs are required for a full fresh Oracle execution. If they
are unavailable or fail validation, the run stops with an explicit
`INPUT_REQUIRED` boundary and does not substitute frozen Oracle outputs as
fresh compute parents.

The synthetic acceptance command exercises the canonical Oracle adapter with
deterministic fixture data. It proves executable wiring and lineage only; it
is not thesis-scale raw-data Oracle evidence and does not relax the scientific
gate for a real fresh replication.
