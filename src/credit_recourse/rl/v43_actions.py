"""Frozen nine-class action representation and factual semantic projection."""
from __future__ import annotations
import numpy as np
import pandas as pd

from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS, KEYS
from credit_recourse.rl.common.actions import ActionSpace, project_actions_to_candidates
from credit_recourse.rl.pipelines.final_stage2_raw_action_source_precompute.semantic_v4 import materialize_semantic_v4_history
from credit_recourse.rl.v43_features import canonical_keys


class V43ActionCodec:
    def __init__(self, contract):
        self.contract = contract
        raw = contract.action_contract
        self.columns = tuple(raw["action_columns"])
        self.vectors = np.array([[raw["fixed_candidates"][a][d] for d in self.columns] for a in ACTION_IDS], dtype=np.float64)
        scales = np.array([raw["normalization_scale_by_dimension"][d.removeprefix("action__")] for d in self.columns])
        intensities = np.abs(self.vectors / scales).sum(axis=1)
        if not np.allclose(intensities, [0]+[1]*8, rtol=0, atol=1e-12):
            raise ValueError("Frozen action nominal intensity changed")
        self.representations = np.concatenate([np.eye(9), (self.vectors != 0).astype(float), np.sign(self.vectors)], axis=1).astype(np.float32)

    def encode(self, class_ids):
        ids = np.asarray(class_ids)
        if ids.dtype.kind not in "iu" or (ids < 0).any() or (ids >= 9).any():
            raise ValueError("Action representation requires observed frozen IDs 0..8; missing is not noop")
        return self.representations[ids]

    def decision_frame(self, keys, class_ids):
        """Serialize fixed candidate decisions for the Stage6 financial seam."""
        keys = canonical_keys(keys)
        ids = np.asarray(class_ids)
        self.encode(ids)  # Strict integer IDs; missing is not A0.
        if ids.shape != (len(keys),):
            raise ValueError("One action ID is required for each decision state")
        frame = keys.copy()
        frame["row_id"] = np.arange(len(keys), dtype=np.int64)
        frame["policy"] = "RL_V43"
        frame["candidate_id"] = [ACTION_IDS[i] for i in ids]
        frame["selected_candidate_id"] = frame.candidate_id
        frame["action_class_id"] = ids
        for j, column in enumerate(self.columns):
            frame[column] = self.vectors[ids, j]
        frame["nominal_intensity"] = (ids != 0).astype(float)
        frame["action_contract_sha256"] = self.contract.action_contract_sha256
        return frame

    def space(self):
        c = self.contract.action_contract
        return ActionSpace(columns=list(self.columns), bounds={k: tuple(v) for k, v in c["action_bounds"].items()},
                           fixed_candidates=c["fixed_candidates"], train_labels=list(ACTION_IDS),
                           row_conditional_baselines=[], final_rl_label="RL_V43", scenario_candidates={}, diagnostic_candidates={},
                           candidate_library_hash=self.contract.action_contract_sha256,
                           final_action_contract_hash=self.contract.action_contract_sha256,
                           candidate_action_contract_version=c["candidate_action_contract_version"],
                           intensity_scale_by_column={d: c["normalization_scale_by_dimension"][d.removeprefix("action__")] for d in self.columns},
                           family_by_candidate=c["family_by_candidate"])

    def ledger(self):
        rows = []
        for i, action in enumerate(ACTION_IDS):
            row = {"candidate_id": action, "class_id": i, "nominal_intensity": 0.0 if i == 0 else 1.0}
            for j, dim in enumerate(self.columns):
                row[dim] = float(self.vectors[i, j])
                row["mask__"+dim] = bool(self.vectors[i, j] != 0)
                row["direction__"+dim] = int(np.sign(self.vectors[i, j]))
            rows.append(row)
        return pd.DataFrame(rows)

    def project_observed(self, canonical_accounts, frozen_operating_actions, *, decision_year_max=None):
        """Use existing V4 observation definitions and frozen L1/P50 geometry.

        Factual labels are only transition supervision. They never enter state
        assembly. No old 11-class label or independent-debt/PPE action is read.
        """
        keys = canonical_keys(canonical_accounts)
        fields = {"capex": "capex", "depreciation": "depreciation", "short_term_debt": "short_debt",
                  "long_term_debt": "long_debt", "current_portion_long_debt": "current_portion_long_debt", "bonds": "bond"}
        current = pd.concat([keys, canonical_accounts[list(fields)].reset_index(drop=True)], axis=1).rename(columns=fields)
        following = current.copy()
        following["fiscal_year"] -= 1
        following = following.rename(columns={c: c+"__next" for c in fields.values()})
        merged = current.merge(following, on=list(KEYS), how="left", validate="one_to_one")
        if decision_year_max is not None:
            merged = merged.loc[merged.fiscal_year <= decision_year_max].reset_index(drop=True)
        semantic, metadata = materialize_semantic_v4_history(merged)
        observed = merged[list(KEYS)].copy()
        for dim in self.columns[:3]:
            suffix = dim.removeprefix("action__")
            observed[dim] = semantic["v4_actual_action__"+suffix]
            observed["action_observed__"+suffix] = semantic["v4_actual_observed__"+suffix]
        cols = [*KEYS, *self.columns[3:], *[d.replace("action__", "action_observed__", 1) for d in self.columns[3:]]]
        operating = frozen_operating_actions.loc[:, cols].copy()
        operating[list(KEYS)] = canonical_keys(operating)
        observed = observed.merge(operating, on=list(KEYS), how="left", validate="one_to_one")
        for dim in self.columns:
            mask = dim.replace("action__", "action_observed__", 1)
            observed[mask] = observed[mask].fillna(False).astype(bool)
            if (observed[mask] & ~np.isfinite(pd.to_numeric(observed[dim], errors="coerce"))).any():
                raise ValueError("Observed action contains missing/nonfinite values: " + dim)
            observed.loc[~observed[mask], dim] = np.nan
        projected = project_actions_to_candidates(observed, self.space(), projection_mode="l1_best", require_observed_action_masks=True)
        projected["action_class_id"] = projected.candidate_id.map({a: i for i, a in enumerate(ACTION_IDS)}).astype("Int64")
        projected["action_contract_sha256"] = self.contract.action_contract_sha256
        projected["action_label_role"] = "factual nearest-archetype supervision; not claimed executed candidate"
        return projected, metadata
