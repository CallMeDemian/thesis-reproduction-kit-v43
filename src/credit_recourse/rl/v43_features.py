"""The sole V4.3 decision-state feature producer and training preprocessing seam."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from credit_recourse.contracts.account_registry import ACCOUNT_REGISTRY
from credit_recourse.rl.contracts.v43_encoder import (
    ACTION_IDS, ASSET_FIELDS, CATEGORICAL_COLUMNS, DEBT_FIELDS, HISTORY_FEATURES,
    KEYS, RATIOS, REVENUE_FIELDS, V43EncoderContract, content_hash,
)
from credit_recourse.simulator.business_plan import calibrate_business_plan_non_rate
from credit_recourse.simulator.executed_primitives import capex_baseline
from credit_recourse.simulator.historical_source import firm_key, state_from_record


def canonical_keys(frame):
    result = frame.loc[:, list(KEYS)].copy()
    if result.isna().any().any():
        raise ValueError("Missing firm/year key")
    result["firm_id"] = result.firm_id.map(firm_key)
    years = pd.to_numeric(result.fiscal_year, errors="raise")
    if not np.isfinite(years).all() or (years != years.astype("int64")).any():
        raise ValueError("Invalid decision year")
    result["fiscal_year"] = years.astype("int64")
    if result.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate firm/year decision keys")
    return result.reset_index(drop=True)


def key_hash(keys):
    keys = canonical_keys(keys).sort_values(list(KEYS))
    return content_hash([[str(f), int(y)] for f, y in keys.itertuples(index=False, name=None)])


def ratio(numerator, denominator):
    n, d = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast_shapes(n.shape, d.shape), np.nan)
    np.divide(n, d, out=out, where=np.isfinite(n) & np.isfinite(d) & (d > 0))
    return np.where(np.isfinite(out), out, np.nan)


def _positive_coverage(available, requested):
    if not np.isfinite(available) or not np.isfinite(requested):
        return np.nan
    return min(max(available, 0.0) / requested, 1.0) if requested > 0 else 0.0


@dataclass
class FeatureBatch:
    keys: pd.DataFrame
    continuous: np.ndarray
    missing: np.ndarray
    categorical: np.ndarray
    schema_hash: str
    statistics_hash: str

    @property
    def feature_hash(self):
        h = hashlib.sha256()
        h.update(self.keys.to_csv(index=False, lineterminator="\n").encode())
        for array in (self.continuous, self.missing, self.categorical):
            h.update(str(array.shape).encode())
            h.update(str(array.dtype).encode())
            h.update(np.ascontiguousarray(array).tobytes())
        h.update(self.schema_hash.encode())
        h.update(self.statistics_hash.encode())
        return h.hexdigest()



def validate_statistics(contract, statistics):
    stats = dict(statistics)
    claimed = stats.pop("statistics_hash", None)
    if claimed != content_hash(stats) or stats.get("schema_hash") != contract.schema_hash:
        raise ValueError("V43 preprocessing identity mismatch")
    if stats.get("columns") != list(contract.continuous_columns) or stats.get("fit_role") != "RL_TRAIN":
        raise ValueError("Invalid V43 preprocessing schema/role")
    if stats.get("fit_rows", 0) < 1 or stats.get("fit_year_max", 9999) > 2022 or stats.get("fit_outcome_year_max", 9999) > 2023:
        raise ValueError("V43 preprocessing is not fitted on the eligible transition population")
    n = len(contract.specs)
    center, scale, counts = (np.asarray(stats[k]) for k in ("center", "scale", "observed_RL"))
    if any(v.shape != (n,) for v in (center, scale, counts)):
        raise ValueError("V43 preprocessing vector shape mismatch")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid V43 numeric statistics")
    if (counts < 0).any() or (counts > stats["fit_rows"]).any():
        raise ValueError("Invalid training coverage counts")
    for name in CATEGORICAL_COLUMNS:
        vocab = stats["vocabulary"][name]
        if sorted(vocab.values()) != list(range(2, len(vocab)+2)):
            raise ValueError("Invalid V43 categorical vocabulary")
    return stats


class V43FeatureProducer:
    """Accept canonical accounts, as-of context and the frozen V4 rate ledger only.

    No action labels, reward columns, next states, or simulated financial results
    are accepted by this boundary. Multiple requests can reuse one panel index.
    """
    @classmethod
    def from_project_root(cls, root, *, source_kind="canonical", source_year_max=None, target_accounts=None, rate_rows_override=None):
        from credit_recourse.simulator.historical_source import read_financial_panel
        from credit_recourse.rl.contracts.v43_encoder import RATE_PATH, PURE_INTEREST_PATH, file_sha256
        root = Path(root)
        # Stage2 inputs are resolved at call time so a fresh run can bind the
        # exact same producer to its run-local input_source namespace.  The
        # legacy deployed root remains a compatibility fallback only.
        from credit_recourse.rl.v43_one_pass_data import input_root as stage2_input_root
        input_root = stage2_input_root(root).resolve()
        paths = {
            "canonical": input_root / "input_splits/canonical_business_plan_history.parquet",
            "historical": input_root / "runtime_inputs/historical_financial_context_v4_3/actual_financial_states.parquet",
        }
        if source_kind not in paths:
            raise ValueError("Unregistered V43 financial source")
        accounts, lineage = read_financial_panel(paths[source_kind])
        if source_year_max is not None:
            accounts = accounts.loc[accounts.fiscal_year <= source_year_max].reset_index(drop=True)
        if target_accounts is not None:
            target_accounts = target_accounts.copy()
            if (target_accounts.fiscal_year > 2023).any():
                raise ValueError("Evaluation outcome in target supplement")
            target_accounts["source__target_only"] = True
            accounts["source__target_only"] = False
            accounts = pd.concat([accounts,target_accounts],ignore_index=True)
            canonical_keys(accounts)
        interest_path = input_root / "v4_3_runtime/01_contract/financial_cost_sources_r2/history_financial_cost_sources.parquet"
        interest = pd.read_parquet(interest_path, columns=[*KEYS, "U01B550010000", "interest_source_conflict"])
        interest[list(KEYS)] = canonical_keys(interest)
        values = pd.to_numeric(interest["U01B550010000"], errors="raise")
        if (interest.interest_source_conflict.fillna(False).astype(bool) & values.notna()).any():
            raise ValueError("Conflicted primary interest source has a numeric value")
        lookup = pd.Series(values.to_numpy(dtype=float), index=pd.MultiIndex.from_frame(interest[list(KEYS)]))
        accounts["pure_interest_expense"] = lookup.reindex(pd.MultiIndex.from_frame(accounts[list(KEYS)])).to_numpy()
        accounts["source__pure_interest_expense"] = np.where(accounts.pure_interest_expense.notna(), "U01B550010000", None)
        lineage["pure_interest_expense"] = {
            "precedence": ["U01B550010000"], "source_path": str(interest_path.relative_to(root)),
            "source_sha256": file_sha256(interest_path), "join": "firm_id,fiscal_year exact; no imputed or shifted years",
            "selected_column_counts": {"U01B550010000": int(accounts.pure_interest_expense.notna().sum())},
            "unobserved_count": int(accounts.pure_interest_expense.isna().sum()), "broad_cost_fallback": False}

        context = pd.read_parquet(input_root / "runtime_inputs/historical_financial_context_v4_3/actual_context.parquet",
                                  columns=[*KEYS, "sector_7"]).rename(columns={"sector_7": "sector"})
        rates = pd.read_parquet(root / RATE_PATH) if rate_rows_override is None else rate_rows_override.copy()
        if source_year_max is not None:
            context = context.loc[context.fiscal_year <= source_year_max].reset_index(drop=True)
            rates = rates.loc[rates.base_year <= source_year_max].reset_index(drop=True)
        result = cls(V43EncoderContract.from_project_root(root), accounts, context, rates)
        result.account_lineage = lineage
        return result

    def __init__(self, contract, accounts, context, rate_rows, *, valid_provider_years=()):
        self.contract: V43EncoderContract = contract
        allowed = set(KEYS) | set(ACCOUNT_REGISTRY) | {"canonical_business_plan_history_row_id"}
        extra = [str(c) for c in accounts if c not in allowed and not str(c).startswith("source__")]
        if extra:
            raise ValueError("Noncanonical account inputs at V43 feature boundary: " + repr(extra[:12]))
        keys = canonical_keys(accounts)
        self.accounts = keys.copy()
        for field in ACCOUNT_REGISTRY:
            values = pd.to_numeric(accounts[field], errors="raise").to_numpy(dtype=float) if field in accounts else np.full(len(keys), np.nan)
            self.accounts[field] = np.where(np.isfinite(values), values, np.nan)
        self.accounts["source__target_only"] = accounts.get("source__target_only", pd.Series(False,index=accounts.index)).fillna(False).to_numpy(dtype=bool)
        self.accounts = self.accounts.sort_values(list(KEYS)).reset_index(drop=True)
        self._target_only = self.accounts.pop("source__target_only").to_numpy(dtype=bool)
        self._positions = {key: i for i, key in enumerate(self.accounts[list(KEYS)].itertuples(index=False, name=None))}
        self._groups = {str(f): np.asarray(i) for f, i in self.accounts.groupby("firm_id", sort=False).indices.items()}
        self._years = self.accounts.fiscal_year.to_numpy()
        self._base = self._base_features()
        if set(context) - set(KEYS) - {"sector"}:
            raise ValueError("Context must be projected to firm/year/sector before production")
        ckeys = canonical_keys(context)
        self._sector = {key: value for key, value in zip(ckeys.itertuples(index=False, name=None), context.sector)}
        required_rate = {"firm_id", "base_year", "selected_rate", "selection_source", "rate_short", "rate_long", "rate_bond", "ceiling"}
        if not required_rate.issubset(rate_rows):
            raise ValueError("Incomplete frozen V4 rate ledger")
        rate_keys = canonical_keys(rate_rows.rename(columns={"base_year": "fiscal_year"}))
        self._rates = {}
        valid_years = sorted(set(map(int, valid_provider_years)))
        for key, row in zip(rate_keys.itertuples(index=False, name=None), rate_rows.to_dict("records")):
            if bool(row.get("future_observation_used", False)):
                raise ValueError("Frozen rate ledger reports a future observation")
            value = float(row["selected_rate"])
            if not np.isfinite(value) or not 0 < value <= float(row["ceiling"]):
                raise ValueError("Invalid frozen BP rate")
            if any(float(row[c]) != value for c in ("rate_short", "rate_long", "rate_bond")):
                raise ValueError("Artificial spread in frozen V4 rate ledger")
            source = str(row["selection_source"])
            year = int(key[1])
            if source == "provider_observed_valid":
                source_year = year
            elif source == "external_contemporaneous_initialization_fallback":
                source_year = None
            else:
                eligible = [y for y in valid_years if y <= year]
                source_year = max(eligible) if eligible else None
            if source_year is not None and source_year > year:
                raise ValueError("Future BP-rate source")
            if "source_observation_year" in row:
                raw_year = row["source_observation_year"]
                source_year = int(raw_year) if pd.notna(raw_year) else None
                if source_year is not None and source_year > year:
                    raise ValueError("Future authoritative BP-rate source year")
            self._rates[key] = (value, source, year-source_year if source_year is not None else np.nan)
        self._bp_cache = {}

    def _base_features(self):
        a = self.accounts
        out = {}
        for field, name in (("total_assets", "log_assets"), ("revenue", "log_revenue")):
            x = a[field].to_numpy()
            out[name] = np.sign(x) * np.log1p(np.abs(x))
        for field in ASSET_FIELDS:
            out[field + "_to_assets"] = ratio(a[field], a.total_assets)
        for field in REVENUE_FIELDS:
            out["broad_financial_cost_to_revenue" if field == "financial_cost" else field + "_to_revenue"] = ratio(a[field], a.revenue)
        for name, (n, d) in RATIOS.items():
            out[name] = ratio(a[n], a[d])
        debt = a[list(DEBT_FIELDS)].sum(axis=1, min_count=4)
        out["total_debt_to_assets"] = ratio(debt, a.total_assets)
        out["gross_ltd_to_assets"] = ratio(a.long_term_debt + a.current_portion_long_debt, a.total_assets)
        for field in (*DEBT_FIELDS, "pure_interest_expense", "operating_cf"):
            out[field + "_to_total_debt"] = ratio(a[field], debt)
        out["net_debt_to_assets"] = ratio(debt - a.cash, a.total_assets)
        out["net_working_capital_to_assets"] = ratio(a.receivables + a.inventory - a.payables, a.total_assets)
        return pd.DataFrame(out)

    def _capex(self, position, previous):
        if position in self._bp_cache:
            return self._bp_cache[position]
        row = self.accounts.iloc[position]
        history = self.accounts.iloc[previous[-3:]]
        # The frozen plan may have explicit defaults; these are not observations.
        # Require observed ratios for the encoder's CAPEX capability block.
        observed_capex = np.isfinite(ratio(history.capex, history.revenue)).any()
        observed_dep = np.isfinite(ratio(history.depreciation, history.ppe)).any()
        parts = {"bp_capex": np.nan, "maintenance_capex": np.nan, "bp_growth_capex": np.nan}
        if observed_capex and np.isfinite(row.revenue):
            states = [state_from_record(r) for r in history.to_dict("records")]
            plan = calibrate_business_plan_non_rate(states)
            current = states[-1]
            calculated = capex_baseline(current, plan)
            parts["bp_capex"] = calculated["bp_capex"]
            if observed_dep and np.isfinite(row.ppe):
                parts = calculated
        self._bp_cache[position] = parts
        return parts

    def _ability(self, position, previous):
        row = self.accounts.iloc[position]
        base = self._base.iloc[position]
        assets = float(row.total_assets)
        debt_values = row[list(DEBT_FIELDS)].to_numpy(dtype=float)
        debt = float(debt_values.sum()) if np.isfinite(debt_values).all() else np.nan
        c = self.contract.action_contract
        vector = c["fixed_candidates"]
        reserve_ratio = float(c["operators"]["DL"]["liquidity_reserve_cash_to_cogs"])
        reserve = float(np.max([0.0, reserve_ratio * row.cogs, row.cash-row.current_assets]))
        available = float(np.maximum(row.cash-reserve, 0.0))
        requested_dl = debt * float(vector["DL"]["action__deleveraging_total_debt_pct"])
        capex = self._capex(position, previous)
        raw = {
            "opening_reserve_to_assets": ratio(reserve, assets).item(),
            "opening_repayment_cash_to_assets": ratio(available, assets).item(),
            "dl_requested_principal_to_assets": ratio(requested_dl, assets).item(),
            "dl_cash_coverage": _positive_coverage(available, requested_dl),
            "rf_transferable_short_to_assets": ratio(np.maximum(row.short_term_debt, 0) * vector["RF"]["action__refinancing_short_debt_pct"], assets).item(),
            "near_maturity_share": ratio(row.short_term_debt+row.current_portion_long_debt, debt).item(),
            "bp_capex_to_assets": ratio(capex["bp_capex"], assets).item(),
            "maintenance_capex_to_assets": ratio(capex["maintenance_capex"], assets).item(),
            "discretionary_capex_to_assets": ratio(capex["bp_growth_capex"], assets).item(),
            "cx_discretionary_share": ratio(capex["bp_growth_capex"], capex["bp_capex"]).item(),
            "ar_stock_to_assets": ratio(np.maximum(row.receivables, 0), assets).item(),
            "inventory_stock_to_assets": ratio(np.maximum(row.inventory, 0), assets).item(),
            "ap_stock_to_assets": ratio(np.maximum(row.payables, 0), assets).item(),
            "ap_turnover_headroom": float(np.maximum(base.ap_turnover - 0.5, 0)),
            "oe_cost_base_to_assets": ratio(np.maximum(row.cogs, 0)+np.maximum(row.sga, 0), assets).item(),
        }
        for candidate in ACTION_IDS[1:]:
            fractions = []
            for column, delta in vector[candidate].items():
                if not column.startswith("action__") or float(delta) == 0:
                    continue
                dimension = column.removeprefix("action__")
                if dimension == "deleveraging_total_debt_pct":
                    frac = _positive_coverage(available, float(delta) * debt)
                elif dimension == "refinancing_short_debt_pct":
                    frac = float(row.short_term_debt > 0 and np.all(debt_values >= 0)) if np.isfinite(debt_values).all() else np.nan
                elif dimension == "growth_capex_reduction_pct":
                    growth = capex["bp_growth_capex"]
                    frac = float(growth > 0) if np.isfinite(growth) else np.nan
                elif dimension in ("inv_turnover_chg", "ar_turnover_chg"):
                    stock, numerator = (row.inventory, row.cogs) if dimension.startswith("inv") else (row.receivables, row.revenue)
                    frac = float(stock > 0 and numerator > 0) if np.isfinite([stock, numerator]).all() else np.nan
                elif dimension == "ap_turnover_chg":
                    frac = _positive_coverage(raw["ap_turnover_headroom"], abs(float(delta)))
                elif dimension in ("cogs_ratio_chg", "sga_ratio_chg"):
                    name = dimension.removesuffix("_ratio_chg") + "_to_revenue"
                    frac = _positive_coverage(float(base[name]), abs(float(delta)))
                else:
                    raise ValueError("Unknown frozen action dimension: " + dimension)
                fractions.append(frac)
            raw[candidate + "__joint_fraction"] = float(np.min(fractions))
        return {"ability__" + name: value for name, value in raw.items()}

    def build(self, requested_keys):
        keys = canonical_keys(requested_keys)
        records, histories = [], []
        for firm, year in keys.itertuples(index=False, name=None):
            position = self._positions.get((firm, year))
            if position is None:
                raise KeyError("Missing authoritative decision state: " + repr((firm, year)))
            group = self._groups[firm]
            previous = group[(self._years[group] <= year) & (~self._target_only[group] | (group == position))]
            if not len(previous) or self._years[previous[-1]] != year:
                raise ValueError("History does not end at decision year")
            record = self._base.iloc[position].to_dict()
            for name in HISTORY_FEATURES:
                values = self._base[name].to_numpy()[previous]
                valid = np.flatnonzero(np.isfinite(values))[-3:]
                vals = values[valid]
                yrs = self._years[previous[valid]]
                prefix = "history__" + name + "__"
                record[prefix + "mean3"] = float(np.mean(vals)) if len(vals) else np.nan
                record[prefix + "std3"] = float(np.std(vals, ddof=0)) if len(vals) >= 2 else np.nan
                record[prefix + "change_per_year"] = float((vals[-1]-vals[-2])/(yrs[-1]-yrs[-2])) if len(vals) >= 2 else np.nan
                record[prefix + "count3"] = float(len(vals))
                record[prefix + "age"] = float(year-yrs[-1]) if len(vals) else np.nan
                histories.append({"firm_id": firm, "fiscal_year": year, "feature": name,
                                  "source_years": ",".join(map(str, yrs)), "count": len(vals),
                                  "latest_source_year": int(yrs[-1]) if len(vals) else None})
            rate, source, age = self._rates.get((firm, year), (np.nan, None, np.nan))
            record.update(bp_rate=rate, rate_source_age=age, decision_year=(year-2000)/25,
                          sector=self._sector.get((firm, year)), rate_source=source)
            record.update(self._ability(position, previous))
            records.append(record)
        frame = pd.DataFrame(records, columns=[*self.contract.continuous_columns, *CATEGORICAL_COLUMNS])
        frame[list(self.contract.continuous_columns)] = frame[list(self.contract.continuous_columns)].astype("float64")
        for name in CATEGORICAL_COLUMNS:
            frame[name] = frame[name].astype(pd.StringDtype(na_value=np.nan))
        expected = set(self.contract.continuous_columns) | set(CATEGORICAL_COLUMNS)
        if records and set(records[0]) != expected:
            raise ValueError("Producer and frozen feature dictionary diverged")
        return pd.concat([keys, frame], axis=1), pd.DataFrame(histories)

    def _fixed_transform(self, frame):
        x = frame.loc[:, list(self.contract.continuous_columns)].to_numpy(dtype=np.float64, copy=True)
        for j, spec in enumerate(self.contract.specs):
            if spec.transform == "asinh":
                x[:, j] = np.arcsinh(x[:, j])
            elif spec.transform != "identity":
                raise ValueError("Unregistered fixed transform")
        x[~np.isfinite(x)] = np.nan
        return x

    def fit_rl(self, training_frame, authoritative_training_rows):
        from credit_recourse.rl.v43_one_pass_contract import assert_training_rows
        time_audit = assert_training_rows(authoritative_training_rows)
        actual = canonical_keys(training_frame)
        expected = canonical_keys(authoritative_training_rows)
        if key_hash(actual) != key_hash(expected):
            raise ValueError("Statistics require the exact eligible training key union")
        x = self._fixed_transform(training_frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            center = np.nanmedian(x, axis=0)
            quartiles = np.nanquantile(x, [.25, .75], axis=0)
        count = np.isfinite(x).sum(axis=0)
        iqr = quartiles[1] - quartiles[0]
        center = np.where(count > 0, center, 0.0)
        scale = np.where(np.isfinite(iqr) & (iqr > 1e-6), iqr, 1.0)
        vocab = {name: {value: i+2 for i, value in enumerate(sorted(set(training_frame[name].dropna().astype(str))))} for name in CATEGORICAL_COLUMNS}
        stats = {"schema_hash": self.contract.schema_hash, "fit_key_hash": key_hash(expected),
                 "fit_rows": len(actual), "fit_year_max": time_audit["decision_year_max"],
                 "fit_outcome_year_max": time_audit["outcome_year_max"], "fit_role": "RL_TRAIN",
                 "evaluation_rows_used": 0, "columns": list(self.contract.continuous_columns),
                 "center": center.tolist(), "scale": scale.tolist(), "observed_RL": count.tolist(), "vocabulary": vocab}
        return {**stats, "statistics_hash": content_hash(stats)}

    def transform_frame(self, frame, statistics):
        validate_statistics(self.contract, statistics)
        stats = dict(statistics)
        claimed = stats.pop("statistics_hash", None)
        if claimed != content_hash(stats) or stats["schema_hash"] != self.contract.schema_hash:
            raise ValueError("V43 preprocessing identity mismatch")
        if stats["columns"] != list(self.contract.continuous_columns) or stats["fit_role"] != "RL_TRAIN" or stats["fit_year_max"] > 2022:
            raise ValueError("Invalid V43 preprocessing fit population/schema")
        x = self._fixed_transform(frame)
        missing = ~np.isfinite(x)
        normalized = (x-np.asarray(stats["center"])) / np.asarray(stats["scale"])
        # Zero is an explicit tensor placeholder ONLY. The raw frame stays NaN.
        normalized[missing] = 0.0
        normalized = normalized.astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("Non-finite float32 encoding; no tail clipping is permitted")
        categorical = np.zeros((len(frame), len(CATEGORICAL_COLUMNS)), dtype=np.int64)
        for j, name in enumerate(CATEGORICAL_COLUMNS):
            categorical[:, j] = [0 if pd.isna(v) else stats["vocabulary"][name].get(str(v), 1) for v in frame[name]]
        return FeatureBatch(canonical_keys(frame), normalized, missing.astype(bool), categorical,
                            self.contract.schema_hash, claimed)

    def transform(self, keys, statistics, *, batch_size=1000):
        keys = canonical_keys(keys)
        if batch_size < 1:
            raise ValueError("Feature batch_size must be positive")
        if len(keys) <= batch_size:
            frame, _ = self.build(keys)
            return self.transform_frame(frame, statistics)
        batches = []
        for start in range(0, len(keys), batch_size):
            frame, _ = self.build(keys.iloc[start:start+batch_size])
            batches.append(self.transform_frame(frame, statistics))
            if (start+batch_size) % 5000 == 0:
                print("V43 features", min(start+batch_size, len(keys)), "/", len(keys), flush=True)
        return FeatureBatch(keys,
                            np.concatenate([b.continuous for b in batches]),
                            np.concatenate([b.missing for b in batches]),
                            np.concatenate([b.categorical for b in batches]),
                            self.contract.schema_hash, statistics["statistics_hash"])
