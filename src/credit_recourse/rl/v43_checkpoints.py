"""Strict Stage3 -> Stage4 -> Stage5 -> Stage6 checkpoint lineage."""
from pathlib import Path
import pyarrow.parquet  # Initialize Arrow before Torch on Windows.
import torch
from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS,SCHEMA_VERSION,file_sha256
from credit_recourse.rl.v43_actions import V43ActionCodec
from credit_recourse.rl.v43_model import V43Policy,encoder_payload,load_encoder_payload

def policy_payload(policy, *, stage, upstream_path, trained):
    if stage not in (4,5):
        raise ValueError("Only Stage4/5 produce policy checkpoints")
    upstream=torch.load(upstream_path,map_location="cpu",weights_only=False)
    expected="Stage3" if stage==4 else "Stage4"
    actual=str(upstream.get("producer_stage",""))
    if not (actual==expected or (not trained and actual==expected+"_schema_dry_run")):
        raise ValueError("Wrong predecessor stage for policy checkpoint")
    if trained and not upstream.get("trained"):
        raise ValueError("Trained policy requires a trained predecessor checkpoint")
    parent_encoder=upstream if stage==4 else upstream["encoder"]
    c=policy.encoder.contract
    return {
        "schema_version":SCHEMA_VERSION,"schema_hash":c.schema_hash,
        "statistics_hash":policy.encoder.statistics["statistics_hash"],
        "producer_stage":"Stage"+str(stage),"trained":bool(trained),
        "checkpoint_role":"trained_policy" if trained else "untrained_schema_fixture",
        "action_ids":list(ACTION_IDS),"action_contract_sha256":c.action_contract_sha256,
        "encoder":encoder_payload(policy.encoder,trained=bool(parent_encoder.get("trained")),producer_stage="Stage3"),
        "policy_state_dict":policy.state_dict(),
        "upstream_checkpoint_path":str(Path(upstream_path).resolve()),
        "upstream_checkpoint_sha256":file_sha256(upstream_path),
    }

def load_policy_payload(payload, contract, statistics, *, expected_stage, allow_untrained_schema_dry_run=False):
    if payload.get("schema_version")!=SCHEMA_VERSION or payload.get("schema_hash")!=contract.schema_hash:
        raise ValueError("Legacy/incompatible policy checkpoint")
    if payload.get("producer_stage")!="Stage"+str(expected_stage):
        raise ValueError("Wrong stage in policy checkpoint chain")
    if payload.get("statistics_hash")!=statistics["statistics_hash"]:
        raise ValueError("Policy uses different DEV preprocessing")
    if payload.get("action_ids")!=list(ACTION_IDS) or payload.get("action_contract_sha256")!=contract.action_contract_sha256:
        raise ValueError("Policy action contract mismatch")
    if not payload.get("trained") and not allow_untrained_schema_dry_run:
        raise ValueError("Untrained schema fixture is forbidden for training transfer or policy evaluation")
    parent=Path(payload["upstream_checkpoint_path"])
    if not parent.is_file() or file_sha256(parent)!=payload.get("upstream_checkpoint_sha256"):
        raise ValueError("Policy predecessor checkpoint lineage is missing or changed")
    if payload["encoder"]["statistics"]["statistics_hash"] != statistics["statistics_hash"]:
        raise ValueError("Nested encoder preprocessing differs from policy")
    encoder=load_encoder_payload(payload["encoder"],contract,allow_untrained_schema_dry_run=allow_untrained_schema_dry_run)
    policy=V43Policy(encoder)
    expected_blocks=encoder.block_ids.clone()
    policy.load_state_dict(payload["policy_state_dict"],strict=True)
    expected_actions=torch.from_numpy(V43ActionCodec(contract).representations)
    if not torch.equal(policy.encoder.action_table,expected_actions) or not torch.equal(policy.encoder.block_ids,expected_blocks):
        raise ValueError("Policy checkpoint overwrote frozen semantic buffers")
    policy.checkpoint_trained=bool(payload["trained"])
    policy.checkpoint_stage=expected_stage
    return policy

