from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import ContractError, canonical_bytes, canonical_hash, utc_now
from .retry_policy import (
    CancellationOrigin,
    classify_provider_job_terminal,
    is_known_retry_class,
    is_retryable_class,
    normalized_cancellation_origin,
)

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS logical_requests (
  request_id TEXT PRIMARY KEY, protocol_release_hash TEXT NOT NULL, cell_id TEXT NOT NULL,
  firm_key TEXT NOT NULL, model_key TEXT NOT NULL, wave INTEGER NOT NULL CHECK (wave IN (1,2)),
  parent_request_id TEXT, request_state TEXT NOT NULL, selected_attempt_index INTEGER,
  selected_raw_sha256 TEXT, selection_lock_timestamp_utc TEXT, selection_reason TEXT,
  UNIQUE(protocol_release_hash, cell_id, firm_key)
);
CREATE TABLE IF NOT EXISTS shards (
  shard_id TEXT PRIMARY KEY, protocol_release_hash TEXT NOT NULL, wave INTEGER NOT NULL,
  model_key TEXT NOT NULL, provider TEXT NOT NULL, attempt_index INTEGER NOT NULL,
  ordinal INTEGER NOT NULL, input_path TEXT NOT NULL, input_sha256 TEXT NOT NULL,
  request_count INTEGER NOT NULL CHECK(request_count BETWEEN 1 AND 500),
  shard_state TEXT NOT NULL, provider_job_id TEXT, submission_correlation_key TEXT,
  cancellation_origin TEXT, cancellation_reason TEXT, cancellation_actor TEXT,
  cancellation_recorded_utc TEXT, created_utc TEXT NOT NULL, submitted_utc TEXT, resolved_utc TEXT,
  UNIQUE(protocol_release_hash,wave,model_key,attempt_index,ordinal)
);
CREATE TABLE IF NOT EXISTS shard_requests (
  shard_id TEXT NOT NULL REFERENCES shards(shard_id), request_id TEXT NOT NULL REFERENCES logical_requests(request_id),
  ordinal INTEGER NOT NULL, PRIMARY KEY(shard_id,request_id), UNIQUE(shard_id,ordinal)
);
CREATE TABLE IF NOT EXISTS attempts (
  request_id TEXT NOT NULL REFERENCES logical_requests(request_id), attempt_index INTEGER NOT NULL CHECK (attempt_index BETWEEN 1 AND 3),
  attempt_id TEXT NOT NULL UNIQUE, shard_id TEXT REFERENCES shards(shard_id), provider_job_id TEXT, provider_request_id TEXT,
  attempt_state TEXT NOT NULL, submitted_utc TEXT, resolved_utc TEXT, raw_visible_text BLOB,
  raw_visible_text_sha256 TEXT, raw_provider_json TEXT, provider_artifact_kind TEXT, completion_kind TEXT, retry_class TEXT,
  PRIMARY KEY(request_id, attempt_index)
);
CREATE TABLE IF NOT EXISTS payloads (
  request_id TEXT PRIMARY KEY REFERENCES logical_requests(request_id), system_text TEXT NOT NULL,
  user_text TEXT NOT NULL, visible_payload_sha256 TEXT NOT NULL, provider_request_body_sha256 TEXT,
  materialized_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parsed_responses (
  request_id TEXT PRIMARY KEY REFERENCES logical_requests(request_id), selected_attempt_index INTEGER,
  parse_json TEXT NOT NULL, parsed_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, shard_id TEXT,
  event_type TEXT NOT NULL, detail_json TEXT, created_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_state ON logical_requests(request_state);
CREATE INDEX IF NOT EXISTS idx_requests_wave_model ON logical_requests(wave, model_key);
CREATE INDEX IF NOT EXISTS idx_shards_state ON shards(shard_state);
"""


def connect(path: Path) -> sqlite3.Connection:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(path); c.row_factory=sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=FULL"); c.executescript(SCHEMA)
    cols={row[1] for row in c.execute("PRAGMA table_info(attempts)")}
    if "shard_id" not in cols: c.execute("ALTER TABLE attempts ADD COLUMN shard_id TEXT")
    if "retry_class" not in cols: c.execute("ALTER TABLE attempts ADD COLUMN retry_class TEXT")
    if "provider_artifact_kind" not in cols: c.execute("ALTER TABLE attempts ADD COLUMN provider_artifact_kind TEXT")
    shard_cols={row[1] for row in c.execute("PRAGMA table_info(shards)")}
    if "submission_correlation_key" not in shard_cols: c.execute("ALTER TABLE shards ADD COLUMN submission_correlation_key TEXT")
    for name in ("cancellation_origin", "cancellation_reason", "cancellation_actor", "cancellation_recorded_utc"):
        if name not in shard_cols:
            c.execute(f"ALTER TABLE shards ADD COLUMN {name} TEXT")
    return c


def attempt_id(request_id: str, attempt_index: int) -> str:
    return hashlib.sha256(canonical_bytes(["LLM_ATTEMPT_V1",request_id,int(attempt_index)])).hexdigest()


def shard_id(release_hash: str, wave: int, model_key: str, attempt_index: int, ordinal: int, input_sha256: str) -> str:
    return hashlib.sha256(canonical_bytes(["LLM_SHARD_V1",release_hash,int(wave),model_key,int(attempt_index),int(ordinal),input_sha256])).hexdigest()


def insert_logical_requests(connection: sqlite3.Connection, records: Iterable[Mapping[str,Any]]) -> int:
    count=0
    with connection:
        for r in records:
            values=(r["request_id"],r["protocol_release_hash"],r["cell_id"],r["firm_key"],r["model_key"],int(r["wave"]),r.get("parent_request_id"),r["request_state"])
            old=connection.execute("SELECT protocol_release_hash,cell_id,firm_key,model_key,wave,parent_request_id FROM logical_requests WHERE request_id=?",(r["request_id"],)).fetchone()
            if old is None:
                connection.execute("INSERT INTO logical_requests(request_id,protocol_release_hash,cell_id,firm_key,model_key,wave,parent_request_id,request_state) VALUES(?,?,?,?,?,?,?,?)",values); count+=1
            elif tuple(old)!=values[1:7]: raise ContractError(f"Immutable logical request collision: {r['request_id']}")
    return count


def materialize_payload(connection: sqlite3.Connection, *, request_id: str, system_text: str, user_text: str, visible_payload_sha256: str, provider_request_body_sha256: str|None=None) -> None:
    with connection:
        old=connection.execute("SELECT * FROM payloads WHERE request_id=?",(request_id,)).fetchone(); values=(system_text,user_text,visible_payload_sha256,provider_request_body_sha256)
        if old is not None:
            current=(old["system_text"],old["user_text"],old["visible_payload_sha256"],old["provider_request_body_sha256"])
            if current!=values: raise ContractError(f"Payload immutability violation: {request_id}")
            return
        connection.execute("INSERT INTO payloads(request_id,system_text,user_text,visible_payload_sha256,provider_request_body_sha256,materialized_utc) VALUES(?,?,?,?,?,?)",(request_id,*values,utc_now()))
        connection.execute("UPDATE logical_requests SET request_state='READY' WHERE request_id=? AND request_state IN ('PLANNED','WAITING_PARENT')",(request_id,))


def register_shard(connection: sqlite3.Connection, *, release_hash: str, wave: int, model_key: str, provider: str, attempt_index: int, ordinal: int, input_path: str, input_sha256: str, request_ids: list[str]) -> str:
    if not 1<=len(request_ids)<=500 or len(request_ids)!=len(set(request_ids)): raise ContractError("Every shard must contain 1..500 unique requests")
    sid=shard_id(release_hash,wave,model_key,attempt_index,ordinal,input_sha256)
    with connection:
        old=connection.execute("SELECT * FROM shards WHERE shard_id=?",(sid,)).fetchone()
        immutable=(release_hash,int(wave),model_key,provider,int(attempt_index),int(ordinal),input_path,input_sha256,len(request_ids))
        if old is None:
            connection.execute("INSERT INTO shards(shard_id,protocol_release_hash,wave,model_key,provider,attempt_index,ordinal,input_path,input_sha256,request_count,shard_state,created_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(sid,*immutable,"PREPARED",utc_now()))
            for index,rid in enumerate(request_ids): connection.execute("INSERT INTO shard_requests(shard_id,request_id,ordinal) VALUES(?,?,?)",(sid,rid,index))
        else:
            current=tuple(old[k] for k in ("protocol_release_hash","wave","model_key","provider","attempt_index","ordinal","input_path","input_sha256","request_count"))
            if current!=immutable: raise ContractError("Shard identity collision")
            prior=[r[0] for r in connection.execute("SELECT request_id FROM shard_requests WHERE shard_id=? ORDER BY ordinal",(sid,))]
            if prior!=request_ids: raise ContractError("Shard request membership drift")
    return sid


def begin_shard_submission(connection: sqlite3.Connection, *, shard_id: str, correlation_key: str) -> None:
    if not correlation_key:
        raise ContractError("Submission correlation key cannot be empty")
    row=connection.execute("SELECT shard_state,provider_job_id,submission_correlation_key FROM shards WHERE shard_id=?",(shard_id,)).fetchone()
    if row is None:
        raise ContractError("Cannot begin submission for an unknown shard")
    if row["provider_job_id"]:
        raise ContractError("Cannot begin submission for a shard that already has a remote job")
    if row["shard_state"] not in {"PREPARED","SUBMITTING"}:
        raise ContractError(f"Shard is not submission-eligible: {row['shard_state']}")
    if row["submission_correlation_key"] not in {None,correlation_key}:
        raise ContractError("Shard submission correlation key is immutable")
    with connection:
        connection.execute("UPDATE shards SET shard_state='SUBMITTING',submission_correlation_key=? WHERE shard_id=?",(correlation_key,shard_id))
    append_event(connection,event_type="REMOTE_SUBMISSION_INTENT",shard_id=shard_id,detail={"correlation_key":correlation_key})


def reset_shard_after_confirmed_remote_absence(connection: sqlite3.Connection, *, shard_id: str, correlation_key: str) -> None:
    row=connection.execute("SELECT shard_state,provider_job_id,submission_correlation_key FROM shards WHERE shard_id=?",(shard_id,)).fetchone()
    if row is None or row["shard_state"]!="SUBMITTING" or row["provider_job_id"] is not None or row["submission_correlation_key"]!=correlation_key:
        raise ContractError("Only an unbound SUBMITTING shard with matching correlation may reset")
    with connection:
        connection.execute("UPDATE shards SET shard_state='PREPARED' WHERE shard_id=?",(shard_id,))
    append_event(connection,event_type="REMOTE_SUBMISSION_CONFIRMED_ABSENT",shard_id=shard_id,detail={"correlation_key":correlation_key})

def bind_remote_job(connection: sqlite3.Connection, *, shard_id: str, provider_job_id: str) -> None:
    row=connection.execute("SELECT shard_state,provider_job_id FROM shards WHERE shard_id=?",(shard_id,)).fetchone()
    if row is None: raise ContractError("Remote job cannot bind an unknown shard")
    if row["provider_job_id"] and row["provider_job_id"]!=provider_job_id: raise ContractError("Shard remote-job ID is immutable")
    with connection:
        connection.execute("UPDATE shards SET shard_state='SUBMITTED',provider_job_id=?,submitted_utc=COALESCE(submitted_utc,?) WHERE shard_id=?",(provider_job_id,utc_now(),shard_id))
    register_attempts_for_shard(connection,shard_id)


def register_attempts_for_shard(connection: sqlite3.Connection, sid: str) -> int:
    shard=connection.execute("SELECT * FROM shards WHERE shard_id=?",(sid,)).fetchone()
    if shard is None or not shard["provider_job_id"]: raise ContractError("Shard must have a remote job before attempt registration")
    rows=connection.execute("SELECT request_id FROM shard_requests WHERE shard_id=? ORDER BY ordinal",(sid,)).fetchall(); count=0
    with connection:
        for item in rows:
            rid=item["request_id"]; index=int(shard["attempt_index"]); aid=attempt_id(rid,index)
            old=connection.execute("SELECT attempt_id,provider_job_id,shard_id FROM attempts WHERE request_id=? AND attempt_index=?",(rid,index)).fetchone()
            if old is None:
                logical=connection.execute("SELECT selected_attempt_index FROM logical_requests WHERE request_id=?",(rid,)).fetchone()
                if logical is None or logical["selected_attempt_index"] is not None: raise ContractError("Shard targets unknown or locked request")
                prior=connection.execute("SELECT attempt_index,attempt_state,retry_class FROM attempts WHERE request_id=? AND attempt_index<? ORDER BY attempt_index",(rid,index)).fetchall()
                if index>1 and (
                    [x["attempt_index"] for x in prior]!=list(range(1,index))
                    or any(
                        x["attempt_state"]!="NO_CONFIRMED_COMPLETION"
                        or not is_retryable_class(x["retry_class"])
                        for x in prior
                    )
                ):
                    raise ContractError("Retry shard has a nonretryable or unresolved lower attempt")
                connection.execute("INSERT INTO attempts(request_id,attempt_index,attempt_id,shard_id,provider_job_id,attempt_state,submitted_utc) VALUES(?,?,?,?,?,?,?)",(rid,index,aid,sid,shard["provider_job_id"],"SUBMITTED",utc_now()))
                connection.execute("UPDATE logical_requests SET request_state='SUBMITTED' WHERE request_id=?",(rid,)); count+=1
            elif tuple(old)!=(aid,shard["provider_job_id"],sid): raise ContractError("Attempt/shard/job immutability violation")
    return count


def submitted_shards(connection: sqlite3.Connection, *, wave: int, attempt_index: int) -> list[sqlite3.Row]:
    return connection.execute("SELECT * FROM shards WHERE wave=? AND attempt_index=? AND provider_job_id IS NOT NULL ORDER BY model_key,ordinal",(int(wave),int(attempt_index))).fetchall()


def append_event(connection: sqlite3.Connection, *, event_type: str, detail: Mapping[str,Any], request_id: str|None=None, shard_id: str|None=None) -> None:
    with connection: connection.execute("INSERT INTO events(request_id,shard_id,event_type,detail_json,created_utc) VALUES(?,?,?,?,?)",(request_id,shard_id,event_type,json.dumps(dict(detail),ensure_ascii=False,sort_keys=True),utc_now()))


def resolve_attempt(
    connection: sqlite3.Connection,
    *,
    request_id: str,
    attempt_index: int,
    delivered: bool,
    raw_visible_text: str | None,
    completion_kind: str,
    retry_class: str | None = None,
    provider_request_id: str | None = None,
    raw_provider_json: str | None = None,
    provider_artifact_kind: str | None = None,
) -> None:
    row=connection.execute(
        "SELECT attempt_state FROM attempts WHERE request_id=? AND attempt_index=?",
        (request_id,int(attempt_index)),
    ).fetchone()
    if row is None:
        raise ContractError("Attempt must be submitted before it is resolved")
    if row["attempt_state"]!="SUBMITTED":
        raise ContractError("Attempt resolution is immutable")
    if delivered and retry_class is not None:
        raise ContractError("Delivered attempt cannot carry a retry class")
    if not delivered and not retry_class:
        raise ContractError("Undelivered attempt requires an explicit retry class")
    if retry_class is not None and not is_known_retry_class(retry_class):
        raise ContractError(f"Unknown retry class: {retry_class}")
    state="DELIVERED" if delivered else "NO_CONFIRMED_COMPLETION"
    raw=None if raw_visible_text is None else raw_visible_text.encode()
    digest=None if raw is None else hashlib.sha256(raw).hexdigest()
    request_state = (
        "RESOLVING"
        if delivered or is_retryable_class(retry_class)
        else "UNRESOLVED_INFRA"
    )
    with connection:
        connection.execute(
            "UPDATE attempts SET attempt_state=?,provider_request_id=?,resolved_utc=?,"
            "raw_visible_text=?,raw_visible_text_sha256=?,raw_provider_json=?,provider_artifact_kind=?,completion_kind=?,"
            "retry_class=? WHERE request_id=? AND attempt_index=?",
            (
                state,provider_request_id,utc_now(),raw,digest,raw_provider_json,provider_artifact_kind,
                completion_kind,retry_class,request_id,int(attempt_index),
            ),
        )
        connection.execute(
            "UPDATE logical_requests SET request_state=? WHERE request_id=?",
            (request_state,request_id),
        )


def record_shard_cancellation_provenance(
    connection: sqlite3.Connection,
    *,
    shard_id: str,
    origin: str | CancellationOrigin,
    reason: str,
    actor: str,
) -> None:
    normalized = normalized_cancellation_origin(origin)
    if normalized is CancellationOrigin.UNKNOWN:
        raise ContractError("Cancellation provenance must be explicitly USER_INTENTIONAL or PROVIDER_UNINTENDED")
    if not str(reason).strip() or not str(actor).strip():
        raise ContractError("Cancellation provenance requires nonempty reason and actor")
    row = connection.execute(
        "SELECT cancellation_origin,cancellation_reason,cancellation_actor FROM shards WHERE shard_id=?",
        (shard_id,),
    ).fetchone()
    if row is None:
        raise ContractError(f"Unknown shard: {shard_id}")
    desired = (normalized.value, str(reason), str(actor))
    current = tuple(row)
    if any(value is not None for value in current):
        if current != desired:
            raise ContractError("Cancellation provenance is immutable")
        return
    with connection:
        connection.execute(
            "UPDATE shards SET cancellation_origin=?,cancellation_reason=?,cancellation_actor=?,"
            "cancellation_recorded_utc=? WHERE shard_id=?",
            (*desired, utc_now(), shard_id),
        )
    append_event(
        connection,
        event_type="PROVIDER_SHARD_CANCELLATION_PROVENANCE",
        shard_id=shard_id,
        detail={"origin": normalized.value, "reason": str(reason), "actor": str(actor)},
    )


def mark_shard_terminal_failure(
    connection: sqlite3.Connection,
    *,
    sid: str,
    provider_state: str,
    unresolved_request_ids: Iterable[str] | None = None,
) -> int:
    shard=connection.execute(
        "SELECT attempt_index,shard_state,provider,cancellation_origin FROM shards WHERE shard_id=?",
        (sid,),
    ).fetchone()
    if shard is None:
        raise ContractError(f"Unknown shard: {sid}")
    if shard["shard_state"]=="FAILED_TERMINAL":
        return 0
    register_attempts_for_shard(connection,sid)
    index=int(shard["attempt_index"])
    members={str(row["request_id"]) for row in connection.execute(
        "SELECT request_id FROM shard_requests WHERE shard_id=?", (sid,)
    )}
    targets=members if unresolved_request_ids is None else {str(item) for item in unresolved_request_ids}
    if not targets.issubset(members):
        raise ContractError("Terminal unresolved-request set is not a subset of the shard")
    try:
        retry_class=classify_provider_job_terminal(
            str(shard["provider"]), provider_state, shard["cancellation_origin"]
        ).value
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    changed=0
    for request_id in sorted(targets):
        state=connection.execute(
            "SELECT attempt_state FROM attempts WHERE request_id=? AND attempt_index=?",
            (request_id,index),
        ).fetchone()
        if state and state[0]=="SUBMITTED":
            resolve_attempt(
                connection,
                request_id=request_id,
                attempt_index=index,
                delivered=False,
                raw_visible_text=None,
                completion_kind=f"provider_terminal_failure:{provider_state}",
                retry_class=retry_class,
            )
            changed+=1
    with connection:
        connection.execute(
            "UPDATE shards SET shard_state='FAILED_TERMINAL',resolved_utc=? WHERE shard_id=?",
            (utc_now(),sid),
        )
    append_event(
        connection,
        event_type="PROVIDER_SHARD_TERMINAL_FAILURE",
        shard_id=sid,
        detail={
            "provider_state": provider_state,
            "retry_class": retry_class,
            "cancellation_origin": str(shard["cancellation_origin"] or CancellationOrigin.UNKNOWN.value),
            "reconciled_requests": changed,
            "unresolved_request_set_sha256": canonical_hash(sorted(targets)),
        },
    )
    return changed


def lock_selected_attempt(connection: sqlite3.Connection, request_id: str) -> sqlite3.Row:
    request=connection.execute("SELECT * FROM logical_requests WHERE request_id=?",(request_id,)).fetchone()
    if request is None: raise ContractError(f"Unknown request: {request_id}")
    if request["selected_attempt_index"] is not None: return connection.execute("SELECT * FROM attempts WHERE request_id=? AND attempt_index=?",(request_id,request["selected_attempt_index"])).fetchone()
    attempts=connection.execute("SELECT * FROM attempts WHERE request_id=? ORDER BY attempt_index",(request_id,)).fetchall(); delivered=[r for r in attempts if r["attempt_state"]=="DELIVERED"]
    if not delivered: raise ContractError("No delivered completion can be selected")
    selected=delivered[0]
    if any(r["attempt_state"]!="NO_CONFIRMED_COMPLETION" for r in attempts if r["attempt_index"]<selected["attempt_index"]): raise ContractError("A lower attempt remains unresolved")
    with connection: connection.execute("UPDATE logical_requests SET request_state='DELIVERED_LOCKED',selected_attempt_index=?,selected_raw_sha256=?,selection_lock_timestamp_utc=?,selection_reason=? WHERE request_id=? AND selected_attempt_index IS NULL",(selected["attempt_index"],selected["raw_visible_text_sha256"],utc_now(),"lowest delivered attempt after all lower attempts resolved",request_id))
    return selected


def lock_selected_attempt_with_parse(
    connection: sqlite3.Connection,
    request_id: str,
    *,
    expected_attempt_index: int,
    parse_json: str,
) -> sqlite3.Row:
    """Commit selection and its parse atomically; also repairs legacy lock-without-parse rows."""
    with connection:
        request = connection.execute(
            "SELECT * FROM logical_requests WHERE request_id=?", (request_id,)
        ).fetchone()
        if request is None:
            raise ContractError(f"Unknown request: {request_id}")
        attempts = connection.execute(
            "SELECT * FROM attempts WHERE request_id=? ORDER BY attempt_index", (request_id,)
        ).fetchall()
        if request["selected_attempt_index"] is None:
            delivered = [row for row in attempts if row["attempt_state"] == "DELIVERED"]
            if not delivered:
                raise ContractError("No delivered completion can be selected")
            selected = delivered[0]
            if any(
                row["attempt_state"] != "NO_CONFIRMED_COMPLETION"
                for row in attempts
                if row["attempt_index"] < selected["attempt_index"]
            ):
                raise ContractError("A lower attempt remains unresolved")
            if int(selected["attempt_index"]) != int(expected_attempt_index):
                raise ContractError("Parsed attempt differs from the lowest selectable delivery")
            connection.execute(
                "UPDATE logical_requests SET request_state='DELIVERED_LOCKED',selected_attempt_index=?,"
                "selected_raw_sha256=?,selection_lock_timestamp_utc=?,selection_reason=? "
                "WHERE request_id=? AND selected_attempt_index IS NULL",
                (
                    int(selected["attempt_index"]), selected["raw_visible_text_sha256"], utc_now(),
                    "lowest delivered attempt with atomic parsed response", request_id,
                ),
            )
        else:
            if int(request["selected_attempt_index"]) != int(expected_attempt_index):
                raise ContractError("Parsed attempt differs from the locked selected attempt")
            selected = next(
                (row for row in attempts if int(row["attempt_index"]) == int(expected_attempt_index)), None
            )
            if selected is None or selected["attempt_state"] != "DELIVERED":
                raise ContractError("Locked selected attempt is not a delivered completion")
        prior = connection.execute(
            "SELECT selected_attempt_index,parse_json FROM parsed_responses WHERE request_id=?", (request_id,)
        ).fetchone()
        if prior is None:
            connection.execute(
                "INSERT INTO parsed_responses(request_id,selected_attempt_index,parse_json,parsed_utc) "
                "VALUES(?,?,?,?)",
                (request_id, int(expected_attempt_index), parse_json, utc_now()),
            )
        elif int(prior["selected_attempt_index"]) != int(expected_attempt_index) or prior["parse_json"] != parse_json:
            raise ContractError("Selected response parse is non-idempotent")
    return connection.execute(
        "SELECT * FROM attempts WHERE request_id=? AND attempt_index=?",
        (request_id, int(expected_attempt_index)),
    ).fetchone()


def record_late_duplicate(connection: sqlite3.Connection, request_id: str, detail_json: str) -> None:
    detail=json.loads(detail_json) if detail_json else {}
    append_event(connection,event_type="DUPLICATE_LATE",request_id=request_id,detail=detail)


def set_dependency_empty(connection: sqlite3.Connection, request_id: str) -> None:
    with connection: connection.execute("UPDATE logical_requests SET request_state='DEPENDENCY_EMPTY_DRAFT' WHERE request_id=? AND request_state='WAITING_PARENT'",(request_id,))
    append_event(connection,event_type="DEPENDENCY_EMPTY_DRAFT",request_id=request_id,detail={"resolution":"terminal_governed_fallback"})


def ledger_content_hash(connection: sqlite3.Connection) -> str:
    tables = {
        "logical_requests": "request_id",
        "shards": "shard_id",
        "shard_requests": "shard_id,ordinal",
        "attempts": "request_id,attempt_index",
        "payloads": "request_id",
        "parsed_responses": "request_id",
        "events": "event_id",
    }
    snapshot: dict[str, list[dict[str, Any]]] = {}
    for table, order_by in tables.items():
        rows = []
        for raw in connection.execute(f"SELECT * FROM {table} ORDER BY {order_by}"):
            item = {}
            for key in raw.keys():
                value = raw[key]
                if isinstance(value, memoryview):
                    value = value.tobytes()
                if isinstance(value, bytes):
                    value = {"blob_sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
                item[key] = value
            rows.append(item)
        snapshot[table] = rows
    return canonical_hash(snapshot)

def status_counts(connection: sqlite3.Connection) -> dict[str,int]:
    return {str(r["request_state"]):int(r["n"]) for r in connection.execute("SELECT request_state,COUNT(*) AS n FROM logical_requests GROUP BY request_state ORDER BY request_state")}


def deviation_rows(connection: sqlite3.Connection) -> list[dict[str,Any]]:
    return [dict(r) for r in connection.execute("SELECT event_id,request_id,shard_id,event_type,detail_json,created_utc FROM events ORDER BY event_id")]

