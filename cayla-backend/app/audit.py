"""The audit-log helper every edit calls.

Pattern (copy this into every create/update/delete handler):

    old = conn.execute("SELECT ... FROM rules WHERE rule_id = %s FOR UPDATE", (rid,)).fetchone()
    new = conn.execute("UPDATE rules SET ... RETURNING ...", (...)).fetchone()
    audit.record(conn, actor_id=user.user_id, action="update",
                 entity_type="rule", entity_id=rid, old=old, new=new)

Because it uses the request's own connection, the audit entry is committed in
the same transaction as the change: you can never get an edit without its log
line, or a log line for an edit that rolled back.

Python workers (no logged-in user) call it with actor_id=None.

Why no HTTP "POST /audit-log"? The sheet marks it "Internal only, called
automatically by the backend". Both the API and the workers share this
database, so they import this function directly. Exposing it over HTTP would
let a client write fake history. If the team still wants the endpoint, put it
behind a worker-only key, never a user login.
"""

import json
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Mapping

# Never written to the audit log, even if a caller passes them in.
REDACTED_FIELDS = {"password_hash", "password", "token_hash"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def diff(
    old: Mapping[str, Any] | None, new: Mapping[str, Any] | None
) -> tuple[dict | None, dict | None]:
    """Return (old_values, new_values) holding only the fields that changed.

    create -> (None, all new fields); delete -> (all old fields, None).
    Bookkeeping columns (updated_at etc.) are ignored so they don't drown out real changes.
    """
    ignore = REDACTED_FIELDS | {"updated_at", "updated_by", "created_at", "created_by"}
    o = {k: _jsonable(v) for k, v in (old or {}).items() if k not in ignore}
    n = {k: _jsonable(v) for k, v in (new or {}).items() if k not in ignore}
    if old is None:
        return None, n
    if new is None:
        return o, None
    changed = {k for k in o.keys() | n.keys() if o.get(k) != n.get(k)}
    return {k: o.get(k) for k in changed}, {k: n.get(k) for k in changed}


def record(
    conn,
    *,
    actor_id: int | None,
    action: str,
    entity_type: str,
    entity_id: Any,
    old: Mapping[str, Any] | None = None,
    new: Mapping[str, Any] | None = None,
) -> bool:
    """Write one audit row. Returns False (and writes nothing) for a no-op update."""
    old_values, new_values = diff(old, new)
    if action == "update" and not new_values:
        return False
    conn.execute(
        """
        INSERT INTO audit_log (actor_user_id, action, entity_type, entity_id, old_values, new_values)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
        """,
        (
            actor_id,
            action,
            entity_type,
            str(entity_id),
            json.dumps(old_values) if old_values is not None else None,
            json.dumps(new_values) if new_values is not None else None,
        ),
    )
    return True
