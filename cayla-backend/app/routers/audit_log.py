"""#45 GET /audit-log   (#46 is app/audit.py, not an HTTP route; see the note there)"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..deps import AdminUser, Conn

router = APIRouter(prefix="/audit-log", tags=["Audit trail"])


class AuditEntry(BaseModel):
    audit_id: int
    occurred_at: datetime
    actor_user_id: int | None
    actor_name: str | None
    action: str
    entity_type: str
    entity_id: str
    old_values: dict[str, Any] | None
    new_values: dict[str, Any] | None


class AuditPage(BaseModel):
    items: list[AuditEntry]
    next_before_id: int | None  # pass back as before_id to get the next (older) page


@router.get("", response_model=AuditPage)
def list_audit_log(
    user: AdminUser,
    conn: Conn,
    entity_type: str | None = None,
    entity_id: str | None = None,
    actor_user_id: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    before_id: int | None = Query(default=None, description="Cursor for paging"),
    limit: int = Query(default=50, ge=1, le=200),
):
    where, params = [], []
    for clause, value in (
        ("a.entity_type = %s", entity_type),
        ("a.entity_id = %s", entity_id),
        ("a.actor_user_id = %s", actor_user_id),
        ("a.occurred_at >= %s", since),
        ("a.occurred_at < %s", until),
        ("a.audit_id < %s", before_id),
    ):
        if value is not None:
            where.append(clause)
            params.append(value)

    rows = conn.execute(
        f"""
        SELECT a.*, u.name AS actor_name
        FROM audit_log a LEFT JOIN users u ON u.user_id = a.actor_user_id
        {"WHERE " + " AND ".join(where) if where else ""}
        ORDER BY a.audit_id DESC
        LIMIT %s
        """,
        (*params, limit + 1),
    ).fetchall()

    has_more = len(rows) > limit
    rows = rows[:limit]
    return AuditPage(
        items=rows, next_before_id=rows[-1]["audit_id"] if has_more else None
    )
