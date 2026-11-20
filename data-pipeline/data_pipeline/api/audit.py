"""cayla-backend's audit helper, for the incidents and reviews routes.

Writes on the request's own transaction, so an edit and its audit entry
are saved together or not at all.
"""

from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from data_pipeline.api.accounts import RequestConnection


def record(
    db: Session,
    *,
    actor_id: int | None,
    action: str,
    entity_type: str,
    entity_id: Any,
    old: Mapping[str, Any] | None = None,
    new: Mapping[str, Any] | None = None,
) -> bool:
    """Write one audit row; False (nothing written) for an update that changed nothing."""
    # cayla-backend/app becomes importable as "app" once accounts.py has loaded it
    # (imported above), so this import stays inside the function.
    from app import audit

    return audit.record(
        RequestConnection(db),
        actor_id=actor_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        old=old,
        new=new,
    )
