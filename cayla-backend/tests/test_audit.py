from datetime import date, datetime, timezone
from decimal import Decimal

from app.audit import diff, record


class FakeConn:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params):
        self.calls.append(params)


def test_update_keeps_only_changed_fields():
    old = {
        "rule_id": 1,
        "severity": "low",
        "min_confidence": Decimal("0.500"),
        "is_active": True,
    }
    new = {
        "rule_id": 1,
        "severity": "high",
        "min_confidence": Decimal("0.500"),
        "is_active": True,
    }
    assert diff(old, new) == ({"severity": "low"}, {"severity": "high"})


def test_create_and_delete():
    row = {"location_id": 3, "name": "Lobby"}
    assert diff(None, row) == (None, row)
    assert diff(row, None) == (row, None)


def test_secrets_and_bookkeeping_never_logged():
    old = {
        "name": "A",
        "password_hash": "x",
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    new = {
        "name": "B",
        "password_hash": "y",
        "updated_at": datetime(2026, 2, 1, tzinfo=timezone.utc),
    }
    assert diff(old, new) == ({"name": "A"}, {"name": "B"})


def test_values_become_json_friendly():
    _, n = diff(
        None, {"dob": date(2002, 1, 31), "conf": Decimal("0.75"), "days": (1, 2)}
    )
    assert n == {"dob": "2002-01-31", "conf": 0.75, "days": [1, 2]}


def test_noop_update_writes_nothing():
    conn = FakeConn()
    assert (
        record(
            conn,
            actor_id=1,
            action="update",
            entity_type="rule",
            entity_id=1,
            old={"a": 1},
            new={"a": 1},
        )
        is False
    )
    assert conn.calls == []


def test_record_writes_one_row():
    conn = FakeConn()
    record(
        conn,
        actor_id=7,
        action="update",
        entity_type="settings",
        entity_id=1,
        old={"t_pre_seconds": 10},
        new={"t_pre_seconds": 15},
    )
    actor, action, etype, eid, old_json, new_json = conn.calls[0]
    assert (actor, action, etype, eid) == (7, "update", "settings", "1")
    assert old_json == '{"t_pre_seconds": 10}' and new_json == '{"t_pre_seconds": 15}'
