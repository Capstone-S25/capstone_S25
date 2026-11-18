"""Create the first admin (nobody can log in to create users until one exists).

    python -m scripts.create_admin --name "Admin" --email admin@insight.sg --staff-id S001 --dob 1990-01-01
It asks for the password so it never lands in your shell history.
"""

import argparse

import psycopg
import pwinput
from app.config import DATABASE_URL
from app.password_policy import password_problem
from app.security import hash_password
from psycopg.rows import dict_row

from app import audit

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--email", required=True)
p.add_argument("--staff-id", required=True)
p.add_argument("--dob", help="YYYY-MM-DD (optional)")
a = p.parse_args()

pw = pwinput.pwinput(
    prompt="Password (8+ chars, upper & lower case, number, symbol): ", mask="*"
)
problem = password_problem(pw)
if problem:
    raise SystemExit(problem)
if pw != pwinput.pwinput(prompt="Again: ", mask="*"):
    raise SystemExit("Passwords don't match")

with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
    row = conn.execute(
        "INSERT INTO users (name, email, staff_id, date_of_birth, role, password_hash) "
        "VALUES (%s, %s, %s, %s, 'admin', %s) RETURNING user_id, name, email, staff_id, role",
        (a.name, a.email, a.staff_id, a.dob, hash_password(pw)),
    ).fetchone()
    audit.record(
        conn,
        actor_id=None,
        action="create",
        entity_type="user",
        entity_id=row["user_id"],
        new=row,
    )
print(f"Created admin user_id={row['user_id']}")
