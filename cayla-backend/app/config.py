"""Settings read from environment variables (see .env.example)."""

import os
from zoneinfo import ZoneInfo

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://cctv:cctv@localhost:5433/cctv"
)

# Session lifetime. Lockout/timeout policy is deferred (Open Items), so keep it simple for now.
SESSION_HOURS = int(os.environ.get("SESSION_HOURS", "12"))
SESSION_COOKIE = "cctv_session"
# Set COOKIE_SECURE=false only for local http:// development.
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"

# Staff invites: how long a temporary password works, and what the invite email says.
TEMP_PASSWORD_HOURS = int(
    os.environ.get("TEMP_PASSWORD_HOURS", "168")
)  # 168 hours = 7days
APP_URL = os.environ.get(
    "APP_URL", "http://localhost:8000/docs"
)  # Where to redirect users after login -- change to dashboard once it exists
APP_NAME = os.environ.get("APP_NAME", "InSight")
SITE_NAME = os.environ.get(
    "SITE_NAME", "Tower A"
)  # "...added you to InSight for Tower A"

# Outgoing email (see app/mailer.py). Empty SMTP_HOST = print emails to the log instead.
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "InSight <no-reply@insight.towera.sg>")
SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "true").lower() == "true"

# Web addresses the dashboard runs at, comma-separated. The browser only lets a page
# on one of these call the API with the login cookie. Must match exactly: scheme, host, port.
FRONTEND_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "FRONTEND_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")
    if o.strip()
]

# Rule days/hours are evaluated in the site's local time, not UTC.
SITE_TZ = ZoneInfo(os.environ.get("SITE_TZ", "Asia/Singapore"))
