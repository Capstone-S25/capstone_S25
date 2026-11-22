"""Sending email.

Local development: docker-compose runs Mailpit, a fake mail server. Every email
the backend sends shows up at http://localhost:8025 instead of reaching anyone.

Production: point SMTP_HOST etc. at a real server. Amazon SES (which the team
plans to use for alerts) has an SMTP interface, so this same code works with it.

No SMTP_HOST set at all: the email is printed to the API's log instead
(`docker compose logs api`), so nothing breaks while testing.

Yee Mon owns email for alerts and reports; she can call send_email() too, so
there is one place that sends mail.
"""

import html
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage

from .config import (
    APP_NAME,
    SITE_NAME,
    SMTP_FROM,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_STARTTLS,
    SMTP_USER,
)

ROLE_LABELS = {"staff": "Security staff", "admin": "an admin"}


class EmailError(Exception):
    pass


@dataclass
class Email:
    subject: str
    text: str  # plain-text version, for mail apps that don't show HTML
    html: str | None = None


def send_email(to: str, email: Email) -> None:
    if not SMTP_HOST:
        print(
            f"\n===== EMAIL (not sent: SMTP_HOST is not set) =====\n"
            f"To: {to}\nSubject: {email.subject}\n\n{email.text}\n"
            f"===================================================\n",
            flush=True,
        )
        return

    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to
    msg["Subject"] = email.subject
    msg.set_content(email.text)
    if email.html:
        msg.add_alternative(email.html, subtype="html")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
            if SMTP_STARTTLS:
                smtp.starttls(context=ssl.create_default_context())
            if SMTP_USER:
                smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.send_message(msg)
    except (OSError, smtplib.SMTPException) as e:
        raise EmailError(str(e)) from e


def _expiry_words(hours: int) -> str:
    if hours % 24 == 0:
        days = hours // 24
        return f"{days} day" + ("" if days == 1 else "s")
    return f"{hours} hours"


def _first_name(name: str) -> str:
    return name.split()[0] if name.split() else name


def _html(
    title: str,
    greeting: str,
    intro: str,
    email: str,
    temp_password: str,
    login_url: str,
    footer: str,
) -> str:
    """Matches the 'Invite email (Staff inbox)' frame: white card, grey credentials box, blue button.
    Inline styles only, because most mail apps ignore <style> blocks."""
    e = html.escape
    return f"""<!doctype html>
<html><body style="margin:0;padding:32px 16px;background:#f3f4f6;font-family:Inter,Arial,sans-serif;color:#111827;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td align="center">
<table role="presentation" width="432" cellspacing="0" cellpadding="0" style="max-width:432px;width:100%;background:#ffffff;border-radius:12px;">
<tr><td style="padding:28px 32px;">
  <div style="font-size:20px;font-weight:600;margin:0 0 20px;">{e(title)}</div>
  <div style="font-size:20px;font-weight:700;font-style:italic;color:#1d4ed8;margin:0 0 20px;">{e(APP_NAME)}</div>
  <p style="font-size:14px;line-height:22px;margin:0 0 16px;">{e(greeting)}</p>
  <p style="font-size:14px;line-height:22px;margin:0 0 16px;">{e(intro)}</p>
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f3f4f6;border-radius:8px;">
    <tr><td style="padding:12px;">
      <div style="font-size:11px;color:#6b7280;">Email</div>
      <div style="font-size:14px;margin:2px 0 12px;">{e(email)}</div>
      <div style="font-size:11px;color:#6b7280;">Temporary password</div>
      <div style="font-size:18px;font-weight:600;letter-spacing:1px;font-family:Menlo,Consolas,monospace;margin-top:2px;">{e(temp_password)}</div>
    </td></tr>
  </table>
  <a href="{e(login_url)}" style="display:inline-block;margin:20px 0 16px;padding:10px 16px;background:#1d4ed8;color:#ffffff;font-size:14px;font-weight:600;text-decoration:none;border-radius:8px;">Sign in to {e(APP_NAME)}</a>
  <p style="font-size:12px;line-height:18px;color:#6b7280;margin:0;">{e(footer)}</p>
</td></tr></table>
</td></tr></table>
</body></html>"""


def invite_email(
    name: str,
    email: str,
    temp_password: str,
    invited_by: str,
    role: str,
    expires_hours: int,
    login_url: str,
) -> Email:
    """New staff member, or 'resend invite' for someone who hasn't signed in yet."""
    title = f"You've been invited to {APP_NAME}"
    greeting = f"Hi {_first_name(name)},"
    intro = (
        f"{invited_by} added you to {APP_NAME} for {SITE_NAME} as {ROLE_LABELS.get(role, role)}. "
        f"Sign in with the temporary password below. "
        f"You'll be asked to set your own password straight after."
    )
    footer = (
        f"This password expires in {_expiry_words(expires_hours)}. "
        f"If you weren't expecting this, you can ignore this email."
    )
    text = (
        f"{greeting}\n\n{intro}\n\nEmail: {email}\nTemporary password: {temp_password}\n\n"
        f"Sign in to {APP_NAME}: {login_url}\n\n{footer}\n"
    )
    return Email(
        title,
        text,
        _html(title, greeting, intro, email, temp_password, login_url, footer),
    )


def reset_email(
    name: str,
    email: str,
    temp_password: str,
    reset_by: str,
    expires_hours: int,
    login_url: str,
) -> Email:
    """Someone who already had a password: an admin reset it."""
    title = f"Your {APP_NAME} password was reset"
    greeting = f"Hi {_first_name(name)},"
    intro = (
        f"{reset_by} reset your {APP_NAME} password. Sign in with the temporary password below. "
        f"You'll be asked to set a new password straight after."
    )
    footer = (
        f"This password expires in {_expiry_words(expires_hours)}. "
        f"If you didn't expect this, tell your admin."
    )
    text = (
        f"{greeting}\n\n{intro}\n\nEmail: {email}\nTemporary password: {temp_password}\n\n"
        f"Sign in to {APP_NAME}: {login_url}\n\n{footer}\n"
    )
    return Email(
        title,
        text,
        _html(title, greeting, intro, email, temp_password, login_url, footer),
    )


def _link_html(
    title: str,
    greeting: str,
    intro: str,
    button_label: str,
    url: str,
    footer: str,
) -> str:
    """Same card as _html, with a button in place of the credentials box."""
    e = html.escape
    return f"""<!doctype html>
<html><body style="margin:0;padding:32px 16px;background:#f3f4f6;font-family:Inter,Arial,sans-serif;color:#111827;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td align="center">
<table role="presentation" width="432" cellspacing="0" cellpadding="0" style="max-width:432px;width:100%;background:#ffffff;border-radius:12px;">
<tr><td style="padding:28px 32px;">
  <div style="font-size:20px;font-weight:600;margin:0 0 20px;">{e(title)}</div>
  <div style="font-size:20px;font-weight:700;font-style:italic;color:#1d4ed8;margin:0 0 20px;">{e(APP_NAME)}</div>
  <p style="font-size:14px;line-height:22px;margin:0 0 16px;">{e(greeting)}</p>
  <p style="font-size:14px;line-height:22px;margin:0 0 16px;">{e(intro)}</p>
  <a href="{e(url)}" style="display:inline-block;margin:4px 0 16px;padding:10px 16px;background:#1d4ed8;color:#ffffff;font-size:14px;font-weight:600;text-decoration:none;border-radius:8px;">{e(button_label)}</a>
  <p style="font-size:12px;line-height:18px;color:#6b7280;margin:0 0 8px;">{e(footer)}</p>
  <p style="font-size:11px;line-height:16px;color:#9ca3af;margin:0;word-break:break-all;">{e(url)}</p>
</td></tr></table>
</td></tr></table>
</body></html>"""


def reset_link_email(name: str, reset_url: str, expires_minutes: int) -> Email:
    """Self-service "Forgot password?": a one-time link to choose a new password."""
    title = f"Reset your {APP_NAME} password"
    greeting = f"Hi {_first_name(name)},"
    intro = (
        f"Someone asked to reset the password for your {APP_NAME} account. "
        f"Choose a new password with the button below. "
        f"Your current password keeps working until you do."
    )
    footer = (
        f"This link works once, for {expires_minutes} minutes. "
        f"If you didn't ask for this, you can ignore this email."
    )
    text = f"{greeting}\n\n{intro}\n\nChoose a new password: {reset_url}\n\n{footer}\n"
    return Email(
        title,
        text,
        _link_html(title, greeting, intro, "Choose a new password", reset_url, footer),
    )
