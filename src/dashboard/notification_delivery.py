"""Manual notification email delivery for CostSense (SMTP, opt-in via .env).

No automatic sends — call ``send_notification_email()`` only from explicit UI
actions. Uses Python stdlib ``smtplib``; no extra dependencies.

Required environment variables (via shell or ``.env`` loaded by ``src.env``):

  COSTSENSE_NOTIFY_TO     — default recipient (overrides draft placeholder)
  COSTSENSE_NOTIFY_FROM   — From header address
  COSTSENSE_SMTP_HOST     — SMTP server hostname
  COSTSENSE_SMTP_PORT     — optional, default 587
  COSTSENSE_SMTP_USER     — optional; when set, AUTH is attempted
  COSTSENSE_SMTP_PASSWORD — optional; paired with SMTP_USER
  COSTSENSE_SMTP_USE_TLS  — optional, default true for port 587
"""
from __future__ import annotations

import hashlib
import os
import smtplib
import ssl
from email.message import EmailMessage

from src.env import load_env


def notify_recipient(fallback: str = "finops-team@example.com") -> str:
    """Configured delivery recipient, or *fallback* from the draft."""
    load_env()
    return (os.environ.get("COSTSENSE_NOTIFY_TO") or fallback).strip() or fallback


def draft_fingerprint(
    *,
    recipient: str,
    subject: str,
    body: str,
    source_type: str,
) -> str:
    """Stable id for duplicate-send protection."""
    payload = f"{recipient}\n{subject}\n{body}\n{source_type}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _smtp_config() -> tuple[str, int, str, str | None, str | None, bool]:
    load_env()
    host = (os.environ.get("COSTSENSE_SMTP_HOST") or os.environ.get("SMTP_HOST") or "").strip()
    port_raw = (os.environ.get("COSTSENSE_SMTP_PORT") or os.environ.get("SMTP_PORT") or "587").strip()
    sender = (os.environ.get("COSTSENSE_NOTIFY_FROM") or os.environ.get("SMTP_FROM") or "").strip()
    user = (os.environ.get("COSTSENSE_SMTP_USER") or os.environ.get("SMTP_USER") or "").strip() or None
    password = (
        os.environ.get("COSTSENSE_SMTP_PASSWORD")
        or os.environ.get("SMTP_PASSWORD")
        or ""
    ).strip() or None
    use_tls_raw = (os.environ.get("COSTSENSE_SMTP_USE_TLS") or "true").strip().lower()
    use_tls = use_tls_raw not in {"0", "false", "no"}
    try:
        port = int(port_raw)
    except ValueError:
        port = 587
    return host, port, sender, user, password, use_tls


def smtp_configured() -> tuple[bool, str]:
    """Return (ready, human-readable status) for the draft panel."""
    host, _port, sender, _user, _password, _tls = _smtp_config()
    if not host:
        return False, (
            "SMTP not configured — set **COSTSENSE_SMTP_HOST** and "
            "**COSTSENSE_NOTIFY_FROM** in `.env` to enable Send email."
        )
    if not sender:
        return False, "Set **COSTSENSE_NOTIFY_FROM** in `.env` (sender address)."
    return True, f"SMTP ready ({host})"


def send_notification_email(
    *,
    recipient: str,
    subject: str,
    body: str,
) -> tuple[bool, str]:
    """Send one plain-text notification email. Manual invocation only."""
    host, port, sender, user, password, use_tls = _smtp_config()
    if not host:
        return False, (
            "SMTP host not configured. Add COSTSENSE_SMTP_HOST to your `.env`."
        )
    if not sender:
        return False, "Sender not configured. Add COSTSENSE_NOTIFY_FROM to your `.env`."
    if not recipient or "@" not in recipient:
        return False, f"Invalid recipient address: {recipient!r}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(body)

    try:
        with smtplib.SMTP(host=host, port=port, timeout=30) as smtp:
            if use_tls:
                smtp.starttls(context=ssl.create_default_context())
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
    except Exception as exc:  # noqa: BLE001
        return False, f"Email send failed: {exc}"

    return True, f"Email sent to {recipient}."
