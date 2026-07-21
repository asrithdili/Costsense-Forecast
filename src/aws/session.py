"""Shared boto3 session factory.

Supports:
- **AWS SSO** (default): ``Session(profile_name=…)`` loads cached SSO tokens.
- **aws-vault exec**: when the app is launched via
  ``aws-vault exec <profile> -- …``, temporary credentials are injected into
  the environment. A named ``Session`` would bypass those env vars and
  re-fetch through the profile config chain, so we use a plain ``Session()``
  when the requested profile matches the active vault profile.
- **aws-vault export**: when ``AWS_PROFILE`` is set and static/session creds
  are present in the environment (e.g. ``eval $(aws-vault export …)``).
"""
from __future__ import annotations

import os

import boto3


def vault_profile() -> str | None:
    """Profile name when running under ``aws-vault exec``."""
    value = os.environ.get("AWS_VAULT")
    return value if value else None


def _env_credentials_available() -> bool:
    return bool(os.environ.get("AWS_ACCESS_KEY_ID"))


def make_session(profile: str | None = None) -> boto3.Session:
    """Return a boto3 Session for *profile*, honoring aws-vault env creds."""
    if not profile:
        return boto3.Session()

    vault = vault_profile()
    if vault == profile and _env_credentials_available():
        return boto3.Session()

    active = os.environ.get("AWS_PROFILE")
    if (active == profile
            and _env_credentials_available()
            and not vault):
        return boto3.Session()

    return boto3.Session(profile_name=profile)
