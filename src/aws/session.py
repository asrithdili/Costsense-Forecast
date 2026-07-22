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


def _session_from_environment() -> boto3.Session:
    """Session backed by ``AWS_*`` env vars (OIDC, aws-vault, export).

    A plain ``boto3.Session()`` still honours ``AWS_PROFILE`` in the
    environment and tries to load that name from ``~/.aws/config``, which
    fails on GitHub Actions runners where only ephemeral OIDC creds exist.
    """
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    return boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=region,
    )


def _should_use_env_credentials(profile: str | None) -> bool:
    if not _env_credentials_available():
        return False
    if not profile:
        return True
    vault = vault_profile()
    if vault == profile:
        return True
    active = os.environ.get("AWS_PROFILE")
    return active == profile and not vault


def make_session(profile: str | None = None) -> boto3.Session:
    """Return a boto3 Session for *profile*, honoring aws-vault env creds."""
    if _should_use_env_credentials(profile):
        return _session_from_environment()

    if not profile:
        return boto3.Session()

    return boto3.Session(profile_name=profile)
