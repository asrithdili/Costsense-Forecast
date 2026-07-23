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

from src.config import get_str


def vault_profile() -> str | None:
    """Profile name when running under ``aws-vault exec``."""
    value = os.environ.get("AWS_VAULT")
    return value if value else None


def runtime_profile_name() -> str:
    """Logical profile label for ECS task roles and other default-credential runtimes."""
    return get_str("aws.runtime_profile", "ecs-task-role") or "ecs-task-role"


def aws_region() -> str | None:
    return get_str("aws.region")


def _env_credentials_available() -> bool:
    return bool(os.environ.get("AWS_ACCESS_KEY_ID"))


def _session_from_environment() -> boto3.Session:
    """Session backed by ``AWS_*`` env vars (OIDC, aws-vault, export).

    Even with explicit keys, botocore still reads ``AWS_PROFILE`` from the
    environment and tries to load that name from ``~/.aws/config``.  Temporarily
    unset profile env vars while constructing the session (same pattern as
    ``profiles.list_profiles``).
    """
    profile_keys = ("AWS_PROFILE", "AWS_DEFAULT_PROFILE")
    saved = {k: os.environ.pop(k) for k in profile_keys if k in os.environ}
    try:
        region = (
            os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
            or aws_region()
        )
        return boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=region,
        )
    finally:
        for key, value in saved.items():
            os.environ[key] = value


def _should_use_env_credentials(profile: str | None) -> bool:
    if not _env_credentials_available():
        return False
    active = os.environ.get("AWS_PROFILE")
    # OIDC / exported creds with no profile env (e.g. GitHub Actions).
    if not active:
        return True
    if not profile:
        return True
    vault = vault_profile()
    if vault == profile:
        return True
    return active == profile and not vault


def make_session(profile: str | None = None) -> boto3.Session:
    """Return a boto3 Session for *profile*, honoring aws-vault env creds."""
    if profile == runtime_profile_name():
        profile = None

    if _should_use_env_credentials(profile):
        return _session_from_environment()

    if not profile:
        region = aws_region()
        if region:
            return boto3.Session(region_name=region)
        return boto3.Session()

    return boto3.Session(profile_name=profile)
