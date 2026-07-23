"""Enumerate local AWS profiles and their account IDs.

Kept read-only: uses STS get-caller-identity per profile, cached in-process so
Streamlit doesn't re-hit STS on every widget change.

On ECS (no ``~/.aws/config``), falls back to the task role via boto3's default
credential chain and exposes it as ``COSTSENSE_AWS_PROFILE`` (default
``ecs-task-role``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound

from src.aws.session import make_session, runtime_profile_name


@dataclass(frozen=True)
class ProfileInfo:
    profile: str
    account_id: str | None
    error: str | None = None

    @property
    def label(self) -> str:
        if self.account_id:
            return f"{self.profile} ({self.account_id})"
        return f"{self.profile} (unreachable)"


@lru_cache(maxsize=1)
def list_profiles() -> tuple[str, ...]:
    try:
        return tuple(boto3.Session().available_profiles)
    except ProfileNotFound:
        # AWS_PROFILE may point at a profile that isn't on this machine.
        saved = os.environ.pop("AWS_PROFILE", None)
        try:
            return tuple(boto3.Session().available_profiles)
        finally:
            if saved is not None:
                os.environ["AWS_PROFILE"] = saved


@lru_cache(maxsize=32)
def resolve(profile: str) -> ProfileInfo:
    try:
        session = make_session(profile)
        acct = session.client("sts").get_caller_identity()["Account"]
        return ProfileInfo(profile=profile, account_id=acct)
    except (BotoCoreError, ClientError) as e:
        return ProfileInfo(profile=profile, account_id=None, error=str(e))


def resolve_runtime() -> ProfileInfo | None:
    """Resolve credentials from the default chain (ECS task role, OIDC, etc.)."""
    name = runtime_profile_name()
    try:
        session = make_session(name)
        acct = session.client("sts").get_caller_identity()["Account"]
        return ProfileInfo(profile=name, account_id=acct)
    except (BotoCoreError, ClientError):
        return None


def resolve_all() -> list[ProfileInfo]:
    configured = [resolve(p) for p in list_profiles()]
    reachable = [p for p in configured if p.account_id]
    if reachable:
        return reachable
    runtime = resolve_runtime()
    if runtime:
        return [runtime]
    return configured
