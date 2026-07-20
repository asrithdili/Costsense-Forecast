"""Enumerate local AWS profiles and their account IDs.

Kept read-only: uses STS get-caller-identity per profile, cached in-process so
Streamlit doesn't re-hit STS on every widget change.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound


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
        session = boto3.Session(profile_name=profile)
        acct = session.client("sts").get_caller_identity()["Account"]
        return ProfileInfo(profile=profile, account_id=acct)
    except (BotoCoreError, ClientError) as e:
        return ProfileInfo(profile=profile, account_id=None, error=str(e))


def resolve_all() -> list[ProfileInfo]:
    return [resolve(p) for p in list_profiles()]
