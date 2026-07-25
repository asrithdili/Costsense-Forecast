"""Cross-account STS AssumeRole helper.

Container deployments (ECS task with an instance role) can't rely on local
SSO profiles — the task has ONE natural identity (its task role). To let
CostSense show data from multiple linked AWS accounts, we let the task role
`sts:AssumeRole` into a `costsense-cross-account-read` role in each target
account. Each of those trust-roles has the SAME read-only policy the local
SSO users have.

Configuration is a single env var, ``COSTSENSE_CROSS_ACCOUNT_ROLES``.
Comma-separated list of entries. Each entry is either::

    arn:aws:iam::<account-id>:role/<role-name>

… or with an optional pipe-delimited display label::

    arn:aws:iam::<account-id>:role/<role-name>|<label>

Example::

    COSTSENSE_CROSS_ACCOUNT_ROLES=\
      arn:aws:iam::014666657409:role/costsense-cross-account-read|dil-data-platform-dev,\
      arn:aws:iam::048177250984:role/costsense-cross-account-read|dil-connector-service-dev

The label becomes the profile name in the UI. Without a label, the
account id is used as the profile name.

Sessions returned by ``make_cross_account_session`` are cached until the
STS temp credentials come within 5 minutes of expiring — they're re-issued
automatically. No user action needed to keep the deployed URL working.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import boto3


@dataclass(frozen=True)
class CrossAccountRole:
    """One cross-account target: which role to assume, what nickname to show."""
    role_arn: str
    label: str
    account_id: str


def parse_cross_account_roles() -> list[CrossAccountRole]:
    """Read + parse ``COSTSENSE_CROSS_ACCOUNT_ROLES``. Returns [] when unset.

    Malformed entries are silently skipped so a typo in one entry doesn't
    take down the whole list. We do log to stderr, so an operator can catch
    it while reviewing container logs.
    """
    raw = os.environ.get("COSTSENSE_CROSS_ACCOUNT_ROLES", "").strip()
    if not raw:
        return []
    out: list[CrossAccountRole] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # Optional label after '|'
        if "|" in entry:
            role_arn, label = entry.split("|", 1)
            role_arn = role_arn.strip()
            label = label.strip()
        else:
            role_arn = entry
            label = ""
        # Parse account id from the ARN
        # arn:aws:iam::<account>:role/<name>
        parts = role_arn.split(":")
        if len(parts) < 6 or parts[0] != "arn" or parts[2] != "iam":
            import sys
            print(f"[cross_account] skipping malformed ARN: {role_arn!r}",
                  file=sys.stderr, flush=True)
            continue
        account_id = parts[4]
        if not label:
            label = account_id
        out.append(CrossAccountRole(
            role_arn=role_arn, label=label, account_id=account_id,
        ))
    return out


# ---------------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------------
#
# STS AssumeRole gives us temporary credentials with a fixed expiry (default
# 1 hour). We cache the boto3.Session per role_arn until it comes within 5
# minutes of expiring, then re-issue. No user action required.

_LOCK = threading.Lock()
_SESSION_CACHE: dict[str, tuple[boto3.Session, datetime]] = {}
# Small buffer before expiry — refresh the session slightly early so an
# in-flight request doesn't fail with ExpiredToken mid-call.
_REFRESH_BUFFER = timedelta(minutes=5)


def make_cross_account_session(role_arn: str) -> boto3.Session:
    """Return a cached boto3.Session with temp credentials for ``role_arn``.

    On the first call for a role, do an sts:AssumeRole from the ambient
    credentials (typically the ECS task role). Cache the result until near
    expiry.

    Ambient credentials come from the default chain — task role via IMDS
    when running on ECS Fargate, or the shell's SSO session when running
    locally with an ambient AWS_PROFILE.
    """
    with _LOCK:
        cached = _SESSION_CACHE.get(role_arn)
        if cached is not None:
            session, expires_at = cached
            if datetime.now(timezone.utc) < expires_at - _REFRESH_BUFFER:
                return session
        # Cache miss or nearing expiry — re-issue
        base = boto3.Session()   # ambient (task role or local SSO)
        sts = base.client("sts")
        # Session name must be <= 64 chars, no illegal chars — use the
        # last path element of the role ARN + a short hostname tag.
        role_name = role_arn.split("/")[-1][:32]
        session_name = f"costsense-{role_name}"[:64]
        resp = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name,
            DurationSeconds=3600,
        )
        creds = resp["Credentials"]
        session = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=base.region_name or os.environ.get("AWS_REGION") or "us-west-2",
        )
        expires_at = creds["Expiration"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        _SESSION_CACHE[role_arn] = (session, expires_at)
        return session


def get_role_by_label(label: str) -> Optional[CrossAccountRole]:
    """Look up a configured cross-account role by its display label.

    Used when Streamlit hands us a profile-name (which is really the
    label) and we need to find the ARN to assume.
    """
    for role in parse_cross_account_roles():
        if role.label == label:
            return role
    return None
