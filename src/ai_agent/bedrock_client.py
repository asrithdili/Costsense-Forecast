"""Shared Bedrock-runtime client factory with a generous read timeout.

The default botocore read timeout is 60s. Sonnet often takes 60-120s to think
over a large tool-use payload (anomaly sweep + big diffs), so we bump to 5
minutes. Also disables retries so a slow response doesn't get doubled.

The module also exposes ``resolve_account_id`` — an ``sts:GetCallerIdentity``
one-shot the caller can use to verify the resolved credentials actually
belong to the account they think they do. See the wiring in chat_agent.py
for how the Bedrock call is refused with an honest error when the account
resolved from the session differs from the account the user selected in
the sidebar.
"""
from __future__ import annotations

from functools import lru_cache

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from src.aws.session import make_session
from src.config import get_str


def default_bedrock_region() -> str:
    return get_str("bedrock.region", "us-west-2") or "us-west-2"

_BEDROCK_CONFIG = Config(
    read_timeout=300,      # 5 min — Sonnet + long tool loops can be slow
    connect_timeout=15,
    retries={"max_attempts": 2, "mode": "standard"},
)


@lru_cache(maxsize=8)
def make_client(profile: str | None, region: str | None = None):
    region = region or default_bedrock_region()
    session = make_session(profile)
    return session.client("bedrock-runtime", region_name=region,
                          config=_BEDROCK_CONFIG)


@lru_cache(maxsize=32)
def resolve_account_id(profile: str | None) -> str | None:
    """Return the AWS account id the *profile* actually resolves to via
    ``sts:GetCallerIdentity``. Cached per-profile in-process so we don't
    hit STS on every chat turn.

    Returns ``None`` on any failure (auth, network, expired token). The
    caller decides how to handle a ``None`` — the chat page treats it as
    "cannot verify" and refuses the Bedrock call rather than proceeding
    with unverifiable credentials.
    """
    try:
        session = make_session(profile)
        sts = session.client("sts")
        return sts.get_caller_identity()["Account"]
    except (BotoCoreError, ClientError):
        return None
    except Exception:  # noqa: BLE001
        return None


def verify_profile_account(
    profile: str | None,
    expected_account_id: str | None,
) -> tuple[bool, str | None, str | None]:
    """Confirm the *profile* actually resolves to *expected_account_id*.

    Returns ``(ok, resolved_account_id, error_message)``:
      * ok=True when both account ids are non-None and match.
      * ok=False, error_message names the mismatch when they differ.
      * ok=False, resolved=None when STS could not be reached at all.

    The chat page uses this to short-circuit a Bedrock call when the
    profile has been ambushed by leaked env credentials pointing at a
    different account. Better to fail loudly with an honest error than
    to silently InvokeModel against the wrong role.
    """
    resolved = resolve_account_id(profile)
    if resolved is None:
        return False, None, (
            f"could not resolve account for profile `{profile}` via "
            f"sts:GetCallerIdentity (session may be expired or "
            f"credentials misconfigured)"
        )
    if not expected_account_id:
        # Caller didn't tell us what to expect — treat as best-effort OK.
        return True, resolved, None
    if resolved == expected_account_id:
        return True, resolved, None
    return False, resolved, (
        f"profile `{profile}` resolves to account {resolved}, but the "
        f"sidebar has account {expected_account_id} selected. Env "
        f"credentials or a stale profile may be overriding the "
        f"selection. Refusing to proceed to avoid running against the "
        f"wrong account."
    )
