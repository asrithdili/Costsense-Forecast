"""Shared Bedrock-runtime client factory with a generous read timeout.

The default botocore read timeout is 60s. Sonnet often takes 60-120s to think
over a large tool-use payload (anomaly sweep + big diffs), so we bump to 5
minutes. Also disables retries so a slow response doesn't get doubled.
"""
from __future__ import annotations

from functools import lru_cache

from botocore.config import Config

from src.aws.session import make_session


DEFAULT_REGION = "us-west-2"

_BEDROCK_CONFIG = Config(
    read_timeout=300,      # 5 min — Sonnet + long tool loops can be slow
    connect_timeout=15,
    retries={"max_attempts": 2, "mode": "standard"},
)


@lru_cache(maxsize=8)
def make_client(profile: str | None, region: str = DEFAULT_REGION):
    session = make_session(profile)
    return session.client("bedrock-runtime", region_name=region,
                          config=_BEDROCK_CONFIG)
