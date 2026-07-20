"""Match an AWS SSO profile name to the repo(s) that deploy to it.

Diligent's convention is `dil-<repo-name>-<env>` for AWS profiles and just
`<repo-name>` for GitHub repos. The `dil-team-*` profiles are team shared
accounts and don't correspond to any specific repo; we skip them.
"""
from __future__ import annotations

import re


_ENV_SUFFIX = re.compile(r"[-_](dev|prod|staging|preprod|qa|test|tools)$",
                         re.IGNORECASE)
_TEAM_SHARED = re.compile(r"^dil[-_]team[-_]", re.IGNORECASE)


def normalize_profile(profile: str) -> str | None:
    """Strip `dil-` prefix and env suffix. Returns None for shared/team
    profiles (they don't map to a repo)."""
    if not profile:
        return None
    if _TEAM_SHARED.match(profile):
        return None
    p = profile
    if p.lower().startswith("dil-") or p.lower().startswith("dil_"):
        p = p[4:]
    p = _ENV_SUFFIX.sub("", p)
    return p.strip("-_ ") or None


def match_repos(profile: str, available_short_names: list[str]) -> list[str]:
    """Return the repo short-names from `available_short_names` that
    match the given SSO profile. Match is case-insensitive prefix or
    equality on the normalized profile name."""
    normalized = normalize_profile(profile)
    if not normalized:
        return []
    n_low = normalized.lower()
    exact = [r for r in available_short_names if r.lower() == n_low]
    if exact:
        return exact
    # Fallback: prefix match (`data-platform-` vs `data-platform`)
    return [r for r in available_short_names
            if r.lower().startswith(n_low) or n_low.startswith(r.lower())]
