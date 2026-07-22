"""AWS Organizations-wide spend rollup.

Uses Cost Explorer's `LINKED_ACCOUNT` dimension on the management/payer
profile to get per-account daily spend across every linked account in the
Organization — even for accounts you don't have direct SSO to.

Kept read-only. Falls back gracefully when `organizations:ListAccounts`
isn't granted (names come out as `<account_id>` in that case).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

from botocore.exceptions import BotoCoreError, ClientError

from src.aws.session import make_session


@dataclass
class AccountSpend:
    account_id: str
    account_name: str
    email: str = ""
    total_usd: float = 0.0
    daily: list[tuple[date, float]] = field(default_factory=list)

    def spend_last_n_days(self, n: int) -> float:
        if not self.daily:
            return 0.0
        return round(sum(v for _, v in self.daily[-n:]), 2)

    def trend_pct(self) -> float | None:
        """Compare last-7-day spend to prior-7-day spend."""
        if len(self.daily) < 14:
            return None
        last = sum(v for _, v in self.daily[-7:])
        prior = sum(v for _, v in self.daily[-14:-7])
        if prior == 0:
            return None
        return round((last - prior) / prior * 100, 1)


def _session(profile: str | None):
    return make_session(profile)


@lru_cache(maxsize=4)
def _org_account_names(profile: str | None) -> dict[str, tuple[str, str]]:
    """Try Organizations to get {account_id: (name, email)}. Returns {} on
    AccessDenied — the sweep will just show IDs."""
    try:
        org = _session(profile).client("organizations")
        out: dict[str, tuple[str, str]] = {}
        paginator = org.get_paginator("list_accounts")
        for page in paginator.paginate():
            for acct in page.get("Accounts", []):
                out[acct["Id"]] = (acct.get("Name") or "", acct.get("Email") or "")
        return out
    except (BotoCoreError, ClientError):
        return {}


def fetch_org_spend(
    profile: str,
    days: int = 30,
    region: str = "us-east-1",
) -> list[AccountSpend]:
    """Return per-linked-account spend for the last `days`."""
    ce = _session(profile).client("ce", region_name=region)
    end = date.today()
    start = end - timedelta(days=days)

    per_account_daily: dict[str, list[tuple[date, float]]] = {}
    next_token: str | None = None
    while True:
        kwargs = {
            "TimePeriod": {"Start": start.isoformat(), "End": end.isoformat()},
            "Granularity": "DAILY",
            "Metrics": ["UnblendedCost"],
            "GroupBy": [{"Type": "DIMENSION", "Key": "LINKED_ACCOUNT"}],
        }
        if next_token:
            kwargs["NextPageToken"] = next_token
        resp = ce.get_cost_and_usage(**kwargs)
        for period in resp.get("ResultsByTime", []):
            day = date.fromisoformat(period["TimePeriod"]["Start"])
            for group in period.get("Groups", []):
                acct_id = group["Keys"][0]
                amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
                per_account_daily.setdefault(acct_id, []).append((day, amount))
        next_token = resp.get("NextPageToken")
        if not next_token:
            break

    names = _org_account_names(profile)

    out: list[AccountSpend] = []
    for acct_id, daily in per_account_daily.items():
        daily.sort()
        name, email = names.get(acct_id, (acct_id, ""))
        total = round(sum(v for _, v in daily), 2)
        out.append(AccountSpend(
            account_id=acct_id,
            account_name=name or acct_id,
            email=email,
            total_usd=total,
            daily=daily,
        ))
    out.sort(key=lambda a: -a.total_usd)
    return out


def top_service_by_account(
    profile: str,
    account_id: str,
    days: int = 30,
    region: str = "us-east-1",
) -> str | None:
    """Best-effort: fetch each account's top service via a filtered CE call."""
    try:
        ce = _session(profile).client("ce", region_name=region)
        end = date.today()
        start = end - timedelta(days=days)
        resp = ce.get_cost_and_usage(
            TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
            Filter={"Dimensions": {"Key": "LINKED_ACCOUNT",
                                    "Values": [account_id]}},
        )
        totals: dict[str, float] = {}
        for period in resp.get("ResultsByTime", []):
            for g in period.get("Groups", []):
                svc = g["Keys"][0]
                amt = float(g["Metrics"]["UnblendedCost"]["Amount"])
                totals[svc] = totals.get(svc, 0.0) + amt
        if not totals:
            return None
        return max(totals.items(), key=lambda kv: kv[1])[0]
    except (BotoCoreError, ClientError):
        return None
