"""Org-level spend data layer for the Org-Level Impact page.

Separates *fetching* org spend from *rendering* it, so the page can run
against real Cost Explorer or deterministic demo data with no UI change.

Design (why it's structured this way):
  - Ownership. Accounts carry team / OU / environment tags so spend can be
    grouped into language leadership speaks. Account IDs alone are not an
    answer.
  - Projection. Month-to-date plus a run-rate forecast to month end, and
    variance against budget. "Will we overshoot" is the actual question
    the page is asked.
  - Movers by $. Ranked by absolute dollar change with a materiality floor,
    not by percent — a -17.4% swing on a $888 account is $187, which is noise
    for an org-level view.
  - Freshness. Cost Explorer lags ~24h. Every result carries `data_through`
    so the UI can say so before someone asks why it doesn't match the invoice.

Providers implement one method: fetch(profile, window_days) -> OrgSpend.
"""

from __future__ import annotations

import calendar
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Protocol


# ============================================================================
# CONTRACTS
# ============================================================================
@dataclass
class AccountSpend:
    """One linked account in the AWS Organization, over the query window."""

    account_id: str
    name: str
    team: str                    # owning team — from account tags or owners map
    ou: str                      # organizational unit
    environment: str             # prod | nonprod | sandbox
    total: float                 # window total
    last7: float
    prior7: float
    services: Dict[str, float] = field(default_factory=dict)
    daily: List[float] = field(default_factory=list)
    open_anomalies: int = 0

    # --- derived -----------------------------------------------------------
    @property
    def delta_abs(self) -> float:
        """Dollar change, last 7d vs prior 7d. This is what ranks movers."""
        return self.last7 - self.prior7

    @property
    def delta_pct(self) -> Optional[float]:
        if self.prior7 <= 0:
            return None          # undefined, not "infinite growth"
        return (self.last7 - self.prior7) / self.prior7 * 100.0

    @property
    def direction(self) -> str:
        if self.delta_abs > 0:
            return "up"
        if self.delta_abs < 0:
            return "down"
        return "flat"

    @property
    def top_service(self) -> str:
        if not self.services:
            return "—"
        return max(self.services.items(), key=lambda kv: kv[1])[0]

    @property
    def top_service_share(self) -> float:
        if not self.services or self.total <= 0:
            return 0.0
        return max(self.services.values()) / self.total * 100.0


@dataclass
class OrgSpend:
    """The full org picture for one query."""

    accounts: List[AccountSpend]
    linked_accounts: int
    window_days: int
    data_through: date
    dates: List[date] = field(default_factory=list)
    month_to_date: float = 0.0
    prior_month_total: float = 0.0
    budget_monthly: Optional[float] = None
    profile: str = ""

    # --- derived -----------------------------------------------------------
    @property
    def total(self) -> float:
        return sum(a.total for a in self.accounts)

    @property
    def accounts_with_spend(self) -> List[AccountSpend]:
        return [a for a in self.accounts if a.total > 0]

    @property
    def run_rate_daily(self) -> float:
        """Average daily spend over the trailing 7 days across the org."""
        last7 = sum(a.last7 for a in self.accounts)
        return last7 / 7.0 if last7 else 0.0

    @property
    def days_remaining_in_month(self) -> int:
        last_day = calendar.monthrange(self.data_through.year,
                                       self.data_through.month)[1]
        return max(0, last_day - self.data_through.day)

    @property
    def projected_month_end(self) -> float:
        """MTD plus trailing run-rate for the days left. Deliberately simple:
        a forecast a reviewer can reproduce on a napkin beats one they can't
        argue with. Swap in ce:GetCostForecast when you want AWS's model."""
        return self.month_to_date + self.run_rate_daily * self.days_remaining_in_month

    @property
    def projection_vs_prior_month_pct(self) -> Optional[float]:
        if self.prior_month_total <= 0:
            return None
        return (self.projected_month_end - self.prior_month_total) / self.prior_month_total * 100.0

    @property
    def budget_used_pct(self) -> Optional[float]:
        if not self.budget_monthly:
            return None
        return self.projected_month_end / self.budget_monthly * 100.0

    def concentration(self, top_n: int = 3) -> float:
        """Share of org spend held by the top N accounts. Tells you whether
        cost governance should be targeted or broad."""
        if self.total <= 0:
            return 0.0
        ranked = sorted(self.accounts, key=lambda a: a.total, reverse=True)
        return sum(a.total for a in ranked[:top_n]) / self.total * 100.0

    def movers(self, min_abs: float = 250.0) -> List[AccountSpend]:
        """Accounts whose week-over-week dollar change clears the floor,
        ranked by magnitude. Percent is reported, never ranked on."""
        moved = [a for a in self.accounts if abs(a.delta_abs) >= min_abs]
        return sorted(moved, key=lambda a: abs(a.delta_abs), reverse=True)

    def suppressed_movers(self, min_abs: float = 250.0) -> int:
        return sum(1 for a in self.accounts
                   if 0 < abs(a.delta_abs) < min_abs)

    def group_by(self, dimension: str) -> Dict[str, float]:
        """Aggregate spend by team / ou / environment / account / service."""
        out: Dict[str, float] = {}
        if dimension == "service":
            for a in self.accounts:
                for svc, amt in a.services.items():
                    out[svc] = out.get(svc, 0.0) + amt
        elif dimension == "account":
            for a in self.accounts:
                key = a.name or a.account_id
                out[key] = out.get(key, 0.0) + a.total
        else:
            attr = {"team": "team", "ou": "ou",
                    "environment": "environment"}[dimension]
            for a in self.accounts:
                label = getattr(a, attr) or "Unallocated"
                out[label] = out.get(label, 0.0) + a.total
        return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))

    def daily_series(self, top_n: int = 5) -> Dict[str, List[float]]:
        """Per-day series for the top N accounts plus a rolled-up 'Other'."""
        ranked = sorted(self.accounts, key=lambda a: a.total, reverse=True)
        series: Dict[str, List[float]] = {}
        for a in ranked[:top_n]:
            if a.daily:
                series[a.name or a.account_id] = a.daily
        rest = ranked[top_n:]
        if rest and any(a.daily for a in rest):
            n = len(self.dates)
            other = [0.0] * n
            for a in rest:
                for i, v in enumerate(a.daily[:n]):
                    other[i] += v
            series["Other"] = other
        return series


class OrgSpendProvider(Protocol):
    """Duck-typed provider protocol. Implement `fetch()` returning OrgSpend
    and, if you want cache correctness, a `cache_key` property that changes
    when the provider's data would change."""

    cache_key: str

    def fetch(self, profile: str, window_days: int) -> OrgSpend: ...


# ============================================================================
# OWNERSHIP MAP
# Account ID -> team / OU / environment. In production, hydrate this from
# account tags (organizations:ListTagsForResource) or a checked-in owners map
# reviewed by the platform team. Unmapped accounts fall to "Unallocated",
# which is itself a useful number to show — it measures tagging hygiene.
# ============================================================================
DEFAULT_OWNERS: Dict[str, Dict[str, str]] = {
    "286668516930": {"name": "dp-team-3pm",       "team": "Data platform",   "ou": "Engineering", "environment": "nonprod"},
    "417220985531": {"name": "dp-prod-analytics", "team": "Data platform",   "ou": "Engineering", "environment": "prod"},
    "551903772284": {"name": "dp-prod-ingest",    "team": "Data platform",   "ou": "Engineering", "environment": "prod"},
    "684412009377": {"name": "gov-prod-core",     "team": "Governance",      "ou": "Product",     "environment": "prod"},
    "739104558216": {"name": "gov-nonprod-sbx",   "team": "Governance",      "ou": "Product",     "environment": "sandbox"},
    "802355471190": {"name": "audit-prod-eu",     "team": "Audit",           "ou": "Product",     "environment": "prod"},
    "918273645500": {"name": "audit-nonprod",     "team": "Audit",           "ou": "Product",     "environment": "nonprod"},
    "104857600321": {"name": "shared-networking", "team": "Shared services", "ou": "Platform",    "environment": "prod"},
    "220099887766": {"name": "shared-securityhub","team": "Shared services", "ou": "Platform",    "environment": "prod"},
}


def resolve_owner(
    account_id: str,
    owners: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, str]:
    owners = owners if owners is not None else DEFAULT_OWNERS
    return owners.get(account_id, {
        "name": account_id, "team": "Unallocated",
        "ou": "Unallocated", "environment": "unknown",
    })


# ============================================================================
# DEMO PROVIDER — deterministic, offline
# Kept in the module so the page's small-N fallback + design decisions can
# be exercised without needing a payer profile.
# ============================================================================
_SERVICE_MIX = {
    "prod":    [("Amazon Bedrock AgentCore", .34), ("Amazon RDS", .19), ("AWS Lambda", .14),
                ("Amazon S3", .12), ("Amazon CloudWatch", .11), ("Amazon DynamoDB", .10)],
    "nonprod": [("Amazon CloudWatch", .30), ("AWS Lambda", .22), ("Amazon S3", .18),
                ("Amazon Bedrock AgentCore", .16), ("Amazon DynamoDB", .14)],
    "sandbox": [("Amazon CloudWatch", .38), ("Amazon EC2", .26), ("AWS Lambda", .20),
                ("Amazon S3", .16)],
}

# account_id -> (daily base $, week-over-week drift, open anomalies)
_DEMO_SHAPE = {
    "417220985531": (2_050, +0.38, 2),
    "551903772284": (1_180, +0.04, 0),
    "684412009377": (1_040, -0.02, 1),
    "802355471190": (  620, -0.17, 0),
    "739104558216": (  180, +0.52, 3),
    "918273645500": (  240, +0.01, 0),
    "104857600321": (  310, -0.03, 0),
    "220099887766": (  190, +0.06, 0),
    "286668516930": (  151, -0.17, 1),
    "663401928475": (  145, +0.09, 0),   # deliberately absent from DEFAULT_OWNERS
}


class DemoOrgSpendProvider:
    """Deterministic synthetic org. Same numbers every run, so screenshots
    and demos are reproducible."""

    def __init__(
        self,
        account_ids: Optional[List[str]] = None,
        budget_monthly: Optional[float] = 208_000.0,
        seed: int = 7,
        as_of: Optional[date] = None,
    ):
        self.account_ids = account_ids or list(_DEMO_SHAPE.keys())
        self.budget_monthly = budget_monthly
        self.seed = seed
        self.as_of = as_of

    @property
    def cache_key(self) -> str:
        """Identity for the UI's cache. Two providers that would return
        different data must not share a key."""
        return (f"{','.join(sorted(self.account_ids))}"
                f"|{self.budget_monthly}|{self.seed}|{self.as_of}")

    def fetch(self, profile: str = "demo", window_days: int = 30) -> OrgSpend:
        rng = random.Random(self.seed)
        today = self.as_of or date.today()
        data_through = today - timedelta(days=1)       # Cost Explorer lag
        dates = [data_through - timedelta(days=window_days - 1 - i)
                 for i in range(window_days)]

        accounts: List[AccountSpend] = []
        for acct_id in self.account_ids:
            base, drift, anomalies = _DEMO_SHAPE.get(acct_id, (200, 0.0, 0))
            owner = resolve_owner(acct_id)
            daily: List[float] = []
            for i in range(window_days):
                # ramp the drift in over the final 7 days
                in_last7 = i >= window_days - 7
                factor = (1 + drift) if in_last7 else 1.0
                weekend = dates[i].weekday() >= 5
                seasonal = 0.82 if weekend else 1.0
                jitter = 1 + rng.uniform(-0.06, 0.06)
                daily.append(round(base * factor * seasonal * jitter, 2))

            total = round(sum(daily), 2)
            last7 = round(sum(daily[-7:]), 2)
            prior7 = round(sum(daily[-14:-7]), 2) if window_days >= 14 else 0.0

            env = owner["environment"]
            mix = _SERVICE_MIX.get(env, _SERVICE_MIX["nonprod"])
            services = {name: round(total * share, 2) for name, share in mix}

            accounts.append(AccountSpend(
                account_id=acct_id, name=owner["name"], team=owner["team"],
                ou=owner["ou"], environment=owner["environment"],
                total=total, last7=last7, prior7=prior7,
                services=services, daily=daily, open_anomalies=anomalies,
            ))

        org_daily_total = (
            sum(sum(a.daily) for a in accounts) / max(window_days, 1)
        )
        mtd = round(org_daily_total * data_through.day, 2)
        prior_month = round(org_daily_total * 30 * 0.89, 2)

        return OrgSpend(
            accounts=accounts,
            linked_accounts=max(len(accounts), 22),
            window_days=window_days,
            data_through=data_through,
            dates=dates,
            month_to_date=mtd,
            prior_month_total=prior_month,
            budget_monthly=self.budget_monthly,
            profile=profile,
        )


# ============================================================================
# COST EXPLORER PROVIDER — the real one
# Uses the repo's shared make_session() so aws-vault / SSO env-var creds are
# honoured. Falls back to boto3.Session for the demo case.
# ============================================================================
class CostExplorerProvider:
    """Real implementation against Cost Explorer + Organizations.

    IAM (read-only) needed on the management/payer profile:
        ce:GetCostAndUsage, ce:GetCostForecast, ce:GetAnomalies,
        organizations:ListAccounts, organizations:ListTagsForResource

    Notes that matter in production:
      - Cost Explorer data lags roughly 24h — always surface `data_through`.
      - GetCostAndUsage grouped by LINKED_ACCOUNT paginates; follow
        NextPageToken rather than capping at the first page. The old page's
        "top 30" limitation was a pagination bug surfaced as UI copy.
      - Cache per (profile, window) with a TTL — Cost Explorer bills per
        request, so an uncached page refresh is a line item.
    """

    def __init__(
        self,
        owners: Optional[Dict[str, Dict[str, str]]] = None,
        budget_monthly: Optional[float] = None,
        include_service_mix: bool = True,
        fetch_org_tags: bool = True,
    ):
        self.owners = owners if owners is not None else DEFAULT_OWNERS
        self.budget_monthly = budget_monthly
        # Set False to save one Cost Explorer call per linked account
        # (n × ~$0.01) when the mix isn't rendered anywhere.
        self.include_service_mix = include_service_mix
        # When True, pulls team / OU / environment tags via
        # organizations:ListTagsForResource in parallel. Static `owners`
        # map is used as a fallback for accounts missing tags. Set False
        # for testing or environments where the permission is unavailable.
        self.fetch_org_tags = fetch_org_tags

    @property
    def cache_key(self) -> str:
        return (f"{len(self.owners)}|{self.budget_monthly}|"
                f"{self.include_service_mix}|{self.fetch_org_tags}")

    # Common tag-key aliases seen in the wild. First match wins per role.
    _TAG_ALIASES = {
        "team": ("Team", "team", "TeamName", "OwnerTeam", "Owner"),
        "ou": ("OU", "ou", "Department", "Org", "OrgUnit", "BusinessUnit"),
        "environment": ("Environment", "environment", "Env", "env", "Stage"),
        "name": ("Name", "AccountName", "name", "DisplayName"),
    }

    # Common environment tokens seen in AWS account names. Order matters:
    # longer/more specific tokens first so 'preprod' doesn't collide with
    # 'prod' when both would match.
    _ENV_TOKENS = [
        ("preprod", "preprod"), ("staging", "staging"), ("stage", "staging"),
        ("nonprod", "nonprod"), ("dev", "dev"), ("test", "test"),
        ("qa", "qa"), ("sandbox", "sandbox"), ("sbx", "sandbox"),
        ("tools", "tools"), ("shared", "shared"),
        ("prod", "prod"), ("prd", "prod"),
    ]
    # Team-name prefixes we know about. Extend this — or replace with a
    # `Team=` tag — when we get real tagging in place.
    _TEAM_PREFIX = {
        "dp": "Data platform",
        "dp-team-3pm": "Data platform",
        "gov": "Governance",
        "audit": "Audit",
        "shared": "Shared services",
        "control-tower": "Platform",
        "connector-service": "Connector service",
        "risk-manager": "Risk manager",
        "data-platform": "Data platform",
    }

    def _parse_account_name(self, name: str) -> Dict[str, str]:
        """Best-effort ownership inference from just an account name.

        Handles Diligent-style names like 'dp-team-3pm-nonprod',
        'diligent-audit-prod-eu', 'control-tower'. Returns only the fields
        we could infer — never fabricates a team from thin air. Used as a
        second-tier fallback after tags/static map.
        """
        if not name:
            return {}
        lower = name.lower()

        picked: Dict[str, str] = {"name": name}

        # Environment: match against known tokens, longest first.
        for token, canonical in self._ENV_TOKENS:
            if f"-{token}-" in f"-{lower}-":
                picked["environment"] = canonical
                break

        # Team: longest prefix match. Try progressively shorter prefixes
        # of the hyphenated name — 'dp-team-3pm-nonprod' checks
        # 'dp-team-3pm-nonprod' → 'dp-team-3pm' → 'dp-team' → 'dp'.
        parts = lower.split("-")
        # Skip 'diligent' if it's the first segment — it's a global brand
        # prefix that adds no ownership info.
        if parts and parts[0] in ("diligent", "dil"):
            parts = parts[1:]
        for i in range(len(parts), 0, -1):
            candidate = "-".join(parts[:i])
            if candidate in self._TEAM_PREFIX:
                picked["team"] = self._TEAM_PREFIX[candidate]
                break

        return picked

    def _tags_to_owner(
        self,
        tags: Dict[str, str],
        account_id: str,
        account_meta: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Map an account's tags → owner shape. Ownership precedence:
          1. Explicit tags (Team, OU, Environment, Name…)
          2. Account NAME from Organizations, parsed for team+environment
          3. Static DEFAULT_OWNERS map
          4. Unallocated (still surfaced — a tagging-hygiene signal)
        """
        static = resolve_owner(account_id, self.owners)
        parsed = self._parse_account_name(
            (account_meta or {}).get("name", "")
        )

        picked: Dict[str, str] = {}
        for role, aliases in self._TAG_ALIASES.items():
            for key in aliases:
                if key in tags and tags[key]:
                    picked[role] = tags[key]
                    break

        return {
            "name": (
                picked.get("name")
                or parsed.get("name")
                or static.get("name", account_id)
            ),
            "team": (
                picked.get("team")
                or parsed.get("team")
                or static.get("team", "Unallocated")
            ),
            "ou": picked.get("ou") or static.get("ou", "Unallocated"),
            "environment": (
                picked.get("environment")
                or parsed.get("environment")
                or static.get("environment", "unknown")
            ),
        }

    def _fetch_org_account_meta(
        self, profile: str,
    ) -> Dict[str, Dict[str, str]]:
        """One paginated call to organizations:ListAccounts.

        Returns {account_id: {"name": "...", "email": "...", "status": "..."}}
        for every account in the org. This is nearly always granted on payer
        roles (unlike ListTagsForResource which frequently isn't), and the
        Name field is usually already meaningful — e.g. 'dp-team-3pm-nonprod'
        or 'diligent-audit-prod-eu' — so we can use it as an ownership
        fallback even when tags are empty.
        """
        try:
            orgs = self._client(profile, "organizations")
            paginator = orgs.get_paginator("list_accounts")
            out: Dict[str, Dict[str, str]] = {}
            for page in paginator.paginate():
                for a in page.get("Accounts", []):
                    out[a["Id"]] = {
                        "name": a.get("Name", "") or "",
                        "email": a.get("Email", "") or "",
                        "status": a.get("Status", "") or "",
                    }
            return out
        except Exception:  # noqa: BLE001
            return {}

    def _fetch_org_tags_bulk(
        self, profile: str, account_ids: List[str],
    ) -> Dict[str, Dict[str, str]]:
        """Return {account_id: {tag_key: tag_value}} for the given accounts.

        Uses `organizations:ListTagsForResource` in parallel. Any account
        the API rejects (permissions, throttle, missing) silently returns
        an empty tag dict so a single failure never takes down the page.
        Skipped when the payer profile can't reach Organizations at all —
        that's normal outside the management account.
        """
        try:
            orgs = self._client(profile, "organizations")
        except Exception:  # noqa: BLE001
            return {}

        def _one(acct_id: str) -> tuple[str, Dict[str, str]]:
            try:
                resp = orgs.list_tags_for_resource(ResourceId=acct_id)
                return acct_id, {
                    t["Key"]: t["Value"] for t in resp.get("Tags", [])
                }
            except Exception:  # noqa: BLE001
                return acct_id, {}

        workers = min(8, max(1, len(account_ids)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(pool.map(_one, account_ids))

    def _client(self, profile: str, service: str):
        # Late import so this module stays importable without boto3
        # installed (e.g. for the demo provider in tests).
        from src.aws.session import make_session
        session = make_session(profile)
        # Cost Explorer + Organizations are global — we pin us-east-1
        # since that's where CE endpoints live.
        return session.client(service, region_name="us-east-1")

    def fetch(self, profile: str, window_days: int = 30) -> OrgSpend:
        ce = self._client(profile, "ce")
        end = date.today()
        start = end - timedelta(days=window_days)

        daily_by_account: Dict[str, Dict[str, float]] = {}
        token = None
        while True:
            kwargs = dict(
                TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
                Granularity="DAILY",
                Metrics=["UnblendedCost"],
                GroupBy=[{"Type": "DIMENSION", "Key": "LINKED_ACCOUNT"}],
            )
            if token:
                kwargs["NextPageToken"] = token
            resp = ce.get_cost_and_usage(**kwargs)
            for period in resp.get("ResultsByTime", []):
                day = period["TimePeriod"]["Start"]
                for group in period.get("Groups", []):
                    acct = group["Keys"][0]
                    amt = float(group["Metrics"]["UnblendedCost"]["Amount"])
                    daily_by_account.setdefault(acct, {})[day] = amt
            token = resp.get("NextPageToken")
            if not token:
                break

        dates = [start + timedelta(days=i) for i in range(window_days)]

        # Ownership signal has three tiers (best to worst):
        #   1. Real AWS tags (organizations:ListTagsForResource per account,
        #      in parallel). Requires the payer to have this permission —
        #      NOT granted by default on many Diligent payer roles.
        #   2. Account NAMES from organizations:ListAccounts (single
        #      paginated call). ~always granted on payer roles, and
        #      Diligent-style names like 'dp-team-3pm-nonprod' already
        #      encode team + environment. This is what stops every
        #      account from showing 'Unallocated' when tag reads are
        #      denied.
        #   3. Static DEFAULT_OWNERS map — hand-curated fallback for a
        #      few known account IDs.
        account_ids = list(daily_by_account.keys())
        account_meta = self._fetch_org_account_meta(profile)
        tags_by_account: Dict[str, Dict[str, str]] = (
            self._fetch_org_tags_bulk(profile, account_ids)
            if self.fetch_org_tags else {}
        )

        # Build the account list without service mix (fast — no extra
        # Cost Explorer calls). Ownership comes from tags → account name
        # → static map, cascading via _tags_to_owner().
        accounts: List[AccountSpend] = []
        for acct_id, by_day in daily_by_account.items():
            owner = self._tags_to_owner(
                tags_by_account.get(acct_id, {}),
                acct_id,
                account_meta.get(acct_id),
            )
            daily = [by_day.get(d.isoformat(), 0.0) for d in dates]
            accounts.append(AccountSpend(
                account_id=acct_id, name=owner["name"], team=owner["team"],
                ou=owner["ou"], environment=owner["environment"],
                total=sum(daily), last7=sum(daily[-7:]),
                prior7=sum(daily[-14:-7]),
                services={}, daily=daily,
            ))

        # Service mix requires one Cost Explorer call per account. On a
        # payer with 20+ linked accounts this dominates wall-clock, so
        # fan them out in parallel. The boto3 CE client is documented as
        # thread-safe for concurrent operation calls on the same instance.
        # Concurrency capped at 8 to stay under CE's per-account rate
        # limits with headroom.
        if self.include_service_mix and accounts:
            def _fetch_mix(acct_id: str) -> tuple[str, Dict[str, float]]:
                try:
                    return acct_id, self._services_for(
                        ce, acct_id, start, end,
                    )
                except Exception:  # noqa: BLE001
                    # Skip mix rather than crash the whole page.
                    return acct_id, {}

            workers = min(8, len(accounts))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                mix_by_account = dict(
                    pool.map(_fetch_mix, [a.account_id for a in accounts])
                )
            for acct in accounts:
                acct.services = mix_by_account.get(acct.account_id, {})

        # Prefer the org's authoritative account count if we already have
        # it from _fetch_org_account_meta (single call above). Falls back
        # to counting accounts with spend when the payer profile can't
        # reach Organizations.
        linked = len(account_meta) or len(accounts)

        data_through = end - timedelta(days=1)
        mtd = sum(
            v for a in accounts
            for d, v in zip(dates, a.daily)
            if d.month == data_through.month
        )
        return OrgSpend(
            accounts=sorted(accounts, key=lambda a: a.total, reverse=True),
            linked_accounts=linked, window_days=window_days,
            data_through=data_through, dates=dates, month_to_date=mtd,
            prior_month_total=0.0, budget_monthly=self.budget_monthly,
            profile=profile,
        )

    @staticmethod
    def _services_for(ce, account_id: str, start: date, end: date) -> Dict[str, float]:
        resp = ce.get_cost_and_usage(
            TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
            Granularity="MONTHLY", Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
            Filter={"Dimensions": {"Key": "LINKED_ACCOUNT",
                                    "Values": [account_id]}},
        )
        out: Dict[str, float] = {}
        for period in resp.get("ResultsByTime", []):
            for group in period.get("Groups", []):
                name = group["Keys"][0]
                out[name] = out.get(name, 0.0) + float(
                    group["Metrics"]["UnblendedCost"]["Amount"])
        return out


def auto_provider(
    budget_monthly: Optional[float] = None,
) -> Callable[[str, int], OrgSpend]:
    """Callable that tries the real Cost Explorer provider first and falls
    back to the deterministic demo if CE / Organizations don't work.

    Used by the page so a non-payer profile (typical for sandbox usage)
    still shows something instead of a red error card.
    """
    real = CostExplorerProvider(budget_monthly=budget_monthly)
    demo = DemoOrgSpendProvider(budget_monthly=budget_monthly)

    class _AutoProvider:
        # Cache key mixes both so the page's cache knows which one was used.
        cache_key = f"auto|{real.cache_key}|{demo.cache_key}"

        def fetch(self, profile: str, window_days: int) -> OrgSpend:
            try:
                return real.fetch(profile, window_days)
            except Exception:  # noqa: BLE001
                # Mark it in the profile string so the UI can call it out.
                spend = demo.fetch(profile, window_days)
                spend.profile = f"{profile} (demo fallback)"
                return spend

    return _AutoProvider()
