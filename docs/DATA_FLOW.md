# CostSense — Data Flow

*Every data source, every cache, every file on disk. Nothing hidden.*

---

## 1. External data sources

### AWS

| API | Called from | Purpose |
|---|---|---|
| `sts.get_caller_identity` | `aws/profiles.py:resolve()` | Map profile → account id, verify SSO |
| `ce.get_cost_and_usage` | `aws/cost_explorer.py` | Historical daily cost (with optional `service` filter or `LINKED_ACCOUNT` groupby) |
| `ce.get_rightsizing_recommendation` | `ai_agent/aws_tools.py` | AWS's own EC2 rightsizing suggestions |
| `cloudwatch.get_metric_statistics` | `pr_scanner/cloudwatch_tool.py` | Lambda invocations/duration, log ingestion, RDS CPU, etc. |
| `cloudtrail.lookup_events` | `ai_agent/aws_tools.py` | Recent management-plane events (GuardDuty toggle, StopDBInstance) |
| `lambda.list_functions` | `ai_agent/aws_tools.py` | Runtime + memory + timeout inventory |
| `rds.describe_db_instances` | `ai_agent/aws_tools.py` | Class, engine, storage, status |
| `ec2.describe_instances` | `ai_agent/aws_tools.py` | Type, state, launch time, tags |
| `ec2.describe_nat_gateways` | `ai_agent/aws_tools.py` | Active NATs (each ~$32/month) |
| `ec2.describe_volumes` | `ai_agent/aws_tools.py` | EBS volumes, filterable by `state=available` (orphans) |
| `compute-optimizer.get_*_recommendations` | `ai_agent/aws_tools_broad.py` | AWS-managed rightsizing (EC2 + Lambda) |
| `budgets.describe_budgets` | `ai_agent/aws_tools_broad.py` | Budget alerts, current vs forecast |
| `service-quotas.list_service_quotas` | `ai_agent/aws_tools_broad.py` | Quota headroom check |
| `s3.list_buckets` + `get_bucket_lifecycle_configuration` | `ai_agent/aws_tools_broad.py` | Buckets and their transition rules |
| `dynamodb.list_tables` + `describe_table` | `ai_agent/aws_tools_broad.py` | Table size / billing mode / provisioned throughput |
| `pricing.get_products` | `pr_scanner/pricing.py` | Live on-demand rates for EC2/NAT (only in `us-east-1` and `ap-south-1` regions per AWS) |
| `organizations.list_accounts` | `aws/org_spend.py` | Optional — friendly names for LINKED_ACCOUNT ids. Falls back gracefully when denied. |
| `bedrock-runtime.invoke_model` | `ai_agent/bedrock_client.py` | All AI calls. Region `us-west-2`. Sonnet 4.6 or Haiku. |

Every AWS call is **read-only**. Zero `create_*`, `delete_*`, `update_*`,
`put_*`, `start_*`, `stop_*`, or `modify_*` in the codebase.

### GitHub

Everything routes through the `gh` CLI so it uses whatever token the
user is already logged in with:

| Tool | CLI command | Purpose |
|---|---|---|
| `github_search_repositories` | `gh api search/repositories` | Resolve short name → `org/name` |
| `github_repo_info` | `gh api repos/{owner}/{repo}` | Metadata (default branch, size, etc.) |
| `github_list_dir` | `gh api repos/{owner}/{repo}/contents/{path}` | Tree listing |
| `github_get_file` | Same, decode base64 | Read a file's contents |
| `github_search_code` | `gh api search/code` | Grep across a repo (rate-limited by GitHub) |
| `github_list_pull_requests` | `gh pr list --json …` | PR metadata (draft, review state, CI, mergeable) |
| `github_pr_diff` | `gh pr diff <n> --repo …` | Raw unified diff of a PR |
| `repos.py` helpers | `gh api user/orgs`, PR-author search | Populate org / repo dropdowns |

### Bedrock

Direct `invoke_model` calls via `boto3.client('bedrock-runtime',
region_name='us-west-2')`. No proxy, no LangChain, no LiteLLM. The
shared `bedrock_client.make_client()` factory sets:

- `read_timeout = 300s` (many tool-use turns can exceed default 60s)
- `connect_timeout = 15s`
- `retries.max_attempts = 2`

---

## 2. Caching layers

| Layer | Where | TTL | Scope |
|---|---|---|---|
| Streamlit `@st.cache_data` | `_fetch_history`, `_fetch_by_service` | 10-30 min | Per (profile, cutoff, days, service) |
| Session state | Anomaly reports, chat history, pending questions | Until browser tab closes | Per browser tab |
| `@lru_cache` | GitHub org list, default branches, profile resolution | Until process restart | Per Streamlit process |
| `data/pricing_cache.json` | Pricing API results | Long-lived (prices rarely change) | Shared across all runs |
| `data/<account>/actuals/<date>.json` | Per-day Cost Explorer totals | Persisted forever | Per account |

### Cache invalidation
- Restart Streamlit → clears all `@st.cache_data` and `@lru_cache`
- Delete `data/` → forces regeneration of everything
- Anomaly report cache uses a `_SCHEMA_VERSION` string in the key so
  schema changes automatically invalidate stale reports

---

## 3. Files written to disk

Every write happens inside `data/` (git-ignored):

```
data/
├── pricing_cache.json                       # AWS Pricing API results
└── <aws_account_id>/                        # e.g. 014666657409/
    ├── predictions/
    │   ├── forecast_2026-07-20.json         # total-account forecast
    │   └── forecast_2026-07-20__AWS_Lambda.json   # service-filtered
    ├── actuals/
    │   └── 2026-07-19.json                  # {day, amount_usd}
    └── backtest/
        └── score_2026-07-19.json            # {target_date, predicted, actual, ape}
```

**Multi-account (organization view)** uses a `+`-joined key when multiple
accounts are combined: `data/914553008430/...` for the payer account
alone, or `data/acct1+acct2/...` if combined.

---

## 4. Forecast JSON schema

The single most important artifact on disk. Every forecast run writes
one of these. Every read on the Dashboard consumes one:

```json
{
  "account_id": "014666657409",
  "profile": "dil-data-platform-dev",
  "run_cutoff": "2026-07-21",
  "history_days": 90,
  "service_filter": null,
  "model": "ewm",
  "tuned_params": {
    "naive_weight": 1.0,
    "trim_window": 7,
    "trim_fraction": 0.2,
    "decay_slope": 0.08,
    "tuning_wape": 0.4232,
    "tuning_days_scored": 28,
    "dow_ratios": {"0": 1.12, "1": 1.08, ...}
  },
  "pr_scan": {
    "repos": ["DiligentCorp/data-platform"],
    "base_branch": "dev",
    "lookback_days": 90,
    "analyzer": "hybrid",
    "llm_model": "us.anthropic.claude-sonnet-4-6",
    "impacts": [
      {
        "repo": "DiligentCorp/data-platform",
        "pr_number": 844,
        "pr_title": "…",
        "merged_at": "2026-07-10T09:04:10Z",
        "est_daily_delta_usd": -6.4,
        "llm_summary": "Resizing Lambda memory…",
        "changes": [{"resource_type": "aws_lambda_function", …}]
      }
    ]
  },
  "pr_daily_series": [
    {"day": "2026-07-10", "pr_cum_usd": -6.4},
    …
  ],
  "pr_delta_daily_usd_at_cutoff": -6.4,
  "open_pr_scan": {
    "count": 15,
    "total_expected_daily_delta_usd": 0.97,
    "prs": [
      {
        "repo": "DiligentCorp/data-platform",
        "pr_number": 877,
        "pr_title": "…",
        "est_daily_delta_usd": 1.49,
        "merge_probability": 0.65,
        "expected_daily_delta_usd": 0.97,
        "expected_merge_day": "2026-07-24",
        …
      }
    ],
    "daily_expected_delta": {"2026-07-22": 0.0, "2026-07-24": 0.97, …}
  },
  "forecast": [
    {
      "target_date": "2026-07-22",
      "baseline_usd": 47.10,
      "pr_delta_usd": 0.0,
      "adjusted_usd": 47.10,
      "lower_usd": 0.0,
      "upper_usd": 126.16
    },
    …
  ]
}
```

Every field feeds a specific UI element. See `PAGES.md` § Dashboard for
the mapping.

---

## 5. The "why this chart looks the way it does" explanation

Deterministic. No AI call. Reads directly from the forecast JSON and
`hist_df` (in-memory DataFrame from Cost Explorer):

| Bullet | Derived from |
|---|---|
| "Steady level / Trend continues" | `hist_df.tail(7).sum()` vs `sum(forecast.adjusted_usd)` |
| "Weekly rhythm" | `tuned_params.dow_ratios` — quotes the highest and lowest day |
| "Merged PR impact" | `pr_delta_daily_usd_at_cutoff` + count of non-zero `pr_scan.impacts` |
| "Upcoming PRs" | `open_pr_scan.total_expected_daily_delta_usd` + `count` |
| "Uncertainty band" | Average of `upper_usd - lower_usd` across the 7 forecast days |

This is why the section renders instantly — no Bedrock latency.

---

## 6. When does the model retrain?

- **On every `Run forecast` click** (pipeline). Auto-tuner walk-forward
  searches 320 param combos, picks the best, writes the new
  `forecast_YYYY-MM-DD.json`.
- **On every Dashboard page load** for the walk-forward backtest. The
  6-origin retrain runs *in-memory only* — doesn't write anything.
- **The regime-shift detector runs on every training pass** (both live
  forecasts and each origin of the backtest). It's a lightweight
  pandas check — no extra retrain cost. When a >=40% drop or >=70%
  rise is detected between the last 5 days and the prior 14 days,
  training gets truncated to post-shift days before the auto-tuner
  runs.

There's no daemon, no scheduled retrain, no online learning. Every run
is deterministic given the same history + same PRs.

To reproduce forecast accuracy numbers for any account:

```bash
python -m scripts.test_forecast_accuracy --profile <name> --origins 8
```

The test mirrors the Dashboard's walk-forward path exactly, reads
merged-PR steps from the newest saved forecast JSON on disk, and
prints direction accuracy + MAE + WAPE per origin.

---

## 7. Data privacy / redaction

Before Bedrock sees any AWS tool output, it passes through
`aws_tools_broad.scrub()`. This is a recursive function that:

1. **Blanks any dict key** matching this regex:
   ```
   secret | password | passwd | apikey | api_key | token
   | credential | privatekey | private_key | access_key
   | accesskey | salt | connection_?string | conn_?str
   | dsn | policy_?document | assume_?role_?policy_?document
   ```
2. **Regex-replaces any value** matching:
   - AWS access keys (`AKIA[0-9A-Z]{16}`)
   - AWS temp/STS keys (`ASIA[0-9A-Z]{16}`)
   - JWT tokens (three base64 segments separated by dots)
   - PEM private keys (`-----BEGIN … PRIVATE KEY-----`)
   - `aws_secret_access_key=…` config lines

Any redacted field comes back as `[REDACTED-BY-COSTSENSE]`. Claude's
system prompt tells it to treat these as inaccessible and not to demand
them.

**What this doesn't protect against:** if you name a resource
`prod-database-password-store` (the WORD "password" in the resource
name), that name will be scrubbed to `[REDACTED-BY-COSTSENSE]`. This is
intentional over-redaction — false positives are cheap; false negatives
leak credentials.

---

## 8. What goes into git

The initial commit + this branch include:

- `src/` — all source code
- `README.md`, `docs/*` — this documentation
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes `data/`, `.venv/`, `__pycache__/`
- `.github/workflows/daily.yml` — GitHub Actions scaffold (currently
  inactive; requires `secrets.AWS_ROLE_TO_ASSUME` to be configured)

**Never committed:**
- `data/` — real cost numbers
- `.env`, `.aws/`, `.streamlit/secrets.toml` — credentials

---

## 9. Failure modes and how they surface

| Failure | Where user sees it |
|---|---|
| SSO token expired | `st.error("Cost Explorer fetch failed: …")` on the affected page |
| Cost Explorer rate limit | Same. The `@cache_data` 10-min TTL mitigates. |
| Bedrock timeout / rate limit | `st.error("Agent error: bedrock invoke failed: …")` on the affected page |
| GitHub `gh` not logged in | `st.warning("gh CLI failed: …")` on Dashboard / Anomalies sidebar |
| Compute Optimizer not opted in | Silent — the sweep just returns an empty list, no crash |
| No forecast JSON on disk for the current service filter | Info message "No saved future forecast yet" |
| LLM returns bad JSON | Parser has truncation recovery + retry; falls back to plain text as a final resort |

The app never crashes silently; every exception surfaces to the user
with enough context to tell whether it's a config problem (fix your
env) or an AWS/Bedrock problem (retry later).
