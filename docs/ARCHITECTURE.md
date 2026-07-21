# CostSense — Architecture

*End-to-end technical walkthrough of the CostSense FinOps agent.*

---

## 1. What CostSense is

CostSense is an **AI-native FinOps assistant** that turns raw AWS billing data
plus GitHub pull-request history into three things a human FinOps engineer
would produce:

1. A **7-day cost forecast** for a selected AWS account, adjusted for the
   PRs that just merged and the ones about to land.
2. **Ranked cost-cutting recommendations** grounded in real AWS state
   (idle NAT gateways, oversized Lambdas, missing lifecycle policies).
3. **Natural-language explanations** — "why did cost spike on Jul 17?",
   "what will this PR do to cost?" — using Bedrock Claude with live
   read-only access to AWS.

It's a single Streamlit app with **five pages**, backed by AWS SDK calls,
GitHub CLI calls, and Bedrock invocations. No custom infrastructure. Runs
on-demand from the user's laptop.

---

## 2. High-level architecture

```
                       ┌─────────────────────────────────────┐
                       │        Streamlit UI (5 pages)       │
                       │   app.py + pages/2..5_*.py          │
                       └─────┬───────────────────┬───────────┘
                             │                   │
             ┌───────────────┘                   └───────────────┐
             ▼                                                   ▼
    ┌────────────────────┐                          ┌────────────────────┐
    │  Forecast pipeline │                          │   AI agents        │
    │  src/pipeline/     │                          │   src/ai_agent/    │
    │  src/forecast/     │                          │  • chat_agent      │
    │                    │                          │  • anomaly_agent   │
    │  • auto-tuned EWM  │                          │  • narrator        │
    │  • walk-forward    │                          │  • PR analyzer     │
    │    backtest        │                          │                    │
    │  • PR overlay      │                          │  Uses 23 read-only │
    └─────────┬──────────┘                          │  tools (below).    │
              │                                     └──────────┬─────────┘
              ▼                                                ▼
     ┌────────────────────────────────────────────────────────────────┐
     │                       Data adapters                            │
     │   src/aws/*  ·  src/pr_scanner/*  ·  src/ai_agent/*_tools.py   │
     └───────┬────────────────┬─────────────────┬────────────────────┘
             │                │                 │
             ▼                ▼                 ▼
     ┌──────────────┐   ┌────────────┐   ┌──────────────────┐
     │ AWS APIs     │   │ GitHub CLI │   │ Amazon Bedrock   │
     │ (boto3)      │   │ (`gh`)     │   │ (Claude Sonnet   │
     │              │   │            │   │  or Haiku)       │
     │ • CE         │   │ • PR list  │   │                  │
     │ • CloudWatch │   │ • PR diff  │   │ tool-use loop:   │
     │ • CloudTrail │   │ • repo API │   │ Claude picks     │
     │ • EC2/RDS/…  │   │            │   │ AWS+GH tools     │
     │ • Pricing    │   │            │   │ live at reason-  │
     │ • Org.       │   │            │   │ time             │
     └──────────────┘   └────────────┘   └──────────────────┘
```

Everything read-only. Nothing mutates AWS or GitHub state.

---

## 3. The five pages

| Page | Purpose | Bedrock calls per open? |
|---|---|---|
| **CostSense AI** (`app.py`) | Free-form chat: ask anything, agent picks tools | Only when you ask |
| **Dashboard** (`2_Dashboard.py`) | Cost forecast, backtest, "why" explanation | 0 (deterministic explanation) |
| **PR Predictor** (`3_PR_Predictor.py`) | Paste PR URL → cost impact + recs | 1 (multi-turn tool-use) |
| **Anomalies** (`4_Anomalies.py`) | Full-repo + full-AWS sweep → ranked actions | 1-2 (tool-use loop) |
| **Org-Level Impact** (`5_Org_Level_Impact.py`) | Per-account spend across the AWS Organization | 0 (pure CE calls) |

See [PAGES.md](PAGES.md) for feature-by-feature details.

---

## 4. Forecast engine — how the number gets made

### 4.1 Model

The forecast model is a **naive-heavy blend** that lives in
`src/forecast/ensemble.py`:

```
level(day) = naive_weight * yesterday_cost
           + (1 - naive_weight) * trimmed_mean_of_last_N_days
```

Then multiplied by a **day-of-week ratio** (learned from the last 4 weeks
of the same account) so weekends dip and weekdays peak the same way the
account historically does.

### 4.2 Auto-tuning (no hardcoded parameters)

Every time a forecast runs, the tuner searches a grid of
**5 × 4 × 4 × 4 = 320 parameter combinations** by walk-forward
cross-validation on that account's own history:

| Parameter | Search grid |
|---|---|
| `naive_weight` | `0.3, 0.5, 0.7, 0.85, 1.0` |
| `trim_window` (days) | `7, 14, 21, 28` |
| `trim_fraction` | `0.0, 0.1, 0.15, 0.2` |
| `decay_slope` | `0.0, 0.03, 0.05, 0.08` |

The winning combo is persisted in the forecast JSON under `tuned_params`
so a reviewer can audit exactly which parameters the model chose.

### 4.3 Why not Prophet by default?

We measured. On real Diligent dev-account data:

| Model | Direction accuracy | WAPE (walk-forward) |
|---|---:|---:|
| Prophet defaults | — | ~92% |
| Prophet tuned | — | ~60% |
| EWM (halflife=5) | — | ~53% |
| Fast-level blend (auto-tuned, no regime) | 50% (4/8) | ~77% |
| **Fast-level blend + regime detector (current default)** | **75% (6/8)** | **~49%** |

Fast-level wins because dev-account spend is dominated by **level shifts**
(GuardDuty toggled off, RDS stopped, someone stops a script), not by
seasonal patterns. Prophet's trend + weekly components anchor on old data
and lag the shift by 1-2 weeks. The naive-heavy blend adapts within 1 day,
and the regime detector (§4.4 below) shortens that even further when a
sudden shift lands.

Prophet is still available in the top-bar controls for accounts with a
stable weekly rhythm (steady prod workloads).

### 4.4 Regime-shift detector

Because our biggest failure mode was the model over-predicting for 5-7
days after a workload paused, `src/forecast/ensemble.py` includes a
lightweight regime-shift detector:

```
recent_mean  = mean(last 5 days of history)
baseline_mean = mean(prior 14 days before that)
shift?       = recent_mean/baseline_mean <= 0.6  (big drop)
            or recent_mean/baseline_mean >= 1.7  (big rise)
```

When a shift is detected, `forecast_auto` trains **only on days from
the shift day onward** — old-regime days no longer drag the trimmed
mean toward the wrong level. For 2-9 days post-shift (too few to run
the walk-forward tuner), the forecast is the shifted-window mean
directly.

This is the single biggest accuracy improvement in the current
codebase. Measured on `dil-data-platform-dev` walk-forward (8 origins,
7-day stride):

| Metric | Without regime detector | With regime detector | Δ |
|---|---:|---:|---:|
| Direction accuracy | 50% (4/8) | **75% (6/8)** | +25 pp |
| MAE / day | $248 | **$205** | -$43 |
| WAPE | 76.7% | **49.4%** | -27 pp |

On stable accounts where no shift is detected the code path is
identical to before — safe by construction.

### 4.5 Walk-forward backtest

`src/forecast/backtest_replay.py` runs the same model at N sampled past
dates (default 6, one per week), each training on data *strictly before*
that origin. The predicted-vs-actual comparison is the credibility check
users see under "Backtest — predicted vs actual" on the Dashboard.

The EWM path applies the same merged-PR + open-PR step deltas the live
future line uses, so the backtest overlay and the future forecast line
are computed by the same math — you can compare them apples-to-apples.

WAPE (weighted absolute percentage error) is the primary metric because
MAPE explodes on near-zero days.

To reproduce the numbers in §4.4 above (or check any other account)
run:

```bash
python -m scripts.test_forecast_accuracy --profile <name> --origins 8
```

Reports direction hits, per-direction precision/recall, MAE, and WAPE.

### 4.6 PR-driven forecast adjustment

When a merged PR modifies AWS resources, its estimated `$/day` delta is
added as a **step function** starting at the merge date, projected forward
into the future forecast. Open PRs get a probability-weighted expected
delta (approved+CI-passing ≈ 0.9, draft ≈ 0.1). This is what makes the
future line adjust to "PRs about to land."

---

## 5. AI agents — how Claude gets grounded

There are four Bedrock-calling agents, all wired to a single client
factory (`src/ai_agent/bedrock_client.py`) with a 300s read timeout.

| Agent | Entry point | System prompt shape |
|---|---|---|
| **CostSense AI chat** | `chat_agent.chat_step()` | "You are a FinOps analyst. Ground every answer in tools." |
| **PR Predictor** | `agent.analyze_pr()` | "You are a FinOps reviewer. Estimate cost impact of a diff." |
| **Anomalies scan** | `anomaly_agent.analyze_anomalies()` | "Produce ranked actions with 2-3 approaches each." |
| **Chart narrator** | `narrator.narrate()` (currently replaced by deterministic explainer) | *n/a on Dashboard* |

### 5.1 Tool inventory

Claude has **23 read-only tools** across three registries:

**AWS core (`aws_tools.py`, 9)**
- `cost_by_service`, `get_cloudwatch_metric`, `cloudtrail_lookup`
- `list_lambda_functions`, `list_rds_instances`, `list_ec2_instances`
- `list_nat_gateways`, `list_ebs_volumes`, `rightsizing_recommendations`

**AWS broad (`aws_tools_broad.py`, 7)**
- `compute_optimizer_ec2`, `compute_optimizer_lambda`
- `list_budgets`, `service_quotas_summary`
- `list_s3_buckets`, `s3_bucket_lifecycle`, `list_dynamodb_tables`

**GitHub (`github_tools.py`, 7)**
- `github_search_repositories`, `github_repo_info`, `github_list_dir`
- `github_get_file`, `github_search_code`
- `github_list_pull_requests`, `github_pr_diff`

Every call goes through **`scrub()`** in `aws_tools_broad.py` which
recursively strips anything looking like a secret before Claude sees it:

- **Keys** matching `secret|password|token|apikey|credential|private_key|access_key|policy_document|assume_role_policy_document`
- **Values** matching AWS access keys (`AKIA…`, `ASIA…`), JWTs, PEM
  private keys, and `aws_secret_access_key=…` lines

Redacted fields come back as `[REDACTED-BY-COSTSENSE]`.

### 5.2 Tool-use loop

Every agent runs a bounded loop:

```
loop up to N turns:
  response = bedrock.invoke_model(system, tools, messages)
  if response has tool_use blocks:
      call each tool (in parallel where independent)
      append tool_result and continue
  else:
      parse final text as JSON verdict
      return
```

Loop budget varies: **8** turns for narrator, **12** for anomaly agent,
**14-16** for the PR Predictor deep-analysis path. The anomaly agent
retries once if the returned JSON is missing required fields (like the
`approaches` array).

### 5.3 Why Bedrock Claude specifically

- **Available in Diligent's stack** via `us.anthropic.claude-sonnet-4-6`
  inference profile in `us-west-2`
- **Tool-use is first-class** — Sonnet reliably picks the right tool and
  chains 8-15 tool calls without going off-piste
- **Structured output** via JSON schemas in the system prompt (not
  perfect, but robust with our extraction + retry logic)
- **Haiku fallback** for cheap operations (chat, quick analysis)

---

## 6. Data flow, top to bottom

For details see [DATA_FLOW.md](DATA_FLOW.md).

**On page load (Dashboard):**

1. Streamlit reads sidebar state → resolves the AWS profile via `boto3.sts`
2. `_fetch_history()` calls Cost Explorer `get_cost_and_usage` for the
   selected window (cached 10 min per profile+cutoff+days+service)
3. `_fetch_by_service()` calls the same API with `GroupBy=SERVICE` (cached
   30 min) for the "cost drivers" panel
4. `_load_latest_forecast()` reads the newest forecast JSON on disk that
   matches the selected service filter
5. Walk-forward backtest re-runs the auto-tuned model at 6 past origins,
   in-memory — no AWS calls needed
6. Deterministic "Why this chart looks the way it does" explanation is
   computed from the forecast JSON's `tuned_params` + `pr_scan`

**On page load (Anomalies) — after user clicks Analyze:**

1. `sweep_account()` fires 12 parallel AWS calls (Cost Explorer,
   Compute Optimizer, resource inventory, budgets)
2. `sweep_repos()` fires parallel GitHub API calls (config.json, open PRs,
   recent IaC files, scheduled rules) for each selected repo
3. Both sweeps summarized into a compact JSON payload
4. `analyze_anomalies()` sends that payload to Bedrock Sonnet with the
   23-tool registry. Sonnet iterates 8-12 tool calls drilling into
   specific findings.
5. Response parsed into `AnomalyReport(summary, actions[])`, cached in
   session state, rendered as Issue/Reason/Recommendation cards with
   colored confidence pills and expandable code snippets.

**On page load (CostSense AI chat):**

1. User types a question
2. `chat_step()` sends message + full 23-tool spec to Sonnet
3. Sonnet loops tool-use (typically 3-8 calls) then produces a markdown reply
4. Tool call trace is stashed with the message for the "🔧 N AWS tool call(s)"
   expander users can open for transparency

---

## 7. Persistence

Everything the app writes lives under `data/<account_id>/`:

```
data/
├── pricing_cache.json           # AWS Pricing API results (long-lived)
└── <aws_account_id>/
    ├── predictions/
    │   └── forecast_YYYY-MM-DD[__<service>].json
    ├── actuals/
    │   └── YYYY-MM-DD.json      # per-day Cost Explorer totals
    └── backtest/
        └── score_YYYY-MM-DD.json # MAE / MAPE per scored day
```

`data/` is `.gitignore`d because forecast JSONs contain real per-day
dollar amounts.

The app **regenerates all of this on demand** — nothing depends on the
files existing.

---

## 8. Non-goals (intentional constraints)

- **CostSense does not mutate AWS.** No stop-instance, no delete-nat.
  Every tool is a `list_*` / `describe_*` / `get_*`.
- **CostSense does not commit code.** The PR Predictor reads diffs and
  proposes changes but never opens PRs.
- **CostSense does not learn.** No feedback loop, no fine-tuning. Every
  Bedrock call is stateless. This is deliberate — reproducibility beats
  novelty for FinOps.
- **CostSense does not run in the background.** No cron, no daemon. The
  user opens Streamlit; each page renders on-demand. Simple to reason
  about, simple to demo.

---

## 9. What CostSense cannot do — honest limits

See [WHY_COSTSENSE.md](WHY_COSTSENSE.md) for the fuller version. Short
version:

- **Cannot predict console-driven events.** Someone toggles GuardDuty at
  midnight, an RDS instance gets manually stopped, a trial expires — none
  of that is visible in git or in leading indicators.
- **Volatile dev accounts land at ~50% WAPE.** With the regime detector
  we measure ~49% WAPE / 75% direction accuracy on `dil-data-platform-dev`
  (was ~77% / 50% before). Half the daily variance on these accounts is
  still non-code-driven — a math floor no forecast can beat.
- **List prices only.** AWS Pricing API returns on-demand rates. We can't
  see your Reserved Instances or Savings Plan discounts, so $ deltas are
  upper bounds.
- **PR analysis is diff-only.** We read code, we don't run it. A memory
  bump gets priced against CloudWatch invocation counts; but a subtle
  algorithmic change (e.g. an N² loop) won't be caught until it hits
  Cost Explorer.

---

## 10. Deployment

- **Local:** `streamlit run src/dashboard/app.py --server.port 8501`
- **AWS credentials:** whatever `~/.aws/config` profiles the user has SSO
  for. Every page has a profile picker in the sidebar.
- **GitHub credentials:** `gh auth status` — CostSense shells out to the
  `gh` CLI for repo access.
- **Bedrock:** account `609400232087` (Diligent's shared Bedrock sandbox)
  is what we currently target; changeable at the top of each
  `*_agent.py` file.

No Docker image, no Kubernetes, no CI/CD required to run the demo. A
GitHub Actions workflow (`daily.yml`) exists in the repo as a scaffold
for a future scheduled forecast run, but is not currently active
(no `AWS_ROLE_TO_ASSUME` secret configured).

---

## 11. Extending CostSense

- **New AWS tool:** add a function + `TOOL_SPEC` dict to `aws_tools.py` or
  `aws_tools_broad.py`, register it in the `TOOLS` / `BROAD_TOOLS` map.
  Every agent picks it up automatically.
- **New GitHub tool:** same pattern in `github_tools.py` +
  `GITHUB_TOOLS`.
- **New forecast model:** add it to `src/forecast/ensemble.py` or a new
  file, wire the `model` param in `pipeline/run_daily.py`.
- **New page:** drop a `pages/N_Name.py` file — Streamlit auto-discovers
  it and adds it to the sidebar nav.

---

## Related docs

- [PAGES.md](PAGES.md) — feature-by-feature walkthrough of each page
- [DATA_FLOW.md](DATA_FLOW.md) — every data source, cache, and file on
  disk
- [WHY_COSTSENSE.md](WHY_COSTSENSE.md) — positioning vs Cost Explorer /
  Budgets / Cost Anomaly Detection, and what CostSense honestly can't do
