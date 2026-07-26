# CostSense

**AI-native FinOps for Diligent.** A Streamlit app that combines Amazon
Bedrock Claude, live AWS APIs, and GitHub PR history to forecast cost,
explain movement, and rank cost-cutting actions — grounded in real
account data, never hallucinated.

Built for the Diligent Hackathon 2026 — submitted under **MCP for
Connected Compliance**.

- **Live deployment:** http://costsense-alb-257440129.us-west-2.elb.amazonaws.com
  (ECS Fargate + ALB, hosts 5 workload accounts via cross-account IAM)
- **Design doc:** [docs/DESIGN.md](docs/DESIGN.md)
- **Architecture deep-dive:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Team AIPDLC log:** [AIPDLC.md](AIPDLC.md)

---

## What it does — in one screenshot per page

| Page | Answers the question |
|---|---|
| 🤖 **CostSense AI** ([app.py](src/dashboard/app.py)) | *"Ask me anything about our AWS spend."* Chat agent with 23 read-only AWS + GitHub tools. |
| 📈 **Dashboard** ([2_Dashboard.py](src/dashboard/pages/2_Dashboard.py)) | *"What will this account cost next week? What's driving it?"* 7-day forecast + walk-forward backtest + PR overlay. |
| 🔍 **PR Predictor** ([3_PR_Predictor.py](src/dashboard/pages/3_PR_Predictor.py)) | *"What will this specific PR do to our bill?"* Paste any PR URL → grounded impact analysis. |
| ⚠️ **Anomalies** ([4_Anomalies.py](src/dashboard/pages/4_Anomalies.py)) | *"Where's the low-hanging fruit?"* Full-account sweep → ranked cards with 2-3 fix approaches each. |
| 🌐 **Org-Level Impact** ([5_Org_Level_Impact.py](src/dashboard/pages/5_Org_Level_Impact.py)) | *"Which accounts in our org moved most this week?"* Per-account rollup across the AWS Organization. |
| 🎯 **Future Forecast** ([7_Close_The_Loop.py](src/dashboard/pages/7_Close_The_Loop.py)) | *"What's the consolidated ledger of everything CostSense already recommended for this account?"* Aggregator — no forecasting, no projection. Mirrors what other pages produced. |

---

## Why CostSense (versus what's already out there)

- **Cost Explorer** shows history; it can't tie tomorrow's forecast to the PR that merged yesterday.
- **Cost Anomaly Detection** fires *after* the spike, without explaining the code cause.
- **Compute Optimizer** doesn't know about your open PRs.
- **General-purpose LLMs** guess — they don't have live read-only access to your account.

CostSense joins **billing history + live AWS state + GitHub PRs** and
lets Bedrock Claude reason over all three via a tool-use loop. Every
number in every answer comes from a real API call made seconds ago.

Full positioning + honest limits: [docs/WHY_COSTSENSE.md](docs/WHY_COSTSENSE.md).

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Log in to AWS (pick one)
aws sso login --profile dil-data-platform-dev
# or, if you use aws-vault:
# aws-vault exec dil-data-platform-dev -- streamlit run src/dashboard/app.py --server.port 8501

# 3. Set GitHub token (for repo tools + draft PRs)
export GITHUB_TOKEN=ghp_...   # or GH_TOKEN — never commit this
or set in an .env file as GITHUB_TOKEN or GH_TOKEN

# 4. Run
streamlit run src/dashboard/app.py --server.port 8501
```

Open http://localhost:8501, pick a profile from the sidebar, and start
clicking. Every page auto-discovers your available AWS profiles.

**GitHub credentials** — set **`GITHUB_TOKEN`** or **`GH_TOKEN`** in your
environment. The app **auto-loads** `.env` from the project root or `src/.env`
on startup (shell exports take precedence). Never commit tokens to the repo.

| Where | Scopes needed |
|---|---|
| **`<repo>/.env`** or **`<repo>/src/.env`** | `repo` (classic), or fine-grained **Contents: Read and write** + **Pull requests: Read and write** |
| Shell: `export GITHUB_TOKEN=…` before `streamlit run` | same |
| GitHub Actions / CI secrets | same |

Read-only features need read access. **Anomalies → Open draft PR** needs
write access. Use **Prepare draft PR** to preview the diff first; **Open draft
PR** only pushes after you confirm.

`gh auth login` still works as a fallback when no token env var is set.

**Bedrock access**: the app calls `us.anthropic.claude-sonnet-4-6`
inference profile in `us-west-2` on Diligent's shared Bedrock sandbox
(account `609400232087`). No API key needed if your SSO role can
`bedrock:InvokeModel`.

---

## Architecture at a glance

```
             ┌─────────────────────────────────────┐
             │      Streamlit UI (6 pages)         │
             └─────┬───────────────────┬───────────┘
                   │                   │
    ┌──────────────┘                   └───────────────┐
    ▼                                                  ▼
┌────────────────────┐                        ┌────────────────────┐
│ Forecast pipeline  │                        │  AI agents         │
│ (auto-tuned EWM,   │                        │  (chat, PR pred.,  │
│  walk-forward BT,  │                        │   anomaly scan)    │
│  PR overlay)       │                        │  23 tool registry  │
└─────────┬──────────┘                        └──────────┬─────────┘
          │                                              │
          ▼                                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Cost Explorer · CloudWatch · CloudTrail · Compute Optimizer   │
│  Budgets · Quotas · S3/DynamoDB · Pricing · Organizations      │
│  GitHub (via `gh` CLI) · Bedrock (Claude Sonnet 4.6 / Haiku)   │
└────────────────────────────────────────────────────────────────┘
```

**Read-only everywhere.** No `create_*`, `delete_*`, `update_*` calls
in the codebase. Every tool output passes through a secret scrubber
before Claude sees it.

Deep dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Multi-account deployment (ECS + ALB, 5 workload accounts)

The deployed app runs as a single ECS Fargate task behind an
Application Load Balancer and reads cost data from **5 linked Diligent
AWS workload accounts** by cross-account `sts:AssumeRole`.

```
                    Public URL
   http://costsense-alb-257440129.us-west-2.elb.amazonaws.com
                        │
                        ▼
                ┌───────────────┐
                │      ALB      │  ← WebSocket + HTTP
                └───────┬───────┘
                        │
                ┌───────▼───────┐
                │  ECS Fargate  │  ← Streamlit container
                │  (Task role)  │      (image in public ECR)
                └───────┬───────┘
                        │  sts:AssumeRole
       ┌────────┬───────┼───────┬────────┐
       ▼        ▼       ▼       ▼        ▼
    Account  Account Account Account  Account
   #1 role  #2 role #3 role #4 role  #5 role
       │        │       │       │        │
       └────────┴───────┼───────┴────────┘
                        ▼
        Cost Explorer · CloudWatch · Bedrock
```

Config lives in the ECS task-def as **`COSTSENSE_CROSS_ACCOUNT_ROLES`**
(comma-separated `arn|label` pairs). Set it locally too if you want the
same dropdown behavior without SSO:

```bash
export COSTSENSE_CROSS_ACCOUNT_ROLES=\
"arn:aws:iam::111111111111:role/costsense-reader|dil-team-hackfest,\
arn:aws:iam::222222222222:role/costsense-reader|dil-data-platform-dev"
```

Each account's role trusts the ECS task role and grants **read-only**
`ce:*`, `cloudwatch:GetMetricData`, `bedrock:InvokeModel`. No mutating
permissions anywhere.

Recreate this stack: `infra/deploy-public-ecr.sh` (initial ECR push) →
manual `aws ecs create-service` + ALB target-group wiring (documented
in [AIPDLC.md § 5.15](AIPDLC.md)). No CloudFormation — the org SCP
`p-uxu2m3ck` blocks `iam:CreateRole` from CFN in this account.

---

## Documentation set

| Doc | What's in it |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | System design + Mermaid diagrams: request flow, cross-account IAM, prediction pipeline, anti-hallucination guards |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | End-to-end technical walkthrough: pages, forecast engine, agents, tool registry, deployment topology |
| [docs/PAGES.md](docs/PAGES.md) | Feature-by-feature guide to each of the 6 pages + shared UI infra |
| [docs/DATA_FLOW.md](docs/DATA_FLOW.md) | Every data source, cache, and file on disk. Forecast JSON schema. Secret scrubbing details. Failure modes. |
| [docs/WHY_COSTSENSE.md](docs/WHY_COSTSENSE.md) | Positioning vs alternatives · design trade-offs · what CostSense honestly cannot do |
| [AIPDLC.md](AIPDLC.md) | Team AIPDLC log — per-contributor phase-by-phase development record |

---

## Forecast model — the short version

The Dashboard uses an **auto-tuned naive-heavy blend with a
regime-shift detector**:

```
if recent_5d_mean/prior_14d_mean <= 0.6 or >= 1.7:
    train_history = history >= shift_day        # regime detected
else:
    train_history = full history

level(day) = naive_weight * yesterday
           + (1 - naive_weight) * trimmed_mean_of_last_N_days
adjusted   = level * day_of_week_ratio + PR_delta_step_function
```

Every run searches **320 parameter combinations** (5 × 4 × 4 × 4) via
walk-forward cross-validation and picks the winner. The tuned
parameters are saved in the forecast JSON so any reviewer can audit
them.

The regime detector is the single biggest accuracy win in the
codebase. Measured on `dil-data-platform-dev` walk-forward (8 origins,
7-day stride):

| Metric | Without regime detector | With regime detector |
|---|---:|---:|
| Direction accuracy | 50% (4/8) | **75% (6/8)** |
| WAPE | 76.7% | **49.4%** |
| MAE per day | $248 | **$205** |

On stable accounts where no shift is detected, the code path is
identical to the un-detected case — safe by construction.

Why not Prophet by default? Because dev-account spend is dominated by
level shifts, not by seasonal patterns, and the naive-heavy blend +
regime detector adapts within 1 day where Prophet lags by 1-2 weeks.
Details in
[docs/ARCHITECTURE.md § 4](docs/ARCHITECTURE.md#4-forecast-engine--how-the-number-gets-made).

Reproduce these numbers for any account:

```bash
python -m scripts.test_forecast_accuracy --profile <name> --origins 8
```

---

## Honest limits

- **Cannot predict console-driven changes** (someone manually stops an
  RDS instance at midnight). Not visible in git.
- **~50% WAPE floor** on volatile dev accounts even with the regime
  detector (down from ~77% before). Half the variance isn't
  code-driven — it's ad-hoc load tests, scheduled workloads pausing,
  GuardDuty toggles, trial evaluations.
- **List prices only.** AWS Pricing API returns on-demand rates; we
  can't see your RI / Savings Plan discounts, so $ deltas are upper
  bounds.
- **Diff-only PR analysis.** We read code, we don't run it. A memory
  bump gets priced correctly; a subtle algorithmic regression won't be
  caught until it hits Cost Explorer.
- **No scheduler.** `.github/workflows/daily.yml` is a scaffold; every
  forecast is user-triggered. Adding a nightly run is a
  `secrets.AWS_ROLE_TO_ASSUME` config away.

Full list with context: [docs/WHY_COSTSENSE.md § 4](docs/WHY_COSTSENSE.md#4-what-costsense-honestly-cannot-do).

---

## Repository layout

```
costsense-forecast/
├── README.md                       # this file
├── requirements.txt
├── docs/                           # long-form docs (linked above)
├── src/
│   ├── dashboard/
│   │   ├── app.py                  # CostSense AI chat (entry point)
│   │   ├── nav.py                  # shared top-bar helper (top_bar / inject_css)
│   │   └── pages/
│   │       ├── 2_Dashboard.py
│   │       ├── 3_PR_Predictor.py
│   │       ├── 4_Anomalies.py
│   │       └── 5_Org_Level_Impact.py
│   ├── ai_agent/                   # Bedrock agents + tool registries
│   │   ├── bedrock_client.py       # shared boto3 client factory
│   │   ├── chat_agent.py           # 12-turn conversational agent
│   │   ├── anomaly_agent.py        # ranked-action generator
│   │   ├── narrator.py             # (used for other pages)
│   │   ├── aws_tools.py            # 9 core AWS tools
│   │   ├── aws_tools_broad.py      # 7 broad AWS tools + scrub()
│   │   └── github_tools.py         # 7 GitHub tools via `gh` CLI
│   ├── aws/                        # profile/CE/org helpers
│   ├── forecast/                   # ensemble model + walk-forward backtest
│   │                               #   ensemble.py = auto-tuner + regime detector
│   │                               #   backtest_replay.py = walk_forward()
│   ├── pr_scanner/                 # PR diff parser, pricing, CW lookup
│   └── pipeline/
│       └── run_daily.py            # forecast orchestrator
├── scripts/
│   └── test_forecast_accuracy.py   # walk-forward accuracy backtest
├── data/                           # gitignored: predictions/actuals/backtest
└── .github/workflows/daily.yml     # scaffold cron (inactive)
```

---

## Contributing / extending

- **Add an AWS tool** → drop a function + `TOOL_SPEC` into
  `src/ai_agent/aws_tools.py` and register it in `TOOLS`. Every agent
  auto-discovers it.
- **Add a GitHub tool** → same pattern in `github_tools.py`.
- **Swap forecast model** → add a new function to `src/forecast/` and
  wire the `model` param in `pipeline/run_daily.py`.
- **Add a page** → drop `src/dashboard/pages/N_Name.py`. Streamlit's
  auto-nav picks it up.

---

## License

Internal Diligent hackathon project. Not for external distribution.
