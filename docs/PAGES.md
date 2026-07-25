# CostSense — Page-by-Page Guide

*What each of the five pages does, how to use it, and what to expect.*

---

## Navigation

Streamlit builds the sidebar from filenames. The nav order is:

1. **app** — CostSense AI (chat)
2. **Dashboard** — cost forecast
3. **PR Predictor** — analyze a single PR
4. **Anomalies** — full-repo + full-AWS sweep
5. **Org Level Impact** — per-account rollup across the AWS Organization

Every page has its own **Controls** strip at the very top (a
collapsible expander). The header always shows the current selections
(e.g. `Controls · Account: dil-data-platform-dev · Model: Claude
Sonnet 4.6`) so you can see the active context without expanding.
Click it to switch the AWS profile, model, or any other input.

Switching account switches everything: which data is queried, which
saved forecast JSON is loaded from disk.

**Available accounts:**
- **Local dev:** whatever SSO profiles you have in `~/.aws/config`.
- **Deployed URL** (http://costsense-alb-257440129.us-west-2.elb.amazonaws.com):
  5 Diligent workload accounts — `dil-team-hackfest`, `dil-team-aura`,
  `dil-data-platform-dev`, `dil-connector-service-dev`,
  `dil-3rdparty-connector-discovery-dev`. The ECS task assumes
  a read-only role into each. See [DESIGN.md § 4](DESIGN.md#4-cross-account-request-flow).

---

## 1. CostSense AI (`app.py`)

### What it does
A conversational FinOps agent. Ask anything about your AWS account or
GitHub repos; the agent picks the right tools live and answers with
grounded numbers.

### Under the hood
- Bedrock Claude (Sonnet 4.6 by default; Haiku selectable)
- 23 read-only tools: AWS + GitHub
- Multi-turn tool-use loop, capped at 12 tool calls per question
- Full chat history preserved across turns for follow-ups
- Every tool output passes through the secret scrubber before the model
  sees it

### Try these
- *"What's my biggest cost driver in the last 14 days?"*
- *"Find idle resources I could shut down."*
- *"Are any of my Lambdas oversized?"*
- *"What changed in cost yesterday vs last Wednesday?"*
- *"Which S3 buckets have no lifecycle policy?"*
- *"Am I close to any service quotas?"*
- *"Look at the data-platform repo and find code paths that call DynamoDB
  in a loop."*

### What you'll see
- User bubble appears immediately when you submit
- Spinner shows *"Thinking — querying AWS / GitHub tools…"*
- Assistant reply with tool-call trace collapsed underneath (click 🔧 to
  see which tools ran and what they returned)

### Limits
- Bedrock call latency: 10-30s for simple questions, 60-120s for
  multi-tool questions
- No memory between sessions — clearing chat wipes history

---

## 2. Dashboard (`pages/2_Dashboard.py`)

### What it does
The forecasting home. Shows past cost, next-7-day prediction, walk-forward
backtest for credibility, and a deterministic explanation of what the
future will look like.

### Top-bar controls (click **Controls ▸** at the top to expand)
- **Account** — which AWS profile / account
- **Cutoff** — the "as of" date (defaults to today)
- **History (days)** — how far back to train (30-180d)
- **Forecast model** — `ewm` (default, empirically best) or `prophet`
- **Service filter** — forecast total spend or a single service
- **GitHub org** — defaults to `DiligentCorp`
- **Repos** — GitHub repos whose PRs affect this forecast
- **Base branch** — auto-discovered per repo (usually `dev` or `main`)
- **PR lookback (d)** — how far back to scan merged PRs
- **PR analyzer** — `hybrid` (LLM + regex), `llm` only, or `regex` only
- **Bedrock model** — Sonnet 4.6 default
- **Show backtest** — toggle the past-predictions overlay
- **Backtest origins / Stride (d)** — how many past origins to score

### Sections (top to bottom)

**Live account banner**
Shows account + profile + cutoff + history window so you always know
what's on screen.

**KPI row**
- Last 7d actual — $ from Cost Explorer
- Next 7d forecast — sum of the 7 future predicted days
- Forecast vs last 7d — $ and % delta
- Rolling 30d MAPE / walk-forward WAPE — accuracy metric

**Cost forecast chart** (main chart)
- Blue line: actual daily spend
- Purple line (dashed): baseline forecast
- Purple line (solid): adjusted forecast (baseline + PR delta)
- Orange dotted line: cumulative PR-attributable $ effect
- Red diamonds: past predictions from walk-forward
- Vertical dashed line: the cutoff between real data and forecast

**Forecast detail table**
Per-day breakdown: baseline / PR delta / adjusted / lower / upper.

**Cost drivers panel** (only when no service filter)
Which services moved most in the last 7 days vs the prior 7 days. Big
movers that aren't in the PR list are usually console changes or
trial expirations.

**Backtest — predicted vs actual**
Full-width chart with the past-prediction line and the actuals overlay,
plus the future 7-day prediction on the same axis. WAPE and MAE numbers
at the top.

**Rolling error trend**
A separate line chart showing how the 7-day rolling MAPE has drifted.
Spikes mean the model got surprised.

**What will happen next — and why**
Deterministic explanation computed from the forecast JSON (no Bedrock
call). Lists concrete reasons: recent trajectory, day-of-week pattern,
merged PR delta at cutoff, expected open-PR merges, and the confidence
band width.

### Data flow
- History → live Cost Explorer call (cached 10 min)
- Forecast → reads latest JSON from `data/<account>/predictions/`
- Backtest → runs the model in-memory at 6 past origins on every render
- No AI call on this page

### Limits
- If no forecast JSON exists for the selected service, the future line
  is hidden (with an info message) — the app doesn't fake a forecast

---

## 3. PR Predictor (`pages/3_PR_Predictor.py`)

### What it does
Paste any GitHub PR URL, get an AI-generated cost-impact prediction
grounded in real AWS usage data.

### How to use
1. Open the **Controls ▸** strip at the top and pick the AWS account
   the PR will run against (defaults to your first available profile)
   plus the Bedrock model (Sonnet default)
2. Paste `https://github.com/DiligentCorp/data-platform/pull/854` (or
   any PR URL you have `gh` access to) into the URL box
3. Click **Predict cost impact**
4. Wait 60-120s for Sonnet to read the diff and query CloudWatch

### Under the hood
- `agent.analyze_pr()` pre-extracts diff resources with regex first (so
  the LLM has a checklist)
- Runs the tool-use loop with a **minimum of 5 tool calls** (7 for a
  "neutral" verdict) to prevent shallow analyses
- If the model tries to bail with too few tool calls, it gets pushed back
  with *"you haven't gathered enough evidence"* and forced to keep working

### What you'll see
- **Direction banner** — ↗ increase, ↘ decrease, or → neutral
- **Metrics**: estimated daily impact, monthly extrapolation, tool calls
- **What this PR does to cost** — findings table
- **Recommendations to reduce cost further** — expandable cards with
  concrete follow-up changes

### Example (verified on real PRs)
PR #854 "Prod resizing L4 L5" on data-platform:
- 17 tool calls, direction=decrease, $-6.40/day
- Found bulkIngest Lambda memory reduced 10240→8192 MiB
- Grounded in real invocation + duration metrics from CloudWatch

### Limits
- **Cannot see runtime performance**. A memory bump gets priced against
  invocation count, but a subtle algorithmic regression won't be caught.
- **List prices only** — no RI/SP discount awareness.
- **Public / accessible PRs only** — the `gh` CLI must be able to read
  the diff.

---

## 4. Anomalies & Recommendations (`pages/4_Anomalies.py`)

### What it does
Full-repo + full-AWS sweep. Fires 12 parallel AWS calls and per-repo
GitHub calls, then asks Sonnet to produce ranked, actionable
recommendations.

### Top-bar controls (click **Controls ▸** to expand)
- **Account** — which AWS profile / account
- **Bedrock model** — Sonnet 4.6 default
- **GitHub org** — defaults to `DiligentCorp`
- **Repos to scan** — defaults to the repo matching the profile name
  (e.g. `dil-data-platform-dev` → `data-platform`)
- **Analyze** button

### What gets swept

**AWS sweep** (parallel, ~30s):
- Cost Explorer: 14-day cost by service, top 10 spenders
- Compute Optimizer: EC2 and Lambda rightsizing recommendations
- Resource inventory: Lambda, RDS, EC2, NAT Gateway, EBS, S3, DynamoDB
- Budgets, Service Quotas, S3 lifecycle policies

**Repo sweep** (parallel per repo):
- `infrastructure/config.json` (accounts + regions per env)
- Open PRs with metadata (review state, CI status, mergeable)
- IaC files touched in the last 30 days
- Files declaring EventBridge / cron schedules

### Output — every card has

- **Category** (Idle resource / Oversized / Log inefficiency / Missing
  lifecycle / Risky upcoming PR / Cost trending up)
- **Save $/day** metric
- **Confidence pill** (colored: 🟢 high / 🟡 medium / ⚪ low)
- **Issue** — one-line problem statement
- **Reason** — why it costs money
- **Recommendation** — the primary action
- **How to fix it** — expandable approach with an optional code snippet
  (bash / terraform / typescript / python)

### Prompt hardening
The system prompt forces every action to include an `approaches` array
with 2-3 entries. If the model returns without them, the agent retries
once with an explicit correction. The parser also has truncation
recovery: if the outer JSON envelope isn't closed (large responses hit
`max_tokens=16000`), it rebuilds the envelope from action-shaped
sub-objects.

### Limits
- Depends on the AWS profile's read permissions. If Compute Optimizer
  isn't opted in on the account, that finding is skipped.
- Repos with 500+ open PRs may get truncated in the summary sent to
  Sonnet — only the first 15-30 titles are shown to the model.
- Sonnet is stateless per Analyze click — the report cache is keyed on
  `(profile, repos, schema_version)`.

---

## 5. Org-Level Impact (`pages/5_Org_Level_Impact.py`)

### What it does
Per-account spend rollup across every linked account in your AWS
Organization. Uses Cost Explorer's `LINKED_ACCOUNT` dimension on the
management (payer) account.

### Top-bar controls (click **Controls ▸** to expand)
- **Management profile** — pick the payer / management account
  (defaults to `control-tower` if such a profile exists)
- **History (days)** — 7-90 days
- **Fetch top service** — checkbox, adds one CE call per account
  (slower but useful)
- **Fetch org spend** button

### What you get
- **Total org spend** for the window
- **Linked account count**, count with any spend
- **Per-account table** — top 30 by spend, with search to look up any
  other account by ID:
  - Account ID
  - `Nd total ($)`
  - Last 7d ($)
  - Prior 7d ($)
  - Trend (%) — last 7d vs prior 7d
  - Top service (if enabled)
- **Biggest movers** — top-3 trending up + top-3 trending down callouts

### How the underlying data works
- Cost Explorer on the payer account can see all linked accounts' spend
  via `GroupBy: [{Type: "DIMENSION", Key: "LINKED_ACCOUNT"}]`
- Organization account names come from `organizations:ListAccounts` when
  permitted; otherwise account IDs are shown

### Limits
- If the current role lacks `organizations:ListAccounts`, only account
  IDs display (no friendly names)
- One Cost Explorer paginated call for the base rollup; top-service
  fetch adds one more per account (parallelized 10-wide)
- Top-service is capped to the top 30 accounts to keep costs bounded

---

## Cross-cutting behaviors

### AWS SSO
Every page uses `boto3.Session(profile_name=…)`. If a profile's SSO
token has expired, Cost Explorer / STS calls fail with a friendly error
and a suggestion to run `aws sso login`.

### Caching
- **Cost Explorer history**: `@st.cache_data(ttl=600)` — 10 min
- **Cost Explorer by-service**: `@st.cache_data(ttl=1800)` — 30 min
- **Session-state** for AI reports on the Anomalies page (invalidated
  on Analyze click)
- **`@lru_cache`** on GitHub `gh_orgs()`, `repo_default_branch()`, etc.

### Secret scrubbing
Every AWS tool output — no matter which agent called it — passes through
`aws_tools_broad.scrub()` before Claude sees the data. If your account
happens to store an IAM policy document, a secret manager entry, or a
JWT in a tag, it comes back as `[REDACTED-BY-COSTSENSE]`.

### Cross-page independence
No page depends on another. You can open only Anomalies and it works.
You can open only Org-Level and it works. State is per-page in
`st.session_state`.
