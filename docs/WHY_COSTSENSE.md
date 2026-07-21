# Why CostSense — Positioning, Trade-offs, and Honest Limits

*Written for hackathon judges: what CostSense actually adds beyond
what's already in the AWS console, and what it deliberately doesn't try
to do.*

---

## TL;DR

CostSense is the only tool in Diligent's environment that answers
**"what will this cost tomorrow, and which PRs are causing that?"** by
combining three signal sources that normally live in separate places:

1. **Historical AWS spend** (from Cost Explorer)
2. **Live AWS state** (from resource-describe APIs)
3. **In-flight and merged code changes** (from GitHub PRs and diffs)

No AWS-native product joins these three signals. That join — and the
7-day forward prediction that comes out of it — is CostSense's core
contribution.

---

## 1. What's already out there — and where each falls short

| Existing tool | What it does well | What it can't do |
|---|---|---|
| **AWS Cost Explorer** | Historical breakdowns, top spenders, service filters | No forward forecast tied to code changes. Its native "forecast" is a linear projection of past days — ignores merged PRs entirely. |
| **AWS Cost Anomaly Detection** | ML-based spike detection on billing data | Only fires *after* the spike lands. Doesn't explain *why*, doesn't link to code. |
| **AWS Budgets** | Threshold alerts, forecasted overrun warnings | No connection to PR-driven cost movement. Static thresholds, no "root cause." |
| **AWS Compute Optimizer** | Rightsizing recommendations for EC2, Lambda, EBS, RDS | Doesn't know about your open PRs. Can't tell you *which* recommendation is about to be moot because a PR changes it. |
| **CloudHealth / Vantage / CloudZero** | Multi-cloud FinOps SaaS with tag-based cost allocation | Requires a paid integration + trust boundary. Doesn't correlate to a specific PR in a specific repo. |
| **`git log` + Cost Explorer, manually** | Full flexibility | Painful. A senior engineer takes 30-60 min to correlate one cost spike to one PR. |
| **A general-purpose LLM (ChatGPT, plain Claude)** | Great reasoning | No live access to your account. Every answer is a guess unless you paste the data in. |

**CostSense fills the gap** by:
- Running the LLM *with* live read-only AWS tools (via Bedrock tool-use)
- Adding a PR-aware step function into the forecast
- Explaining the future line deterministically so users trust it

---

## 2. What CostSense adds

### 2.1 A forecast that responds to code, not just history

Every commercial forecast (including AWS's own) is a function of *past
spend*. If someone merged a memory-doubling PR yesterday, that impact
takes 3-5 days to show up in the historical trend the forecast is
extrapolating.

CostSense adds a **PR step function** on top of the baseline. Merged PRs
become instant step changes on the merge date; open PRs get a
probability-weighted expected value. The user can see the same day the
PR merges what its projected daily impact is.

### 2.2 A Bedrock agent grounded in read-only AWS + GitHub tools

Most "AI FinOps" tools are RAG over your billing PDFs. CostSense's chat
agent has 23 real tools (`cost_by_service`, `list_lambda_functions`,
`get_cloudwatch_metric`, `github_pr_diff`, …) and picks them
autonomously. Answers are grounded in numbers that came out of your
account in the last few seconds, not out of a stale index.

### 2.3 Ranked actions with 2-3 concrete approaches each

The Anomalies page doesn't just say "you have an idle NAT gateway" —
it produces:

- **Issue**: `NAT Gateway nat-0abc in us-east-1 processes < 1 GB/day`
- **Reason**: `Fixed $32/mo + $0.045/GB. Traffic doesn't justify keeping it.`
- **Recommendation**: `Delete or route via VPC endpoints`
- **How to fix — 3 approaches** with copy-paste Terraform / AWS CLI /
  console steps

Each card also carries a confidence pill (green/amber/gray) so the user
can triage.

### 2.4 Deterministic explanation of the forecast

The "What will happen next — and why" section on the Dashboard is
computed from the forecast JSON — no LLM call. This means:

- **Instant render** (no Bedrock latency)
- **Reproducible** — same JSON → same explanation, always
- **Auditable** — every bullet points to a concrete field in the JSON

We tried an LLM narrator first and rolled it back because judges asked
"how do we know the narrator isn't lying about the numbers?" A
deterministic explainer sidesteps that concern entirely.

---

## 3. Design decisions and their trade-offs

### 3.1 Streamlit vs. a full web app

**Chose Streamlit** because a hackathon deliverable needs to demo in
2 minutes without deploy pain. Trade-offs:
- No custom auth — pages inherit whoever ran `streamlit run`
- Session state resets on server restart
- Not designed for concurrent users at scale

**When to migrate**: if this becomes a team tool with 20+ concurrent
users, migrate the UI to a proper React frontend and keep the Python
backend as a FastAPI service.

### 3.2 EWM blend vs. Prophet vs. an LLM-generated forecast

**Chose an auto-tuned naive-heavy EWM blend + regime-shift detector.**
Empirically wins on dev-account data (~49% WAPE / 75% direction on
`dil-data-platform-dev` walk-forward vs. Prophet's ~60% WAPE with no
direction signal). Prophet's weekly + trend components anchor on old
data; the naive-heavy blend adapts to level shifts within 1 day, and
the regime detector catches sudden drops/rises immediately.

We considered asking an LLM to forecast directly. Rejected because:
- Non-reproducible (Sonnet gives different numbers on different runs)
- No confidence bands
- Can't backtest an LLM at 6 past origins in seconds

Auto-tuning searches 320 param combos per account. Trade-off: run
takes 30-60s longer than a fixed-param model. Worth it — no user ever
has to guess a parameter.

### 3.3 Bedrock Sonnet 4.6 vs. Haiku vs. GPT-4o

**Chose Bedrock Sonnet 4.6** as the default:
- **Bedrock**: keeps traffic inside AWS, no third-party API keys, uses
  Diligent's shared Bedrock sandbox account
- **Sonnet 4.6**: the current generation with best tool-use reliability
  for multi-turn (12+ tool calls) chains
- **Haiku fallback**: available for cheap operations (chat quick
  answers) via a model dropdown

Trade-offs: Sonnet is ~10× more expensive per call than Haiku. We use it
only for the tool-use agents where reliability matters; the deterministic
explainer + forecast pipeline don't call Bedrock at all.

### 3.4 Read-only, always

**No mutations, no exceptions.** Even when a user says "delete this NAT
gateway", CostSense produces the CLI/Terraform snippet and stops.
Trade-off: an operator still has to run the fix by hand.

This is deliberate. A tool that can spend money on your behalf is a
tool that will eventually spend money you didn't approve. The value of
never-mutates far outweighs the mild inconvenience of a manual
copy-paste.

### 3.5 Aggressive secret scrubbing (over-redaction)

The scrubber matches on the *substring* `password`, `token`, `secret`,
etc. — so a resource legitimately named `prod-password-service` gets
redacted to `[REDACTED-BY-COSTSENSE]`.

**Trade-off**: occasional false positives where the model can't see a
resource name.

We accept this because false negatives (leaking a real secret to
Bedrock, which then logs it) are a lot more expensive than false
positives.

---

## 4. What CostSense honestly cannot do

### 4.1 Predict console-driven changes

If someone opens the AWS console at 11 PM and stops an RDS instance,
CostSense will not see it coming. That change doesn't exist in git.
It'll show up in Cost Explorer the next day and CostSense will fold it
into the baseline going forward — but as a prediction, it's blind.

### 4.2 Beat ~50% WAPE on volatile dev accounts

Half of the day-to-day variance on Diligent's dev accounts is not
code-driven — it's script runs, ad-hoc load tests, GuardDuty toggles,
scheduled workloads pausing, trial evaluations. Even with the
regime-shift detector, no forecast that only reads git + billing
history can do better than ~50% WAPE here (measured 49.4% WAPE / 75%
direction accuracy on `dil-data-platform-dev`, down from 77% / 50%
before the detector).

On steady prod workloads with consistent weekly rhythm, WAPE typically
drops to 15-25%.

### 4.3 See runtime performance regressions

The PR Predictor reads the *diff* and queries *current* CloudWatch
metrics. It can tell you a Lambda got twice the memory. It cannot tell
you the code inside now runs an N² algorithm and will double your
invocation duration.

Runtime regressions surface in the *next-day* Cost Explorer data, which
is where the forecast picks them up. Not in the PR analysis.

### 4.4 Reflect Reserved Instance or Savings Plan discounts

The AWS Pricing API returns list on-demand rates. If your account has a
70% Compute Savings Plan, CostSense's $ deltas are 70% higher than what
you actually pay. This is a labeled upper bound — users see "list
price" in tooltips.

### 4.5 Handle 500+ open PRs per repo

The Anomalies page's PR summarizer trims the open-PR list to the first
15-30 titles that get sent to Sonnet. On a very high-throughput repo we
may miss risky PRs that happen to be lower in the list.

### 4.6 Run scheduled forecasts

There's no cron, no daemon, no scheduler. `daily.yml` exists in
`.github/workflows/` as a scaffold but requires
`secrets.AWS_ROLE_TO_ASSUME` to be configured — it's inactive by
default. Every forecast is user-triggered.

**When to add this**: if the tool ships to a team and someone wants
"send me a Slack notification if tomorrow's forecast is 20% above last
week's average", the pipeline is already ready — just wire the workflow
to a real IAM role and add a webhook step.

---

## 5. What's genuinely novel about CostSense

Four things you won't find in any AWS-native or SaaS FinOps product:

1. **PR-aware forecast overlay.** Merged PRs become deterministic step
   functions in the future forecast, and open PRs get
   probability-weighted expected values. Nothing else joins git and
   billing this closely.

2. **Auto-tuned baseline per account.** 320-combo walk-forward search
   picks the model parameters, so a volatile dev account and a stable
   prod account both get an honest baseline — no manual tuning.

3. **Regime-shift detector.** The single biggest accuracy improvement:
   when recent-5-day mean drops below 60% (or above 170%) of the prior
   14-day baseline, training gets truncated to post-shift days. Boosted
   direction accuracy from 50% → 75% on our dev-account test.

4. **Grounded LLM with 23 read-only tools.** The chat and anomaly
   agents don't hallucinate numbers because every claim is backed by a
   tool call whose output Claude just saw. And every tool output passes
   through secret scrubbing before Claude sees it.

---

## 6. How we'd measure success (post-hackathon)

If this ships as an internal tool, the metrics that matter are:

- **Time-to-find-cause for a cost spike.** Before: 30-60 min of manual
  correlation. Target with CostSense: <5 min from the Anomalies page.
- **PR-attributable $ delta detected vs. missed.** Track how many of a
  quarter's cost movements were flagged by CostSense's PR analysis
  before landing.
- **Forecast WAPE per account, per month.** Should trend down as the
  history window fills and dow-ratios stabilize.
- **Actions surfaced / actions taken.** Anomaly page is only useful if
  people act on the cards. Track click-through on the "How to fix it"
  expanders as a proxy.

---

## 7. Closing note for judges

CostSense is deliberately small in scope: a 5-page Streamlit app, one
forecast model, four AI agents, 23 tools. It doesn't try to be a
platform.

What it does try to do is answer three concrete questions well:

- **Yesterday** → why did cost move? (Anomalies + chat)
- **Today** → what's on fire? (Dashboard cost drivers, Org-level rollup)
- **Tomorrow** → what should we do? (Forecast + PR Predictor)

Every design decision — read-only, PR-aware, auto-tuned, Bedrock,
deterministic explainer — falls out of taking those three questions
seriously.
