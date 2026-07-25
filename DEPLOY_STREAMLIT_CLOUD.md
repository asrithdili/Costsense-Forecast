# Deploy CostSense to Streamlit Community Cloud

**Why this and not AWS App Runner:** the hackfest AWS organisation
has an SCP that blocks WebSocket upgrades on App Runner services.
Streamlit needs WebSockets to render the UI, so App Runner is not
a viable host regardless of container config. Details in the
`deploy-apprunner-public-ecr` branch commits — this file is the
alternative that actually works.

Streamlit Community Cloud is Streamlit's own hosting service. It
runs Python apps directly from a GitHub repo, WebSockets are
native, and the free tier is fine for a demo (no idle-cost, sleeps
after ~15 min of inactivity and wakes in ~30s on next request).

## Prerequisites

- Repo must be **public** — free tier requirement.
  (`asrithdili/Costsense-Forecast` is already public as of
  2026-07-25.)
- A **GitHub account** authorised to Streamlit Cloud via OAuth.
- **AWS credentials** for a limited-scope IAM user (see below).
- Optional: **GitHub personal access token** if you want the PR
  Predictor / Anomalies "read repo files" tools to work.

## Files in this repo that make Streamlit Cloud work

| File | Purpose |
|---|---|
| `streamlit_app.py` | Root entry point Streamlit Cloud picks up by default. Delegates to `src/dashboard/app.py` and promotes `st.secrets` into `os.environ` so the rest of the app doesn't need changes. |
| `requirements.txt` | Python deps — Streamlit Cloud installs these automatically. |
| `packages.txt` | apt-get deps (`libgomp1` for scikit-learn/lightgbm). Streamlit Cloud installs these too. |
| `.streamlit/secrets.toml.example` | Template listing every secret you must paste into the Streamlit Cloud secrets vault. |
| `.streamlit/config.toml` | Theme + client behaviour. Left alone — Streamlit Cloud's proxy handles WS correctly at defaults. |

## Step-by-step deploy

### 1. Create the IAM user for read-only AWS access

The Streamlit Cloud container needs long-lived credentials (no SSO
possible from a third-party host). Create a limited-scope user
in whichever AWS account you want the deployed app to see data
from — probably `dil-team-hackfest` (the current demo target)
though NB the SCP blocks `iam:CreateUser` in hackfest, so this
step may need to happen in `dil-team-aura` or a different
account without that SCP.

If `iam:CreateUser` works in `dil-team-aura`:

```bash
export AWS_PROFILE=dil-team-aura

aws iam create-user --user-name costsense-streamlit-cloud
aws iam put-user-policy \
  --user-name costsense-streamlit-cloud \
  --policy-name costsense-readonly \
  --policy-document file://infra/instance-role-policy.json
# (Uses the same JSON that's on the deploy-apprunner-public-ecr branch;
#  copy it locally first if you're on main.)

aws iam create-access-key --user-name costsense-streamlit-cloud
# Output includes AccessKeyId + SecretAccessKey. Save both — you
# CAN'T see the SecretAccessKey again after this call.
```

If `iam:CreateUser` is blocked everywhere you have access, ask
DevOps to create the user once with the read-only policy from
`infra/instance-role-policy.json` (that branch/file, or copy it
locally).

### 2. Deploy on share.streamlit.io

1. Go to https://share.streamlit.io
2. Sign in with GitHub (authorises Streamlit Cloud to see your repos)
3. Click **"New app"** in the top right
4. Fill in:
   - **Repository**: `asrithdili/Costsense-Forecast`
   - **Branch**: `main` (or whichever branch has this file)
   - **Main file path**: `streamlit_app.py`
   - **App URL**: pick a slug like `costsense-demo` — you'll get
     `https://costsense-demo.streamlit.app`
5. Click **"Advanced settings..."** → **Secrets** tab
6. Paste the contents of `.streamlit/secrets.toml.example` with the
   placeholder values replaced by the real AWS access key + secret.
7. Click **"Deploy"**
8. Wait ~2–5 minutes for the first build (installs `requirements.txt` +
   `packages.txt`). Subsequent redeploys after a git push take ~30s.

The URL is live as soon as the build finishes.

### 3. Verify

Load the URL. You should see the Diligent brand card in the sidebar,
"Ask CostSense" title, and the account dropdown showing the AWS
account tied to the IAM user you just created.

If the account dropdown says **"No AWS profiles reachable"**, the
IAM user's credentials didn't make it into the app. Double-check
that the secrets you pasted have the exact keys `AWS_ACCESS_KEY_ID`
and `AWS_SECRET_ACCESS_KEY` (case matters).

## Rollback

Just delete the app from the Streamlit Cloud dashboard. Takes ~5s.
The IAM user + access key are the only AWS resources touched:

```bash
export AWS_PROFILE=dil-team-aura

# Find and delete the access key(s) — get access-key-id from the
# creation output you saved earlier
aws iam list-access-keys --user-name costsense-streamlit-cloud
aws iam delete-access-key \
  --user-name costsense-streamlit-cloud \
  --access-key-id AKIA...

# Then delete the inline policy and the user itself
aws iam delete-user-policy \
  --user-name costsense-streamlit-cloud --policy-name costsense-readonly
aws iam delete-user --user-name costsense-streamlit-cloud
```

## Comparison to the App Runner path

| | App Runner (blocked in this org) | Streamlit Cloud |
|---|---|---|
| WebSockets | Envoy 403 on every upgrade | Native support, no config |
| Setup | ~40 min, four services (ECR, IAM, App Runner, autoscaling) | ~10 min, GitHub OAuth + secrets paste |
| Cost | ~$8/month idle + build time | Free tier, sleep-on-idle |
| Auth | AWS SSO to hackfest | Read-only IAM user access key |
| Container | Custom Docker image | Streamlit-managed Python env |
| Cold start | ~30s from `MinSize=0`; container hibernates after 10 min | Wakes in ~30s after idle |
| Rollback | 4 aws-cli commands | Click "delete" in dashboard |

For this project's constraints (SCP-blocked WebSockets, need a
public URL that renders the app), Streamlit Cloud is the correct
choice.
