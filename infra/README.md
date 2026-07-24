# CostSense — App Runner deploy

This directory contains the ONLY deploy path for CostSense. If you're
looking at `deploy/cloudformation/` — that's the old ECS/CloudFormation
path, which is blocked by an org SCP in the hackfest account and is
scheduled for removal. Use this instead.

## The story: why we use public ECR

The hackfest AWS account (`609400232087` / `dil-team-hackfest`) has an
organisation Service Control Policy that DENIES:

- `ecr:CreateRepository` — no private ECR
- `iam:CreateRole` — no fresh IAM roles

Standard App Runner deploys pull from a private ECR repo and need an
`AppRunnerECRAccessRole` IAM role to do so. Both are blocked here, so
that path is dead on arrival.

**The workaround:** public ECR. A public ECR repo needs no IAM access
role, and `ecr-public:CreateRepository` is allowed. That is the entire
reason `deploy-public-ecr.sh` exists — it is the same pattern
`golden-thread-hackathon` uses.

## What's here

```
infra/
├── README.md                   ← this file
├── deploy-public-ecr.sh        ← the deploy script (5 steps, idempotent)
└── instance-role-policy.json   ← IAM policy attached to the instance role
```

Related file outside this directory: `.github/workflows/deploy-apprunner.yml`
delegates to the same script — same source of truth, no CI-side drift.

## Prerequisites (one-time setup, done for hackfest already)

- Public ECR alias exists (auto-created on first `create-repository` call)
- **Instance role** exists in the target account with a trust policy
  allowing `tasks.apprunner.amazonaws.com` to assume it. Currently
  reused from golden-thread:
  ```
  arn:aws:iam::609400232087:role/golden-thread-hackathon-instance-role
  ```

If the instance role is ever missing (someone deleted it, or you're
deploying into a different account), the deploy script hard-stops. To
find the correct role name in any account, run:

```bash
aws iam list-roles --profile <profile> \
  --query "Roles[?AssumeRolePolicyDocument.Statement[?Principal.Service=='tasks.apprunner.amazonaws.com']].RoleName" \
  --output text
```

If that returns nothing, someone with `iam:CreateRole` (a different
account, or a temporary admin session) needs to create one. Minimum
trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "tasks.apprunner.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
```

No initial inline policies needed — `deploy-public-ecr.sh` attaches
`costsense-readonly` via `iam:PutRolePolicy` (allowed by the SCP) on
every run.

## Current live deployment

| | |
|---|---|
| URL | https://pma8dqvi4m.us-west-2.awsapprunner.com |
| Account | 609400232087 (hackfest) |
| Region | us-west-2 |
| Service ARN | `arn:aws:apprunner:us-west-2:609400232087:service/costsense/0b806d58770943eab615538073a624c2` |
| Instance role | `golden-thread-hackathon-instance-role` |

## Deploying manually (Linux / macOS)

```bash
# Log in first (SSO)
aws sso login --profile dil-team-hackfest
export AWS_PROFILE=dil-team-hackfest

# Deploy — tag is REQUIRED and MUST be unique per deploy
AWS_REGION=us-west-2 ./infra/deploy-public-ecr.sh sha-abc1234

# Or a date-based tag if you don't want to grab a git sha
AWS_REGION=us-west-2 ./infra/deploy-public-ecr.sh 2026-07-25-01
```

Successful run ends with a `https://<hash>.us-west-2.awsapprunner.com`
URL printed. Wait ~2 minutes after the script prints RUNNING for the
first HTTP request to succeed — App Runner takes a moment past
"Running" to be reachable externally.

## Deploying manually from Windows with Rancher Desktop

On Diligent-managed Windows machines the default `docker_engine` named
pipe is held by Docker Desktop, whose daemon is blocked by an org SSO
policy that we can't dismiss from a script. Rancher Desktop provides a
working Docker-compatible daemon that lives inside a WSL2 distro. The
deploy script can't run end-to-end as one shell command because
`docker` (inside Rancher's WSL) and `aws` (on Windows, holding the
SSO session) live in different environments — but the same 5 steps
work if you drive them explicitly.

### One-time setup

1. **Install Rancher Desktop** and launch it. In **Preferences →
   Container Engine** confirm the runtime is `dockerd (moby)`.
2. **Quit Docker Desktop** if it's running (right-click whale icon in
   the system tray → Quit Docker Desktop). This isn't strictly required
   — Rancher works via WSL regardless — but leaving Docker Desktop
   running only wastes memory.
3. Verify Rancher's WSL distro is reachable:
   ```bash
   wsl -d rancher-desktop -- docker info --format 'os={{.OperatingSystem}}'
   # expect: os=Rancher Desktop WSL Distribution   (or similar)
   ```

### Two Rancher WSL fixups that keep coming up

Both are done inside Rancher's WSL distro, non-persistent — apply
per-shell before running docker.

**A. Credential helper isn't installed.** Rancher's default config
sets `credsStore=secretservice` but that helper isn't shipped. Point
to a per-invocation config with credsStore="none":

```bash
wsl -d rancher-desktop -- sh -c '
  mkdir -p /tmp/dcfg
  echo "{\"credsStore\":\"none\"}" > /tmp/dcfg/config.json
'
# then run every docker call with DOCKER_CONFIG=/tmp/dcfg
```

**B. Rancher's internal DNS proxy is often dead.** `/etc/resolv.conf`
inside the distro points at `192.168.127.1` which frequently times
out. Override to a public resolver:

```bash
wsl -d rancher-desktop -- sh -c '
  echo "nameserver 8.8.8.8" > /etc/resolv.conf
  echo "nameserver 1.1.1.1" >> /etc/resolv.conf
'
```

### The five steps (Windows + Rancher path)

```bash
# All aws commands run on Windows (SSO session lives there)
export AWS_PROFILE=dil-team-hackfest
export AWS_REGION=us-west-2
IMAGE_TAG="$(date +%Y-%m-%d)-01"   # or any unique string

# --- Step 0: verify instance role ---------------------------------
ROLE_ARN=$(aws iam get-role --role-name golden-thread-hackathon-instance-role \
  --query 'Role.Arn' --output text)

# --- Step 1a: ensure public ECR repo in us-east-1 -----------------
aws ecr-public describe-repositories --region us-east-1 --repository-names costsense \
  >/dev/null 2>&1 \
  || aws ecr-public create-repository --region us-east-1 --repository-name costsense >/dev/null
REGISTRY_URI=$(aws ecr-public describe-registries --region us-east-1 \
  --query 'registries[0].registryUri' --output text)
IMAGE_URI="$REGISTRY_URI/costsense:$IMAGE_TAG"

# --- Step 1b: attach IAM policy to the pre-existing instance role -
aws iam put-role-policy --role-name golden-thread-hackathon-instance-role \
  --policy-name costsense-readonly \
  --policy-document file://infra/instance-role-policy.json

# --- Step 2: docker build inside Rancher's WSL --------------------
MSYS_NO_PATHCONV=1 wsl -d rancher-desktop -- sh -c "
  mkdir -p /tmp/dcfg
  echo '{\"credsStore\":\"none\"}' > /tmp/dcfg/config.json
  echo 'nameserver 8.8.8.8' > /etc/resolv.conf
  cd /mnt/c/Users/<you>/Documents/hackathon/costsense-forecast &&
  DOCKER_CONFIG=/tmp/dcfg docker build --platform linux/amd64 --load \
    -f Dockerfile -t $IMAGE_URI .
"

# --- Step 3: login + push (login password fetched on Windows,     -
#              piped into wsl docker) -----------------------------
aws ecr-public get-login-password --region us-east-1 \
  | MSYS_NO_PATHCONV=1 wsl -d rancher-desktop -- sh -c '
      DOCKER_CONFIG=/tmp/dcfg docker login --username AWS \
        --password-stdin public.ecr.aws
    '
MSYS_NO_PATHCONV=1 wsl -d rancher-desktop -- sh -c "
  DOCKER_CONFIG=/tmp/dcfg docker push $IMAGE_URI
"

# --- Step 4: create OR update App Runner service ------------------
SERVICE_ARN=$(aws apprunner list-services --region us-west-2 \
  --query \"ServiceSummaryList[?ServiceName=='costsense'].ServiceArn | [0]\" \
  --output text)

if [ -z "$SERVICE_ARN" ] || [ "$SERVICE_ARN" = "None" ]; then
  # first-time create — see infra/deploy-public-ecr.sh step 4 for
  # the full source/instance/health config JSON
  ./infra/deploy-public-ecr.sh "$IMAGE_TAG"   # or run create-service inline
else
  # subsequent update
  aws apprunner update-service --region us-west-2 --service-arn "$SERVICE_ARN" \
    --source-configuration "{ ...ImageIdentifier=$IMAGE_URI... }"
  aws apprunner start-deployment --region us-west-2 --service-arn "$SERVICE_ARN"
fi

# --- Step 5: wait for RUNNING -------------------------------------
aws apprunner describe-service --region us-west-2 --service-arn "$SERVICE_ARN" \
  --query 'Service.[Status,ServiceUrl]' --output text
```

In practice the script `infra/deploy-public-ecr.sh` handles all of
this in one command on Linux/macOS. Windows-with-Rancher requires
the explicit split above because `docker` and `aws` are on different
sides. The GitHub Actions workflow in
`.github/workflows/deploy-apprunner.yml` also runs the script as a
single command — it uses ubuntu-latest where docker and aws are
co-located.

## Two gotchas that will waste hours

### 1. Always pass a NEW tag

Reusing a tag means the image URI in the service config is unchanged,
so `update-service` is a no-op — App Runner keeps serving the OLD
image. The script REQUIRES a tag argument for exactly this reason. CI
uses `sha-${GITHUB_SHA::12}`, guaranteeing uniqueness per commit.

If you're deploying repeatedly from a dirty working tree, just append
a counter: `-01`, `-02`, etc.

### 2. `AWS_REGION` must be us-west-2

If your shell has `AWS_REGION` exported to anything else (`us-east-1`,
`us-west-1` from a different project), the App Runner create/update
calls will target the wrong region. Always set it explicitly:

```bash
AWS_REGION=us-west-2 ./infra/deploy-public-ecr.sh <tag>
```

Public ECR is unaffected — it always uses `us-east-1` because that's
the only region public ECR exists in, regardless of `AWS_REGION`.

## What the script does (5 steps, idempotent)

| # | Step | Notes |
|---|---|---|
| 0 | Verify the instance role exists AND trusts App Runner | Hard stops if missing — SCP blocks creating one |
| 1a | Ensure public ECR repo `costsense` in us-east-1 | Create-or-reuse |
| 1b | Attach `costsense-readonly` inline policy to the instance role | Uses `iam:PutRolePolicy`, not `CreateRole` |
| 2 | Build linux/amd64 image locally (docker buildx) | App Runner requires amd64 |
| 3 | Login + push to public ECR | `public.ecr.aws/<alias>/costsense:<tag>` |
| 4 | Create or update the App Runner service in us-west-2 | Idempotent — same script for first deploy and updates |
| 5 | Poll `describe-service` until RUNNING, print URL | Up to 15 min timeout |

## Rollback and teardown

### Roll back to a previous image tag

App Runner keeps the last several image tags in the service config
history. To roll back, redeploy the previous tag:

```bash
export AWS_PROFILE=dil-team-hackfest

# Find previous tags
aws ecr-public describe-images \
  --region us-east-1 \
  --repository-name costsense \
  --query "imageDetails[].imageTags[]" --output json

# Deploy the one you want
AWS_REGION=us-west-2 ./infra/deploy-public-ecr.sh <old-tag>
```

Because the script always ends with `start-deployment`, rolling forward
to an existing tag DOES re-pull the image (even though App Runner
normally wouldn't). That's belt-and-suspenders for exactly this case.

### Full teardown (destructive — irreversible)

Removes the App Runner service, the public ECR repo, and the inline
IAM policy this deploy attached. Leaves the reused
`golden-thread-hackathon-instance-role` intact along with its
golden-thread policies (`bedrock-invoke`, `golden-thread-datalake-secret`,
`golden-thread-selections-ddb`).

```bash
export AWS_PROFILE=dil-team-hackfest

# 1. Delete the App Runner service (async, takes ~5 min)
aws apprunner delete-service --region us-west-2 \
  --service-arn arn:aws:apprunner:us-west-2:609400232087:service/costsense/0b806d58770943eab615538073a624c2

# 2. Delete the public ECR repo and every image tag in it (irreversible)
aws ecr-public delete-repository --region us-east-1 \
  --repository-name costsense --force

# 3. Detach the costsense inline policy from the reused instance role.
#    IMPORTANT: this leaves golden-thread's own policies untouched.
aws iam delete-role-policy \
  --role-name golden-thread-hackathon-instance-role \
  --policy-name costsense-readonly
```

Verify nothing costsense-shaped remains:

```bash
aws apprunner list-services --region us-west-2 \
  --query "ServiceSummaryList[?ServiceName=='costsense']"
# expect: []

aws ecr-public describe-repositories --region us-east-1 --repository-names costsense
# expect: RepositoryNotFoundException

aws iam list-role-policies --role-name golden-thread-hackathon-instance-role \
  --query 'PolicyNames'
# expect: ["bedrock-invoke","golden-thread-datalake-secret","golden-thread-selections-ddb"]
# (i.e. NO "costsense-readonly")
```

## CI

`.github/workflows/deploy-apprunner.yml` runs the same script on:

- Push to `main` touching `src/**`, `requirements.txt`, `Dockerfile`,
  `infra/**`, or the workflow file itself
- Manual `workflow_dispatch` (with an optional tag override)

Auth via OIDC; the workflow assumes the `github-actions-costsense` role
in the hackfest account. Concurrency group `costsense-apprunner`
serialises deploys so two branches can't race the single shared
service.

## Removing the old ECS/CloudFormation path

After this App Runner path is validated end-to-end, the following are
scheduled for removal:

- `deploy/cloudformation/costsense-ecs.yaml`
- `deploy/cloudformation/deploy.sh`
- `deploy/cloudformation/parameters.example.json`
- `deploy/ecs/task-definition.json`
- `.github/workflows/deploy-ecs.yml`

Do NOT delete them until the App Runner service has served a real HTTP
response.
