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

## Deploying manually

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

## Rollback

App Runner keeps the last several image tags in the service config
history. To roll back, redeploy the previous tag:

```bash
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
