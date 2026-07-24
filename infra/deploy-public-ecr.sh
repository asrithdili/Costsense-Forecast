#!/usr/bin/env bash
#
# CostSense deploy to AWS App Runner via PUBLIC ECR.
# =====================================================================
#
# Why this exists (the story):
#   The hackfest AWS account has an org SCP that DENIES:
#     * ecr:CreateRepository        (no private ECR)
#     * iam:CreateRole              (no fresh IAM roles)
#
#   The standard "App Runner pulls from private ECR via an IAM access
#   role" path is dead in this account. Public ECR needs no access role
#   at all, and `ecr-public:CreateRepository` is allowed. Same pattern
#   golden-thread-hackathon uses (see packages/golden-thread-hackathon/
#   infra/deploy-public-ecr.sh in the golden-thread repo).
#
# What this script does (idempotent create-or-update, 5 steps):
#   1a. Ensure public ECR repo `costsense` in us-east-1 (public ECR API
#       ONLY exists in us-east-1, regardless of App Runner region).
#   1b. Attach a `costsense-readonly` inline policy to the pre-existing
#       instance role. Uses iam:PutRolePolicy (allowed by the SCP) on an
#       existing role — never iam:CreateRole.
#   2.  Build a linux/amd64 image locally (App Runner requires amd64).
#   3.  Login + push to public ECR (public.ecr.aws/<alias>/costsense:<tag>).
#   4.  Create or update the App Runner service in us-west-2.
#   5.  Wait for the service to reach RUNNING, print the public URL.
#
# Two critical gotchas — both learned from golden-thread the hard way:
#
#   1. ALWAYS pass a NEW tag. Reusing a tag means the image URI in the
#      service config is unchanged, so update-service is a no-op and
#      App Runner keeps serving the OLD image. This script REQUIRES a
#      tag argument for exactly that reason — no default. CI uses
#      `sha-${GITHUB_SHA::12}` to guarantee uniqueness.
#
#   2. AWS_REGION must be us-west-2. If your shell has AWS_REGION set to
#      anything else (e.g. us-west-1 from another project), the App
#      Runner create/update calls will target the wrong region and the
#      service will be created there instead. This script defaults to
#      us-west-2 but does not force it — set it explicitly:
#
#          AWS_REGION=us-west-2 ./infra/deploy-public-ecr.sh <new-tag>
#
# Usage:
#   AWS_REGION=us-west-2 ./infra/deploy-public-ecr.sh sha-abc1234
#   AWS_REGION=us-west-2 ./infra/deploy-public-ecr.sh 2026-07-25-01
#
# Environment (all optional except the tag arg):
#   AWS_REGION           App Runner region.       default: us-west-2
#   AWS_PROFILE          named SSO profile.       default: (from env)
#   SERVICE_NAME         App Runner service.      default: costsense
#   REPO_NAME            Public ECR repo.         default: costsense
#   INSTANCE_ROLE_NAME   Pre-existing role.       default: golden-thread-hackathon-instance-role
#   PORT                 Container port.          default: 8501
#   HEALTH_PATH          HTTP health path.        default: /_stcore/health
#   CPU                  vCPU (millicores).       default: 1024   (1 vCPU)
#   MEMORY               memory (MB).             default: 2048   (2 GB)
#   BUILD_CONTEXT        docker build ctx.        default: .
#   DOCKERFILE           dockerfile path.         default: Dockerfile
#
# Requirements:
#   * aws CLI v2, docker (with buildx enabled), jq
#   * hackfest SSO session active (aws sso login --profile dil-team-hackfest)
#   * The instance role must ALREADY exist in the account (see
#     `infra/README.md` for the discovery command)

set -euo pipefail

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
AWS_REGION="${AWS_REGION:-us-west-2}"
AWS_PUBLIC_ECR_REGION="us-east-1"     # public ECR API is always us-east-1
SERVICE_NAME="${SERVICE_NAME:-costsense}"
REPO_NAME="${REPO_NAME:-costsense}"
INSTANCE_ROLE_NAME="${INSTANCE_ROLE_NAME:-golden-thread-hackathon-instance-role}"
PORT="${PORT:-8501}"
HEALTH_PATH="${HEALTH_PATH:-/_stcore/health}"
CPU="${CPU:-1024}"
MEMORY="${MEMORY:-2048}"
BUILD_CONTEXT="${BUILD_CONTEXT:-.}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
POLICY_FILE="$(cd "$(dirname "$0")" && pwd)/instance-role-policy.json"

if [ "$#" -lt 1 ]; then
  echo "ERROR: image tag argument is required."                >&2
  echo "Usage: AWS_REGION=us-west-2 $0 <new-tag>"              >&2
  echo "       tag MUST be unique per deploy or App Runner"    >&2
  echo "       will not re-pull. CI uses sha-<gitsha>."        >&2
  exit 2
fi
IMAGE_TAG="$1"

# ---------------------------------------------------------------------
# Sanity: session valid, on the intended account, jq/docker present
# ---------------------------------------------------------------------
for tool in aws docker jq; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: '$tool' not found in PATH" >&2
    exit 2
  fi
done

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "ERROR: no valid AWS session. Run 'aws sso login' first." >&2
  exit 2
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
CALLER_ARN="$(aws sts get-caller-identity --query Arn --output text)"
echo "[deploy] Account: $ACCOUNT_ID  ·  Caller: $CALLER_ARN"
echo "[deploy] Region:  $AWS_REGION (App Runner) / $AWS_PUBLIC_ECR_REGION (public ECR)"
echo "[deploy] Image:   public.ecr.aws/<alias>/$REPO_NAME:$IMAGE_TAG"
echo "[deploy] Role:    $INSTANCE_ROLE_NAME"
echo

# ---------------------------------------------------------------------
# Verify the pre-existing instance role exists AND trusts App Runner.
# We refuse to proceed if either check fails — creating a role is
# blocked by the SCP, so a missing role is a HARD stop that a human
# with different perms must resolve.
# ---------------------------------------------------------------------
echo "[step 0] Verify instance role $INSTANCE_ROLE_NAME exists and trusts tasks.apprunner.amazonaws.com"
ROLE_ARN="$(aws iam get-role --role-name "$INSTANCE_ROLE_NAME" \
  --query 'Role.Arn' --output text 2>/dev/null || true)"
if [ -z "$ROLE_ARN" ] || [ "$ROLE_ARN" = "None" ]; then
  echo "ERROR: instance role '$INSTANCE_ROLE_NAME' does not exist in account $ACCOUNT_ID." >&2
  echo "       The account SCP blocks iam:CreateRole — a role MUST be pre-created." >&2
  echo "       Ask the account owner to create a role trusting" >&2
  echo "       tasks.apprunner.amazonaws.com. See infra/README.md." >&2
  exit 2
fi
TRUSTS_APPRUNNER="$(aws iam get-role --role-name "$INSTANCE_ROLE_NAME" \
  --query "Role.AssumeRolePolicyDocument.Statement[?Principal.Service=='tasks.apprunner.amazonaws.com'] | length(@)" \
  --output text)"
if [ "$TRUSTS_APPRUNNER" -eq 0 ]; then
  echo "ERROR: instance role $INSTANCE_ROLE_NAME does not trust tasks.apprunner.amazonaws.com." >&2
  echo "       App Runner will reject the InstanceRoleArn." >&2
  exit 2
fi
echo "[step 0] OK: $ROLE_ARN"
echo

# ---------------------------------------------------------------------
# Step 1a: ensure public ECR repo exists (us-east-1 only)
# ---------------------------------------------------------------------
echo "[step 1a] Ensure public ECR repo '$REPO_NAME' in $AWS_PUBLIC_ECR_REGION"
if aws ecr-public describe-repositories \
      --region "$AWS_PUBLIC_ECR_REGION" \
      --repository-names "$REPO_NAME" >/dev/null 2>&1; then
  echo "[step 1a] repo already exists — reusing"
else
  aws ecr-public create-repository \
    --region "$AWS_PUBLIC_ECR_REGION" \
    --repository-name "$REPO_NAME" \
    --catalog-data '{"description":"CostSense — AI-native FinOps Streamlit app.","architectures":["x86-64"],"operatingSystems":["Linux"]}' \
    >/dev/null
  echo "[step 1a] created"
fi

REGISTRY_URI="$(aws ecr-public describe-registries \
  --region "$AWS_PUBLIC_ECR_REGION" \
  --query 'registries[0].registryUri' --output text)"
IMAGE_URI="$REGISTRY_URI/$REPO_NAME:$IMAGE_TAG"
echo "[step 1a] Registry: $REGISTRY_URI"
echo "[step 1a] Image URI: $IMAGE_URI"
echo

# ---------------------------------------------------------------------
# Step 1b: attach costsense-readonly inline policy to the existing role.
# We use PutRolePolicy (allowed by SCP). If golden-thread also uses
# this role (it does), that is fine — inline policies stack; we only
# manage the 'costsense-readonly' policy name and never touch others.
# ---------------------------------------------------------------------
echo "[step 1b] Attach inline policy 'costsense-readonly' to $INSTANCE_ROLE_NAME"
if [ ! -f "$POLICY_FILE" ]; then
  echo "ERROR: policy file not found: $POLICY_FILE" >&2
  exit 2
fi
aws iam put-role-policy \
  --role-name "$INSTANCE_ROLE_NAME" \
  --policy-name costsense-readonly \
  --policy-document "file://$POLICY_FILE"
echo "[step 1b] OK"
echo

# ---------------------------------------------------------------------
# Step 2: build linux/amd64 image (App Runner requires amd64)
# ---------------------------------------------------------------------
echo "[step 2] Build linux/amd64 image (this can take a few minutes)"
LOCAL_TAG="$REPO_NAME:$IMAGE_TAG"
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t "$LOCAL_TAG" \
  -f "$DOCKERFILE" \
  "$BUILD_CONTEXT"
docker tag "$LOCAL_TAG" "$IMAGE_URI"
echo "[step 2] Built $LOCAL_TAG and tagged as $IMAGE_URI"
echo

# ---------------------------------------------------------------------
# Step 3: push to public ECR
# ---------------------------------------------------------------------
echo "[step 3] Login + push to public ECR"
aws ecr-public get-login-password --region "$AWS_PUBLIC_ECR_REGION" \
  | docker login --username AWS --password-stdin public.ecr.aws
docker push "$IMAGE_URI"
echo "[step 3] Pushed $IMAGE_URI"
echo

# ---------------------------------------------------------------------
# Step 4: create or update App Runner service
# ---------------------------------------------------------------------
echo "[step 4] Create or update App Runner service '$SERVICE_NAME' in $AWS_REGION"
SERVICE_ARN="$(aws apprunner list-services \
  --region "$AWS_REGION" \
  --query "ServiceSummaryList[?ServiceName=='$SERVICE_NAME'].ServiceArn | [0]" \
  --output text 2>/dev/null || true)"

SOURCE_CONFIG=$(jq -n \
  --arg image_uri "$IMAGE_URI" \
  --arg port "$PORT" \
  '{
    ImageRepository: {
      ImageIdentifier: $image_uri,
      ImageRepositoryType: "ECR_PUBLIC",
      ImageConfiguration: {
        Port: $port,
        RuntimeEnvironmentVariables: {
          COSTSENSE_AWS_REGION: "us-west-2",
          STREAMLIT_SERVER_HEADLESS: "true",
          STREAMLIT_BROWSER_GATHER_USAGE_STATS: "false"
        }
      }
    },
    AutoDeploymentsEnabled: false
  }')

INSTANCE_CONFIG=$(jq -n \
  --arg role_arn "$ROLE_ARN" \
  --arg cpu "$CPU" \
  --arg memory "$MEMORY" \
  '{
    Cpu: $cpu,
    Memory: $memory,
    InstanceRoleArn: $role_arn
  }')

HEALTH_CONFIG=$(jq -n \
  --arg path "$HEALTH_PATH" \
  '{
    Protocol: "HTTP",
    Path: $path,
    Interval: 20,
    Timeout: 5,
    HealthyThreshold: 1,
    UnhealthyThreshold: 5
  }')

if [ -n "$SERVICE_ARN" ] && [ "$SERVICE_ARN" != "None" ]; then
  echo "[step 4] existing service found — updating"
  aws apprunner update-service \
    --region "$AWS_REGION" \
    --service-arn "$SERVICE_ARN" \
    --source-configuration "$SOURCE_CONFIG" \
    --instance-configuration "$INSTANCE_CONFIG" \
    --health-check-configuration "$HEALTH_CONFIG" \
    >/dev/null
  # update-service usually triggers a deployment on its own, but we
  # force start-deployment for safety when the image URI didn't change
  # (which happens if you reuse a tag — DON'T reuse tags, but belt and
  # suspenders).
  aws apprunner start-deployment \
    --region "$AWS_REGION" \
    --service-arn "$SERVICE_ARN" \
    >/dev/null
  echo "[step 4] update requested"
else
  echo "[step 4] no existing service — creating"
  SERVICE_ARN="$(aws apprunner create-service \
    --region "$AWS_REGION" \
    --service-name "$SERVICE_NAME" \
    --source-configuration "$SOURCE_CONFIG" \
    --instance-configuration "$INSTANCE_CONFIG" \
    --health-check-configuration "$HEALTH_CONFIG" \
    --query 'Service.ServiceArn' --output text)"
  echo "[step 4] created: $SERVICE_ARN"
fi
echo

# ---------------------------------------------------------------------
# Step 5: wait for RUNNING, print URL
# ---------------------------------------------------------------------
echo "[step 5] Wait for service to reach RUNNING"
for _ in $(seq 1 60); do
  STATUS="$(aws apprunner describe-service \
    --region "$AWS_REGION" \
    --service-arn "$SERVICE_ARN" \
    --query 'Service.Status' --output text)"
  echo "  status: $STATUS"
  case "$STATUS" in
    RUNNING) break ;;
    CREATE_FAILED|DELETE_FAILED|PAUSED)
      echo "ERROR: service in $STATUS — inspect the AWS console for the failure reason." >&2
      exit 1
      ;;
  esac
  sleep 15
done

SERVICE_URL="$(aws apprunner describe-service \
  --region "$AWS_REGION" \
  --service-arn "$SERVICE_ARN" \
  --query 'Service.ServiceUrl' --output text)"

echo
echo "===================================================================="
echo "  CostSense deployed"
echo "===================================================================="
echo "  Service:  $SERVICE_NAME"
echo "  ARN:      $SERVICE_ARN"
echo "  URL:      https://$SERVICE_URL"
echo "  Tag:      $IMAGE_TAG"
echo "===================================================================="
