#!/usr/bin/env bash
set -euo pipefail

STACK_NAME="${STACK_NAME:-costsense-ecs}"
TEMPLATE="${TEMPLATE:-deploy/cloudformation/costsense-ecs.yaml}"
PARAMS_FILE="${PARAMS_FILE:-deploy/cloudformation/parameters.json}"
REGION="${AWS_REGION:-us-west-2}"

if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "Missing $PARAMS_FILE"
  echo "Copy deploy/cloudformation/parameters.example.json to parameters.json and edit it."
  exit 1
fi

PARAM_OVERRIDES=()
while IFS= read -r line; do
  PARAM_OVERRIDES+=("$line")
done < <(python3 - "$PARAMS_FILE" <<'PY'
import json
import sys

for entry in json.load(open(sys.argv[1], encoding="utf-8")):
    print(f"{entry['ParameterKey']}={entry['ParameterValue']}")
PY
)

aws cloudformation deploy \
  --stack-name "$STACK_NAME" \
  --template-file "$TEMPLATE" \
  --parameter-overrides "${PARAM_OVERRIDES[@]}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$REGION"

echo
echo "Stack outputs:"
aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query "Stacks[0].Outputs" \
  --output table
