#!/usr/bin/env bash
# Helper script to deploy project on AWS Lambda using zappa

PROJECT_DIR=$(pwd)
CMD=$1
STAGE=$2
ARG=$3
APP_DIR=app

echo "$PROJECT_DIR"

docker run --rm -v "$PROJECT_DIR":/"$APP_DIR" \
      -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
      -e stage="$STAGE" -e cmd="$CMD" -e arg="$ARG" gautamborg/zappa_deployer_pytorch:latest
