#!/usr/bin/env bash

# Helper script to start a dev server of the project inside docker

PROJECT_DIR=$(pwd)
APP_DIR=app

docker run -it --rm -p 5000:5000 \
    -v "$PROJECT_DIR":"/$APP_DIR" \
    gautamborg/zappa_developer_pytorch:latest
