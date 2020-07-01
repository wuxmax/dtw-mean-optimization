#!/bin/bash

source .env
export DOCKER_BUILDKIT=1
docker build -t $IMAGE_NAME .
