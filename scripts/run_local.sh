#!/bin/bash

source .env

python ./scripts/prepare_run.py

if [[ "$ALWAYS_BUILD" == 1 ]]; then
    scripts/build_image.sh
fi

docker run -it -v $(pwd)/results:/results $IMAGE_NAME