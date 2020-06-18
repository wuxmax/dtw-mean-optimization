#!/bin/bash

source .env

if [[ "$ALWAYS_BUILD" == 1 ]]; then
    scripts/build_image.sh
fi

docker run -d --name=$USER-$(uuidgen) --user $(id -u):$(id -u) -it --gpus all -v $(pwd)/datasets:/datasets -v $(pwd)/results:/results $IMAGE_NAME remote
# docker run -d --name=$USER-$(uuidgen) -it --gpus all -v $(pwd)/datasets:/datasets -v $(pwd)/results:/results $IMAGE_NAME remote