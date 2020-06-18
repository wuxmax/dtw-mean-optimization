#!/bin/bash

source .env

if [[ "$ALWAYS_BUILD" == 1 ]]; then
    ./build_image.sh
fi

docker run --name=$USER-$(uuidgen) --user $(id -u):$(id -g) -it --gpus all -v $(pwd)/datasets:/datasets -v $(pwd)/results:/results $IMAGE_NAME remote