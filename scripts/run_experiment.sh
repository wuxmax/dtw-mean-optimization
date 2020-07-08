#!/bin/bash

source .env

if [[ "$ALWAYS_BUILD" == 1 ]]; then
    scripts/build_image.sh
fi

CONFIG="default"
if [[ -n "$1" ]]; then
    CONFIG="$1"
fi

docker run -dt --name=$USER-$(uuidgen) -v $(pwd)/datasets:/datasets -v $(pwd)/results:/results $IMAGE_NAME $CONFIG -d /datasets/UCRArchive_2018 -r /results
