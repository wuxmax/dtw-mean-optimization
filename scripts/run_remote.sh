source .env

docker run --name=$USER-$(uuidgen) --user $(id -u):$(id -g) -it --gpus all -v $(pwd)/datasets:/datasets -v $(pwd)/results:/results $IMAGE_NAME remote