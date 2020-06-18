source .env

docker run --name=$USER-$(uuidgen) --user $(id -u):$(id -g) -it --gpus all -v $(pwd)/results:/results $IMAGE_NAME remote