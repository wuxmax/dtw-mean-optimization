source .env

docker run -it -v $(pwd)/results:/results $IMAGE_NAME