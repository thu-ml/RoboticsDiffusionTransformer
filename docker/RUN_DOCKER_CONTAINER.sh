#!/bin/sh
xhost +

if [ ! -z $1 ]; then
  TAG_NAME=$1
else
  TAG_NAME="latest"
fi

docker-compose up ${TAG_NAME} &