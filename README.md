# Region_detection
yolov8을 활용한 영역 관리 툴

# Docker 명령어
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v {local_vol}:/Region_detection --net=host --ipc host --name ultratyics ultralytics/ultralytics:latest /bin/bash