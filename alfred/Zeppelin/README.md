# Zeppelin: http://zeppelin.apache.org/
### Installation Steps

Install Docker and download zeppelin docker image. 

Make sure that docker is installed in your local machine.

Use this command to launch Apache Zeppelin in a container.

docker run -p 8080:8080 --rm --name zeppelin apache/zeppelin:0.9.0

To persist logs and notebook directories, use the volume option for docker container.

docker run -p 8080:8080 --rm -v $PWD/logs:/logs -v $PWD/notebook:/notebook -v $PWD/data:/data -e ZEPPELIN_LOG_DIR='/logs' -e ZEPPELIN_NOTEBOOK_DIR='/notebook' --name zeppelin apache/zeppelin:0.9.0
