# MySQL container
## Build the image
To build a docker image run `docker build -t <name>/<tag> .`.
## Run and configuration
To run use `docker run --name <name_of_container> -e MYSQL_ROOT_PASSWORD=<password> -p 3306:3307 -d <image_name>`. 
This creates a detached container and forward port 3306 from the container into port 3307 of localhost.
Use `docker stop <container name>` to stop the container.
Use `docker exec -it <container name> bash` to enter the container`s bash.
## Connect to the server
The connection is the same as the os installation, so can connect from pycharm as usually is done. Check that in this 
case ***the port is 3307*** to make the connection ***instead of 3306.***
