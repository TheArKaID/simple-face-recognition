# Face Recognition with Docker Swarm
docker swarm init
docker stack deploy -c docker-stack.yml face_recognition_swarm
docker service ls
docker ps

# To remove the stack and leave the swarm
docker stack rm face_recognition_swarm
docker swarm leave --force
