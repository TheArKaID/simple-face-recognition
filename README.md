# Face Recognition

Face verification for employee attendance. HRIS integration contract:
[docs/API.md](docs/API.md). Third-party model licences:
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

Run `python tools/roster_audit.py` after each enrolment round — separation
between employees narrows as the roster grows, and it names the closest pair.

## Docker Swarm

```sh
docker swarm init
docker stack deploy -c docker-stack.yml face_recognition_swarm
docker service ls
docker ps
```

To remove the stack and leave the swarm:

```sh
docker stack rm face_recognition_swarm
docker swarm leave --force
```
