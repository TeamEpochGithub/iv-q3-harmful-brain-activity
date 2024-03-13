# Using docker in this project

## Docker commands

Remove all containers

```shell
docker rm -f $(docker ps -aq)
```

Remove image

```shell
docker rmi <image_name>
```

## Step 1: Install docker

### Arch - Docker

```shell
sudo pacman -S docker docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo groupadd docker
sudo usermod -aG docker $USER
```

## Step 2: Install nvidia containers

### Arch - Nvidia containers

```shell
yay -S nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Step 2.5: Change command for sweeps, etc

```Dockerfile
CMD ["Custom command"]
```

## Step 3: Build docker container

```shell
docker build -t detect_harmful_brain_activity -f Dockerfile.wandb .
```

## Step 4: Run docker container

```shell
docker run --gpus <all | '"device=0,1"'> -v <project-path>/data:/app/data detect_harmful_brain_activity
```
