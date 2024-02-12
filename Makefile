#################
# General setup #
#################

GIT_BRANCH=main # IF YOU WANT TO AUTOMATICALLY SET UP THE VM ON A SPECIFIC BRANCH

# The following variables are assumed to already exist as environment variables locally,
# or can be uncommented and edited below.

#GITHUB_USER_TOKEN=$(GITHUB_USER_TOKEN)
#GITHUB_ACCESS_TOKEN=$(GITHUB_ACCESS_TOKEN)
#NEPTUNE_API_TOKEN=$(NEPTUNE_API_TOKEN)

#######
# TPU #
#######
# Shared set-up.
#BASE_CMD=gcloud alpha compute tpus tpu-vm
#BASE_NAME=popcorn
#WORKER=all
#PORT=8889
#NUM_DEVICES=8

BASE_CMD=gcloud alpha compute tpus tpu-vm
BASE_NAME=poppy# popcorn
WORKER=all
PORT=8889
NUM_DEVICES=8

# Basic v3 TPU configuration.
PROJECT=research-294715#int-research-tpu-trc-2
ZONE=us-central1-a
ACCELERATOR_TYPE=v3-$(NUM_DEVICES)
NAME=$(BASE_NAME)-$(ACCELERATOR_TYPE)
RUNTIME_VERSION=v2-alpha

.PHONY: set_project
set_project:
	gcloud config set project $(PROJECT)

.PHONY: create_vm
create_vm:
	$(BASE_CMD) create $(NAME) --zone $(ZONE) \
		--project $(PROJECT) \
		--accelerator-type $(ACCELERATOR_TYPE) \
		--version $(RUNTIME_VERSION) \

.PHONY: prepare_vm
prepare_vm:
	$(BASE_CMD) ssh --zone $(ZONE) $(NAME) \
		--project $(PROJECT) \
		--worker=$(WORKER) \
		--command="git clone -b ${GIT_BRANCH} https://${GITHUB_USER_TOKEN}:${GITHUB_ACCESS_TOKEN}@github.com/instadeepai/popcorn.git"

.PHONY: create
create: create_vm prepare_vm

.PHONY: start
start:
	$(BASE_CMD) start $(NAME) --zone=$(ZONE) --project $(PROJECT)

.PHONY: connect
connect:
	$(BASE_CMD) ssh $(NAME) --zone $(ZONE) --project $(PROJECT)

#.PHONY: connect_redirect_port
#connect_redirect_port:  # IF YOU WANT TO REDIRECT NOTEBOOKS
#	$(BASE_CMD) ssh $(NAME) --zone $(ZONE) --project $(PROJECT) -- -N -f -L $(PORT):localhost:$(PORT)

.PHONY: list
list:
	$(BASE_CMD) list --zone=$(ZONE) --project $(PROJECT)

.PHONY: describe
describe:
	$(BASE_CMD) describe $(NAME) --zone=$(ZONE) --project $(PROJECT)

.PHONY: stop
stop:
	$(BASE_CMD) stop $(NAME) --zone=$(ZONE) --project $(PROJECT)

.PHONY: delete
delete:
	$(BASE_CMD) delete $(NAME) --zone $(ZONE) --project $(PROJECT)

.PHONY: run
run:
	$(BASE_CMD) ssh --zone $(ZONE) $(NAME) --project $(PROJECT) --worker=$(WORKER) --command="$(command)"

##########
# Docker #
##########

SHELL := /bin/bash

# variables
WORK_DIR = $(PWD)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

DOCKER_BUILD_ARGS = \
	--build-arg USER_ID=$(USER_ID) \
	--build-arg GROUP_ID=$(GROUP_ID) \
	--build-arg GITHUB_USER_TOKEN=$(GITHUB_USER_TOKEN) \
	--build-arg GITHUB_ACCESS_TOKEN=$(GITHUB_ACCESS_TOKEN)

DOCKER_RUN_FLAGS = --rm --privileged -p ${PORT}:${PORT} --network host
#DOCKER_RUN_FLAGS_LOCAL = --rm --privileged -p ${PORT}:${PORT}
DOCKER_VARS_TO_PASS = -e NEPTUNE_API_TOKEN=$(NEPTUNE_API_TOKEN)
DOCKER_IMAGE_NAME = popcorn
DOCKER_CONTAINER_NAME = popcorn_container


.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rfv
	find . | grep -E ".pytest_cache" | xargs rm -rfv
	find . | grep -E "nul" | xargs rm -rfv

.PHONY: docker_build_tpu
docker_build_tpu:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) -f docker/tpu.Dockerfile .

.PHONY: docker_build_local
docker_build_local:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) -f docker/local.Dockerfile .

.PHONY: docker_run
docker_run:
	sudo docker run $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) $(command)

.PHONY: docker_start
docker_start:
	sudo docker run -itd $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME)

.PHONY: docker_enter
docker_enter:
	sudo docker exec -it $(DOCKER_CONTAINER_NAME) /bin/bash

.PHONY: docker_interactive
docker_interactive: docker_start docker_enter
	echo "Running container $(DOCKER_CONTAINER_NAME) in interactive mode."

.PHONY: docker_kill
docker_kill:
	sudo docker kill $(DOCKER_CONTAINER_NAME)

.PHONY: docker_list
docker_list:
	sudo docker ps

#.PHONY: docker_notebook
#docker_notebook: docker_run_local
#	#echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L $(PORT):localhost:$(PORT)"
#	jupyter notebook --port=$(PORT) --no-browser --ip=0.0.0.0 --allow-root
#	echo "Go to http://localhost:${PORT} and enter token above."
