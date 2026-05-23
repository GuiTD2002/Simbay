BASE_IMAGE ?= rayproject/ray:2.55.1-py312-cu129
IMAGE_REPO ?= marianascosta/simbay-ray
IMAGE_TAG ?= demo.2.55.1
IMAGE ?= $(IMAGE_REPO):$(IMAGE_TAG)
RAY_CLUSTER_FILE ?= raycluster.yaml

.PHONY: help install-dev docker-build-image docker-push-image docker-release-image cluster-apply cluster-delete cluster-recreate cluster-status

help:
	@echo "install-dev           Install dependencies with dev tools"
	@echo "docker-release-image  Build and push $(IMAGE)"
	@echo "cluster-recreate      Delete RayCluster, build image, and apply $(RAY_CLUSTER_FILE)"

install-dev:
	poetry install --with dev

docker-build-image:
	docker build --build-arg BASE_IMAGE=$(BASE_IMAGE) -t $(IMAGE) .

docker-push-image:
	docker push $(IMAGE)

docker-release-image: docker-build-image docker-push-image

cluster-apply: docker-build-image
	kubectl apply -f $(RAY_CLUSTER_FILE)

cluster-delete:
	kubectl delete -f $(RAY_CLUSTER_FILE) --ignore-not-found=true

cluster-recreate: cluster-delete cluster-apply

cluster-status:
	kubectl get pods -o wide

