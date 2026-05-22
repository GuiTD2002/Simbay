## Setup 

Check Python version (must be ≤ 3.12):
```
python --version
```
If higher, install Python 3.12 first.

Create venv and install dependencies:
```
python -m venv .venv
pip install poetry
poetry install
```

To add a new dependency later:
```
poetry add <dep>
```

---

## Connect to the Ray cluster

You need **two terminals on your laptop**.

**Terminal 1 (the remote)** — opens the connection (leave it open):

```
ssh user@<remote-machine-ip>
kubectl port-forward --address 0.0.0.0 svc/simbay-cluster-head-svc 10001:10001 8265:8265
```

to know the ip run this on the remote computer (it's the first one) - skip this if you already have it 
```
hostname -I
```


**Terminal 2** — runs your script:
```
export SIMBAY_RAY_IP=<remote-machine-ip>
python scripts/warp_pos_estimation_2D.py
```

`SIMBAY_RAY_IP` sets the Ray cluster IP — both `warp_pos_estimation_2D.py` and `real_pos_estimation_2D.py` read it. If unset, they fall back to the hardcoded default in the script.

---

## After changing code

Rebuild and push the image (local machine):
```
docker build --build-arg BASE_IMAGE=rayproject/ray:2.55.1-py312-cu129 -t marianascosta/simbay-ray:demo.2.55.1 .
docker push marianascosta/simbay-ray:demo.2.55.1
```

Restart the cluster (remote machine, in `/home/simbay/Documents/Simbay`):
```
kubectl delete -f raycluster.yaml --ignore-not-found=true
kubectl apply -f raycluster.yaml
```
