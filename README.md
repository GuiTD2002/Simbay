# Simbay


Seting up the franka emika robot

```bash
source ~/Desktop/Simbay/ros_ws/install/setup.bash
ros2 launch franka_lxv franka_lxv_bringup.launch.py

# or alternative
ros2 launch ./ros_ws/src/franka_lxv/launch/franka_lxv_bringup.launch.py
```

to expose the ray service:
```
kubectl port-forward service/simbay-cluster-head-svc 8265:8265 10002:10001
``