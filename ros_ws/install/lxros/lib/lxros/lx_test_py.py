#!/usr/bin/env python3
import lxros as lx
import rclpy
from std_msgs.msg import String
import time

a=''

t_cb1=time.time()

def cb1(msg: String):
    global a,t_cb1
    lx.get_context()._nodes[0].logger.info(f"py simple callback heard: {msg.data}, dt={time.time()-t_cb1:.3f}")
    a=msg.data
    t_cb1=time.time()



def cb2(msg: String):
    rclpy.logging.get_logger('whatevs').info(f"!!node1 heard: {msg.data}")
    

class MyClass():
    def __init__(self):
        self.counter = 0
        self.node=lx.init_node("node3")
        self.node.sub("chatter3", String, self.cb)
        self.t_cbc=time.time()

    def cb(self, msg: String):
        self.counter += 1
        self.node.info(f"py class method callback heard ({self.counter}): {msg.data}, dt={time.time()-self.t_cbc:.3f}")
        self.t_cbc=time.time()
    

def main():
    
    node1 = lx.init_node("node1")
    node2 = lx.init_node("node2")
    cl=MyClass()


    node1.info("node1 started in namespace %s", node1.namespace)
    node2.info("node2 started")

    node1.sub("chatter", String, cb1)
    
    node1.sub("chatter2", String, lambda msg: node1.info(f"py lambda heard: {msg.data}"))
    pub2 = node2.pub("chatter2", String)
    pub1 = node1.pub("chatter3", String)

    # publish something
    k=1.0
    while lx.ok():
        node1.info(f"publishing {a} from main()")
        pub2.publish(data=a)        
        pub1.publish(data=f"{k}")
        
        k*=0.999
        lx.spin_once(timeout=0.001)
        lx.sleep(0.1)
        

    #lx.spin()  # spins both nodes in shared MultiThreadedExecutor


if __name__ == "__main__":
    main()
