from lxros.launch import launch, node, group

def generate_launch_description():
    return launch(

   group("nodes_ns1", 
        node(
            name="talker_1",
            package="lxros",
            executable="lx_talker_cpp",
            remap={"chatter_cpp": "chatter"},
        ),
        node(name=None, package="lxros", executable="lx_listener_cpp", 
             remap={"chatter_cpp": "chatter3",}),        
        node(name=None, package="lxros", executable="lx_test_py.py",            
        ),
    )
    )