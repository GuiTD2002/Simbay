#!/usr/bin/env python3

import lxros as lx
from example_interfaces.action import Fibonacci
import time


def main():
    node = lx.init_node("fib_cancel")

    client = node.action_client("fibonacci", Fibonacci)

    # Send a goal    
    goal = Fibonacci.Goal()
    goal.order = 10
    print("Sending Fibonacci goal...")
    goal_handle, result_future = client.send_async(goal)

    time.sleep(1.0)
    print("Sending Fibonacci goal2...")
    goal_handle, result_future = client.send_async({"order": 10})

    # Wait a bit
    time.sleep(5.0)

    # Cancel goal
    print("Cancelling goal...")
    cancel_future = client.cancel(goal_handle)

    # Spin until cancellation is processed
    ctx = node.context
    while lx.ok() and not cancel_future.done():
        lx.spin_once(0.1)

    cancel_res = cancel_future.result()
    print("Cancel response:", cancel_res)

    # Try to see if there is a partial result
    # (server may or may not provide one)
    if result_future.done():
        print("Result received despite cancel:", result_future.result())
    else:
        print("Action did not complete.")


if __name__ == "__main__":
    main()
