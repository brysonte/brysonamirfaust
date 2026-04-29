import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
import jax
import jax.numpy as jnp
import time

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.subscription = self.create_subscription(Float64MultiArray, 'raw_sensor_data', self.callback, 10)
        self.publisher_ = self.create_publisher(Float64, 'object_detections', 10)
        
        # JAX Initialization
        self.key = jax.random.PRNGKey(0)
        self.weights = jax.random.normal(self.key, (200, 200))
        self.compute = jax.jit(lambda d, w: jnp.dot(d, w))
        self.get_logger().info('Perception Node Ready (JAX JIT Warm)')

    def callback(self, msg):
        start = time.perf_counter()
        self.get_logger().info('Sensor data received. Starting JAX processing...')
        
        # Convert back to JAX array and multiply
        data = jnp.array(msg.data).reshape(200, 200)
        result = self.compute(data, self.weights).block_until_ready()
        
        exec_time = (time.perf_counter() - start) * 1000
        self.get_logger().info(f'JAX Compute finished in {exec_time:.2f}ms')
        
        # Send out a simple "Detection Score" (the mean of the result)
        out_msg = Float64()
        out_msg.data = float(jnp.mean(result))
        self.publisher_.publish(out_msg)

def main():
    rclpy.init()
    rclpy.spin(PerceptionNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()