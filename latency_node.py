import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import time
import jax
import jax.numpy as jnp

class WaymoSimNode(Node):
    def __init__(self):
        super().__init__('waymo_sim_node')
        self.publisher_ = self.create_publisher(Header, 'vehicle_state', 10)
        
        # 1. Initialize JAX arrays to represent our "sensor data"
        self.get_logger().info('Initializing JAX sensor matrices...')
        self.key = jax.random.PRNGKey(0)
        
        # Create 1000x1000 matrices to simulate a heavy perception payload
        self.sensor_data = jax.random.normal(self.key, (1000, 1000)) 
        self.weights = jax.random.normal(self.key, (1000, 1000))
        
        # 2. JAX uses Just-In-Time (JIT) compilation. Let's compile our "brain" function
        self.process_data_jit = jax.jit(self._compute_logic)
        
        # 3. Warm up the JIT compiler (the first run is always slow)
        self.get_logger().info('Warming up JIT compiler...')
        self.process_data_jit(self.sensor_data, self.weights).block_until_ready()
        self.get_logger().info('Brain ready. Starting control loop.')

        # Run at 10Hz
        self.timer = self.create_timer(0.1, self.process_waymo_frame)

    def _compute_logic(self, data, weights):
        # Simulated neural network pass (heavy math)
        return jnp.dot(data, weights)

    def process_waymo_frame(self):
        start_time = time.perf_counter()
        
        # --- THE ACTUAL JAX COMPUTATION ---
        # We use block_until_ready() because JAX operations are asynchronous by default
        result = self.process_data_jit(self.sensor_data, self.weights).block_until_ready()
        # ----------------------------------
        
        exec_duration_ms = (time.perf_counter() - start_time) * 1000
        
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = f"JAX Exec Time: {exec_duration_ms:.2f} ms"
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.frame_id}')

def main(args=None):
    rclpy.init(args=args)
    node = WaymoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()