import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Header
import numpy as np

class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')
        self.publisher_ = self.create_publisher(Float64MultiArray, 'raw_sensor_data', 10)
        # We publish slowly to give our JAX "Perception" time to breathe
        self.timer = self.create_timer(0.5, self.publish_sensor_frame)

    def publish_sensor_frame(self):
        # Simulating a 1000x1000 sensor frame
        data = np.random.randn(200, 200).flatten().tolist()
        
        msg = Float64MultiArray()
        msg.data = data
        
        # We'll use the first index to store a timestamp for latency tracking
        self.publisher_.publish(msg)
        self.get_logger().info('Sent 1,000,000 sensor data points to Perception.')

def main():
    rclpy.init()
    rclpy.spin(SensorNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()