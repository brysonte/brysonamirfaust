import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from std_msgs.msg import Float64

class LatencyReceiverNode(Node):
    def __init__(self):
        super().__init__('latency_receiver_node')
        self.subscription = self.create_subscription(Header, 'vehicle_state', self.listener_callback, 10)
        self.latency_publisher = self.create_publisher(Float64, 'network_latency', 10)

    def listener_callback(self, msg):
        now = self.get_clock().now()
        arrival_ns = now.nanoseconds
        sent_ns = msg.stamp.sec * 1e9 + msg.stamp.nanosec
        
        latency_ms = (arrival_ns - sent_ns) / 1e6
        
        self.get_logger().info(f'Network Latency: {latency_ms:.2f} ms')
        
        lat_msg = Float64()
        lat_msg.data = latency_ms
        self.latency_publisher.publish(lat_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LatencyReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
