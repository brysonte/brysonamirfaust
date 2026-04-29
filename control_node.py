import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.subscription = self.create_subscription(Float64, 'object_detections', self.callback, 10)

    def callback(self, msg):
        self.get_logger().info(f'RECEIVING CONTROL COMMAND. Object Score: {msg.data:.4f}')

def main():
    rclpy.init()
    rclpy.spin(ControlNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()