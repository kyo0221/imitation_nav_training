import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool

class Buttons:
    PS = 10
    Share = 8
    Circle = 1
    Options = 9

class Axes:
    L_Y = 1
    R_X = 3

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        self.declare_parameter('linear_max_vel', 0.5)
        self.declare_parameter('angular_max_vel', 1.57)
        self.declare_parameter('autonomous_flag', False)

        self.linear_max_vel = self.get_parameter('linear_max_vel').value
        self.angular_max_vel = self.get_parameter('angular_max_vel').value
        self.is_autonomous = self.get_parameter('autonomous_flag').value
        self.is_save = False

        self.prev_buttons = [0] * 12  # ボタン数を仮定して初期化

        qos = 10
        self.publisher_vel = self.create_publisher(Twist, 'cmd_vel', qos)
        self.publisher_stop = self.create_publisher(Empty, 'stop', qos)
        self.publisher_restart = self.create_publisher(Empty, 'restart', qos)
        self.publisher_save = self.create_publisher(Bool, 'save', qos)
        self.publisher_autonomous = self.create_publisher(Bool, 'autonomous', qos)
        self.publisher_nav_start = self.create_publisher(Empty, 'nav_start', qos)

        self.subscription_joy = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            qos
        )

        self.publisher_restart.publish(Empty())

    def joy_callback(self, msg: Joy):
        buttons = msg.buttons
        axes = msg.axes

        def is_pressed(idx):
            return buttons[idx] == 1 and self.prev_buttons[idx] == 0

        if is_pressed(Buttons.Options):
            self.is_save = not self.is_save
            msg_save = Bool()
            msg_save.data = self.is_save
            self.publisher_save.publish(msg_save)
            self.get_logger().info(f'データ収集フラグ: {self.is_save}')

        # Shareボタン → 自律 / 手動 切り替え
        if is_pressed(Buttons.Share):
            self.is_autonomous = not self.is_autonomous
            msg_autonomous = Bool()
            msg_autonomous.data = self.is_autonomous
            self.publisher_autonomous.publish(msg_autonomous)
            self.get_logger().info(f'自動フラグ: {self.is_autonomous}')

        if is_pressed(Buttons.Circle):
            self.publisher_nav_start.publish(Empty())
            self.get_logger().info('自律走行開始')

        if not self.is_autonomous:
            twist = Twist()
            twist.linear.x = self.linear_max_vel * axes[Axes.L_Y]
            twist.angular.z = self.angular_max_vel * axes[Axes.R_X]
            self.publisher_vel.publish(twist)

        # ボタン状態を更新
        self.prev_buttons = list(buttons)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
