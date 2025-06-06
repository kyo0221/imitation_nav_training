import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool, String

from time import time

class Buttons:
    Circle = 1
    Triangle = 2
    Square = 3
    L1 = 4
    R1 = 5
    L2 = 6
    R2 = 7
    Share = 8
    Options = 9
    PS = 10
    L_PRESS = 11

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
        self.is_autorun = False

        self.prev_buttons = [0] * 13
        self.prev_left_pressed = False
        self.prev_right_pressed = False

        self.last_pressed_time = {}
        self.debounce_interval = 0.5

        qos = 10
        self.publisher_vel = self.create_publisher(Twist, 'cmd_vel', qos)
        self.publisher_stop = self.create_publisher(Empty, 'stop', qos)
        self.publisher_restart = self.create_publisher(Empty, 'restart', qos)
        self.publisher_save = self.create_publisher(Bool, 'save', qos)
        self.publisher_autonomous = self.create_publisher(Bool, 'autonomous', qos)
        self.publisher_nav_start = self.create_publisher(Empty, 'nav_start', qos)
        self.publisher_action_command = self.create_publisher(String, 'cmd_route', qos)
        self.publisher_save_image = self.create_publisher(Empty, 'save_image', qos)

        self.subscription_joy = self.create_subscription(Joy, 'joy', self.joy_callback, qos)
        self.publisher_restart.publish(Empty())

    def joy_callback(self, msg: Joy):
        buttons = msg.buttons
        axes = msg.axes

        def is_pressed(idx):
            now = time()
            last_time = self.last_pressed_time.get(idx, 0.0)
            if buttons[idx] == 1 and self.prev_buttons[idx] == 0 and (now - last_time) > self.debounce_interval:
                self.last_pressed_time[idx] = now
                return True
            return False

        if is_pressed(Buttons.Options) or is_pressed(Buttons.Circle):
            self.is_save = not self.is_save
            msg_save = Bool()
            msg_save.data = self.is_save
            self.publisher_save.publish(msg_save)
            self.get_logger().info(f'データ収集フラグ: {self.is_save}')

        if is_pressed(Buttons.Share):
            self.is_autonomous = not self.is_autonomous
            msg_autonomous = Bool()
            msg_autonomous.data = self.is_autonomous
            self.publisher_autonomous.publish(msg_autonomous)
            self.get_logger().info(f'自動フラグ: {self.is_autonomous}')

        if is_pressed(Buttons.Triangle):
            self._publish_route_command('straight')
            self.publisher_save_image.publish(Empty())

        left_pressed = buttons[Buttons.L1] == 1 or buttons[Buttons.L2] == 1
        if left_pressed and not self.prev_left_pressed:
            self._publish_route_command('left')
            self.publisher_save_image.publish(Empty())
        elif not left_pressed and self.prev_left_pressed:
            self._publish_route_command('straight')
            self.publisher_save_image.publish(Empty())
        self.prev_left_pressed = left_pressed

        right_pressed = buttons[Buttons.R1] == 1 or buttons[Buttons.R2] == 1
        if right_pressed and not self.prev_right_pressed:
            self._publish_route_command('right')
            self.publisher_save_image.publish(Empty())
        elif not right_pressed and self.prev_right_pressed:
            self._publish_route_command('straight')
            self.publisher_save_image.publish(Empty())
        self.prev_right_pressed = right_pressed

        if is_pressed(Buttons.L_PRESS):
            self.is_autorun = not self.is_autorun
            self.get_logger().info(f'オートラン: {self.is_autorun}')

        if not self.is_autonomous:
            twist = Twist()
            if self.is_autorun:
                twist.linear.x = self.linear_max_vel
            else:
                twist.linear.x = self.linear_max_vel * axes[Axes.L_Y]
            twist.angular.z = self.angular_max_vel * axes[Axes.R_X]
            self.publisher_vel.publish(twist)

        self.prev_buttons = list(buttons)

    def _publish_route_command(self, command: str):
        msg = String()
        msg.data = command
        self.publisher_action_command.publish(msg)
        self.get_logger().info(f'ルート指令送信: {command}')


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
