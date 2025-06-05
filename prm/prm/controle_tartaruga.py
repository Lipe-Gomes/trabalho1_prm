import rclpy
from rclpy.node import Node

from turtlesim.msg  import Pose
from geometry_msgs.msg import Twist

from math import atan2
from math import pi

class ControleTartaruga(Node):

    def __init__(self):
        super().__init__('controle_tartaruga')

        self.pub_cmd_vel = self.create_publisher(Twist,
                                '/turtle1/cmd_vel', 10)
        
        self.sub_pose = self.create_subscription(
            Pose,
            '/turtle1/pose',
            self.nova_pose_callback,
            10)

        self.sub_pose  # prevent unused variable warning

        self.ax = 2
        self.ay = 2


    def nova_pose_callback(self, msg):

        x = msg.x
        y = msg.y
        t = msg.theta

        t_alvo = self.calcula_angulo(x,y, self.ax, self.ay)

        erro_angulo = t_alvo - t

        # Normaliza o erro do ângulo para o intervalo [-pi, pi]
        if erro_angulo > pi:
            erro_angulo -= 2 * pi
        elif erro_angulo < -pi:
            erro_angulo += 2 * pi

        msg = Twist()


        if abs( erro_angulo) > 0.1:
            msg.angular.z = erro_angulo * 2.0
            msg.linear.x = 0.0            
            self.pub_cmd_vel.publish(msg)
            self.get_logger().info('Girando para o alvo')
        else:
            msg.linear.x = 1.0
            msg.angular.z = 0.0
            self.get_logger().info('Indo para alvo')
            self.pub_cmd_vel.publish(msg)


    def calcula_angulo(self, x1, y1, x2, y2):
        # Calcula o ângulo entre dois pontos
        angulo = 0

        if x2 - x1 != 0:
            angulo = atan2(y2 - y1, x2 - x1)
        else:
            if y2 > y1:
                angulo = pi / 2
            else:
                angulo = -pi / 2
        return angulo


    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.pub_cmd_vel.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = ControleTartaruga()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
