#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Imu, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class ControleRobo(Node):

    def __init__(self):
        super().__init__('controle_robo')

        # Publisher para comando de velocidade
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Image, '/robot_cam', self.camera_callback, 10)

        # Timer para enviar comandos continuamente
        self.timer = self.create_timer(0.1, self.move_robot)

        # Estado interno
        self.obstaculo_a_frente = False

    def scan_callback(self, msg: LaserScan):
        # Verifica uma faixa estreita ao redor de 0° (frente)
        num_ranges = len(msg.ranges)
        if num_ranges == 0:
            return

        # Índices de -30° a +30° (equivalente a 330 até 30)
        indices_frente = list(range(330, 360)) + list(range(0, 31))

        # Filtra distancias
        distancias = [msg.ranges[i] for i in indices_frente]

        if distancias and min(distancias) < 0.5:
            self.obstaculo_a_frente = True
            self.get_logger().info('Obstáculo detectado a {:.2f}m à frente'.format(min(distancias)))
        else:
            self.obstaculo_a_frente = False

    def imu_callback(self, msg: Imu):
        # Mensagens da IMU!
        pass

    def odom_callback(self, msg: Odometry):
        # Mensagens de Odometria das rodas!
        pass

    def camera_callback(self, msg: Image):
        # Mensagens com imagens da câmera!
        pass

    def move_robot(self):
        twist = Twist()
        if not self.obstaculo_a_frente:
            twist.linear.x = 0.1  # Move para frente
        else:
            twist.angular.z = -0.3  # Gira em torno do proprio eixo

        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ControleRobo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
