#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Imu, Image
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, Pose

from std_msgs.msg import Header

from scipy.spatial.transform import Rotation as R
from queue import PriorityQueue

from cv_bridge import CvBridge
import cv2
import numpy as np

# Necessario para publicar o frame map:
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

# ============================================
# 2. Representação do Mapa e Planejamento Simples
# ============================================
# O ambiente é representado por uma matriz 2D (grid_map), onde:
# -1 = célula desconhecida
#  0 = célula livre (não implementado ainda)
# 100 = célula ocupada (obstáculo ou posição do robô)
# Essa estrutura permite planejamento simples por gradiente, A* ou direção heurística.
# ============================================

# Classe principal do nó de mapeamento do robô
class RoboMapper(Node):
    # Construtor da classe
    def __init__(self):
        super().__init__('robo_mapper')

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Pose, '/model/prm_robot/pose', self.odom_callback, 10)
        self.create_subscription(Image, '/robot_cam/colored_map', self.camera_callback, 10)

        # Utilizado para converter imagens ROS -> OpenCV
        self.bridge = CvBridge()

        # Timer para enviar comandos continuamente
        self.timer = self.create_timer(0.5, self.atualiza_mapa)

        # Estado atual do robo:
        self.x = 0
        self.y = 0
        self.heading = 0

        # Atributos de configuração do mapa
        self.grid_size = 50
        self.resolution = 0.25
        self.grid_map = -np.ones((self.grid_size, self.grid_size), dtype=np.int8)

        # Publisher do mapa
        self.map_pub = self.create_publisher(OccupancyGrid, '/grid_map', 10)

        # Publicando o frame map para vizualização no RVis
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = "map"
        static_tf.child_frame_id = "odom_gt"
        static_tf.transform.translation.x = 0.0
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.0
        static_tf.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(static_tf)
        
    # Callback do sensor
    def scan_callback(self, msg: LaserScan):
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x_rel = r * np.cos(angle)
                y_rel = r * np.sin(angle)
                x_abs = self.x + x_rel * np.cos(self.heading) - y_rel * np.sin(self.heading)
                y_abs = self.y + x_rel * np.sin(self.heading) + y_rel * np.cos(self.heading)
                gx, gy = self.world_to_grid(x_abs, y_abs)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    self.grid_map[gy, gx] = 100
            angle += msg.angle_increment
            
    # Callback da odometria
    def odom_callback(self, msg: Pose):
        self.x = msg.position.x
        self.y = msg.position.y
        orientation_q = msg.orientation
        quat = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        r = R.from_quat(quat)
        euler = r.as_euler('xyz', degrees=False)
        self.heading = euler[2]
        
    # Callback da câmera
    def camera_callback(self, msg: Image):
        pass
        
    # Converte coordenadas do mundo para coordenadas do grid
    def world_to_grid(self, x, y):
        origin_offset = self.grid_size * self.resolution / 2
        gx = int((x + origin_offset) / self.resolution)
        gy = int((y + origin_offset) / self.resolution)
        return gx, gy
        
    # Atualiza o mapa com a posição atual do robô
    def atualiza_mapa(self):
        gx, gy = self.world_to_grid(self.x, self.y)
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            self.grid_map[gy, gx] = 100
        self.publish_occupancy_grid()
        
    # Publica o mapa como uma mensagem OccupancyGrid
    def publish_occupancy_grid(self):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.grid_size
        grid_msg.info.height = self.grid_size
        origin = Pose()
        origin.position.x = - (self.grid_size * self.resolution) / 2
        origin.position.y = - (self.grid_size * self.resolution) / 2
        origin.position.z = 0.0
        origin.orientation.w = 1.0
        grid_msg.info.origin = origin
        grid_msg.data = self.grid_map.flatten().tolist()
        self.map_pub.publish(grid_msg)
        
    # Planejamento A*
    def a_star(self, start, goal):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = set()
        queue = PriorityQueue()
        queue.put((0, start))
        came_from = {}
        
        # Heurística de Manhattan
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        cost_so_far = {start: 0}

        while not queue.empty():
            _, current = queue.get()
            if current == goal:
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            visited.add(current)
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                    if self.grid_map[neighbor[1], neighbor[0]] == 100:
                        continue
                    new_cost = cost_so_far[current] + 1
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + heuristic(goal, neighbor)
                        queue.put((priority, neighbor))
                        came_from[neighbor] = current
        return None
        
# Função principal para iniciar o nó
def main(args=None):
    rclpy.init(args=args)
    node = RoboMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

