#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from enum import Enum, auto
import math
import cv2
import numpy as np
import random
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import LaserScan, Imu, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid

# --- Enumeração dos Estados do Robô (MODIFICADO) ---
class EstadoRobo(Enum):
    AGUARDANDO_COMANDO = auto()
    EXPLORANDO = auto()
    BANDEIRA_DETECTADA = auto()
    NAVEGANDO_PARA_BANDEIRA = auto()
    POSICIONANDO_PARA_COLETA = auto()
    COLETANDO_BANDEIRA = auto()      # NOVO ESTADO
    RETORNANDO_A_BASE = auto()       # NOVO ESTADO
    MISSAO_CONCLUIDA = auto()

# --- Classe Principal do Nodo de Controle do Robô ---
class ControleRobo(Node):

    # --- Inicialização do Nodo ---
    def __init__(self):
        super().__init__('controle_robo')
        self.get_logger().info("Nó de Controle do Robô com Máquina de Estados Iniciado.")
        self.mask_publisher = self.create_publisher(Image, '/debug_camera/flag_mask', 10)
        
        # --- Configurações Essenciais ---
        self.bridge = CvBridge()
        
        # Cor da bandeira em BGR
        self.cor_bandeira_bgr_inferior = np.array([40, 40, 40])
        self.cor_bandeira_bgr_superior = np.array([40, 40, 40])

        # --- Limiares e Parâmetros de Controle ---
        self.limiar_distancia_posicionamento = 0.2
        self.limiar_angulo_para_posicionamento_final = 0.1
        self.distancia_final_ideal = 0.25
        self.tolerancia_distancia = 0.05
        self.tolerancia_angulo = 0.05

        self.estado_atual = EstadoRobo.EXPLORANDO

        # --- Variáveis de Percepção ---
        self.obstaculo_a_frente = False
        self.bandeira_visivel = False
        self.posicao_bandeira_relativa = None
        self.posicao_inicial = None # NOVO: Armazena a posição inicial
        self.odom_atual = None      # NOVO: Armazena a odometria atual

        # LIDAR
        self.distancia_lidar_frontal = float('inf') 
        self.min_dist_lado_esquerdo_fisico = float('inf')
        self.min_dist_lado_direito_fisico = float('inf')

        # --- Parâmetros para Estimativa de Distância Visual ---
        self.ALTURA_REAL_BANDEIRA_M = 0.2
        largura_imagem_px = 320
        fov_horizontal_rad = 1.57
        if math.tan(fov_horizontal_rad / 2.0) > 1e-5:
            self.distancia_focal_px = (largura_imagem_px / 2.0) / math.tan(fov_horizontal_rad / 2.0)
        else:
            self.distancia_focal_px = 277
        self.area_bandeira_detectada = 0.0

        # --- Novos parâmetros para evitar travamento ---
        self.min_valid_range = 0.1
        self.consistent_obstacle_threshold = 3
        self.obstacle_avoidance_timeout = 5.0
        self.last_obstacle_time = 0.0
        self.obstacle_count = 0
        self.consecutive_right_turns = 0
        self.max_consecutive_turns = 5

        # --- Limiares de Navegação ---
        self.limiar_obstaculo_frontal = 0.55
        
        # Para NAVEGANDO_PARA_BANDEIRA
        self.limiar_distancia_navegacao_para_posicionamento = 0.6
        self.limiar_angulo_para_avancar_nav = 0.25
        self.Kp_angular_nav = 0.6
        self.Kp_linear_nav = 0.1  
        self.velocidade_max_aproximacao_nav = 0.25
        self.velocidade_min_aproximacao_nav = 0.04
        self.velocidade_max_angular_nav = 0.5

        # Para POSICIONANDO_PARA_COLETA
        self.Kp_pos_ang = 0.3
        self.Kp_pos_lin = 0.08
        self.velocidade_max_linear_pos = 0.05
        self.velocidade_max_angular_pos = 0.2
        
        # NOVO: Para COLETANDO_BANDEIRA
        self.tempo_inicio_coleta = 0.0
        self.duracao_coleta_s = 3.0  # Robô para por 3 segundos

        # NOVO: Para RETORNANDO_A_BASE
        self.limiar_distancia_base = 0.25 # metros
        self.Kp_angular_base = 0.7
        self.Kp_linear_base = 0.15
        self.velocidade_max_retorno = 0.25
        self.limiar_angulo_retorno_base = 0.2 # radianos

        # Para Desvio Inteligente de Obstáculo
        self.velocidade_giro_desvio_obstaculo = 0.9
        self.margem_decisao_lateral_desvio = 0.1

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- Subscribers ---
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Image, '/robot_cam/labels_map', self.camera_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10) 
        self.create_subscription(Odometry, '/odom_gt', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/grid_map', self.map_callback, 10)

        # --- Timer ---
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.loop_maquina_estados)
        self.get_logger().info("ControleRobo inicializado e pronto.")
        
        self.ultimo_desvio_tempo = 0.0
        self.tempo_cooldown_pos_desvio = 2.0  # segundos

        # NOVO: estado de avanço forçado
        self.em_avanco_pos_desvio = False
        self.duracao_avanco_pos_desvio = 3  # segundos de avanço após desvio

        # Controle para avanço após desvio na navegação para bandeira
        self.ultimo_desvio_nav_tempo = 0.0
        self.em_avanco_pos_desvio_nav = False
        self.duracao_avanco_pos_desvio_nav = 3  # segundos para avançar após desvio


    # --- Funções Auxiliares (NOVO) ---
    def euler_from_quaternion(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    # --- Callbacks de Sensores ---
    def map_callback(self, msg: OccupancyGrid):
        self.map_data = msg

    def scan_callback(self, msg: LaserScan):
        num_ranges = len(msg.ranges)
        if num_ranges == 0:
            self.obstaculo_a_frente = False
            self.distancia_lidar_frontal = float('inf')
            self.min_dist_lado_esquerdo_fisico = float('inf')
            self.min_dist_lado_direito_fisico = float('inf')
            return

        angulo_abertura_frontal_graus = 30 
        if num_ranges == 360:
            offset_frontal = angulo_abertura_frontal_graus
        else: 
            offset_frontal = int(num_ranges * (angulo_abertura_frontal_graus / 360.0))

        indices_frente = list(range(num_ranges - offset_frontal, num_ranges)) + \
                           list(range(0, offset_frontal + 1))
        
        distancias_frente_raw = [d for i in indices_frente if 0 <= i < num_ranges and msg.ranges[i] > self.min_valid_range and not (math.isinf(msg.ranges[i]) or math.isnan(msg.ranges[i])) for d in [msg.ranges[i]]]
        
        if distancias_frente_raw:
            self.distancia_lidar_frontal = min(distancias_frente_raw)
            if self.distancia_lidar_frontal < self.limiar_obstaculo_frontal:
                self.obstacle_count += 1
                if self.obstacle_count >= self.consistent_obstacle_threshold:
                    if not self.obstaculo_a_frente:
                        self.get_logger().info(f'OBSTÁCULO FRONTAL detectado a {self.distancia_lidar_frontal:.2f}m.')
                        self.last_obstacle_time = self.get_clock().now().nanoseconds / 1e9
                    self.obstaculo_a_frente = True
            else:
                self.obstacle_count = 0
                if self.obstaculo_a_frente: self.get_logger().info('Caminho FRONTAL livre.')
                self.obstaculo_a_frente = False
        else:
            self.distancia_lidar_frontal = float('inf')
            self.obstaculo_a_frente = False

        angulo_inicio_setor_lateral_graus, angulo_fim_setor_lateral_graus = 20, 75
        
        idx_inicio_esq = angulo_inicio_setor_lateral_graus
        idx_fim_esq = angulo_fim_setor_lateral_graus
        dist_esq = [d for i in range(idx_inicio_esq, idx_fim_esq + 1) if 0 <= i < num_ranges and msg.ranges[i] > self.min_valid_range and not (math.isinf(msg.ranges[i]) or math.isnan(msg.ranges[i])) for d in [msg.ranges[i]]]
        self.min_dist_lado_esquerdo_fisico = min(dist_esq) if dist_esq else float('inf')

        idx_inicio_dir = num_ranges - angulo_fim_setor_lateral_graus
        idx_fim_dir = num_ranges - angulo_inicio_setor_lateral_graus
        dist_dir = [d for i in range(idx_inicio_dir, idx_fim_dir + 1) if 0 <= i < num_ranges and msg.ranges[i] > self.min_valid_range and not (math.isinf(msg.ranges[i]) or math.isnan(msg.ranges[i])) for d in [msg.ranges[i]]]
        self.min_dist_lado_direito_fisico = min(dist_dir) if dist_dir else float('inf')

    def camera_callback(self, msg: Image):
        try:
            cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Erro ao converter imagem com CvBridge: {e}")
            self.bandeira_visivel = False; self.posicao_bandeira_relativa = None; self.area_bandeira_detectada = 0.0
            return

        mask_bandeira = cv2.inRange(cv_image_bgr, self.cor_bandeira_bgr_inferior, self.cor_bandeira_bgr_superior)
        try:
            self.mask_publisher.publish(self.bridge.cv2_to_imgmsg(mask_bandeira, encoding="mono8"))
        except CvBridgeError as e:
            self.get_logger().error(f'Erro ao converter/publicar máscara de debug: {e}')

        height, width = cv_image_bgr.shape[:2]
        contours, _ = cv2.findContours(mask_bandeira, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            maior_contorno = max(contours, key=cv2.contourArea)
            self.area_bandeira_detectada = cv2.contourArea(maior_contorno)
            if self.area_bandeira_detectada > 50:
                M = cv2.moments(maior_contorno)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    _x, _y, _w, h_rect = cv2.boundingRect(maior_contorno)
                    
                    angulo_horizontal_rad = (float(cX - (width // 2)) / (width / 2.0)) * (1.57 / 2.0)
                    dist_visual_m = (self.ALTURA_REAL_BANDEIRA_M * self.distancia_focal_px) / h_rect if h_rect > 0 else float('inf')
                    
                    if not self.bandeira_visivel: self.get_logger().info(f"BANDEIRA VISÍVEL! (Área: {self.area_bandeira_detectada:.1f}, DistVis~{dist_visual_m:.2f}m, Angulo~{angulo_horizontal_rad:.2f}rad)")
                    self.bandeira_visivel = True
                    self.posicao_bandeira_relativa = (dist_visual_m, angulo_horizontal_rad)
                else: self.bandeira_visivel = False
            else:
                if self.bandeira_visivel: self.get_logger().info("Bandeira não mais proeminente.")
                self.bandeira_visivel = False
        else:
            if self.bandeira_visivel: self.get_logger().info("Bandeira perdida de vista.")
            self.bandeira_visivel = False

        if not self.bandeira_visivel:
            self.posicao_bandeira_relativa = None
            self.area_bandeira_detectada = 0.0

    def imu_callback(self, msg: Imu): pass
    
    # --- Callback de Odometria (MODIFICADO) ---
    def odom_callback(self, msg: Odometry):
        self.odom_atual = msg
        # Salva a primeira leitura de odometria como a posição inicial (base)
        if self.posicao_inicial is None:
            self.posicao_inicial = msg
            pos = self.posicao_inicial.pose.pose.position
            self.get_logger().info(f"Posição inicial (base) salva: (x={pos.x:.2f}, y={pos.y:.2f})")

    # --- Loop Principal da Máquina de Estados (MODIFICADO) ---
    def loop_maquina_estados(self):
        try:
            twist = Twist() 
            if self.estado_atual == EstadoRobo.AGUARDANDO_COMANDO: twist = self.logica_aguardando_comando()
            elif self.estado_atual == EstadoRobo.EXPLORANDO: twist = self.logica_explorando()
            elif self.estado_atual == EstadoRobo.BANDEIRA_DETECTADA: twist = self.logica_bandeira_detectada()
            elif self.estado_atual == EstadoRobo.NAVEGANDO_PARA_BANDEIRA: twist = self.logica_navegando_para_bandeira()
            elif self.estado_atual == EstadoRobo.POSICIONANDO_PARA_COLETA: twist = self.logica_posicionando_para_coleta()
            elif self.estado_atual == EstadoRobo.COLETANDO_BANDEIRA: twist = self.logica_coletando_bandeira()
            elif self.estado_atual == EstadoRobo.RETORNANDO_A_BASE: twist = self.logica_retornando_a_base()
            elif self.estado_atual == EstadoRobo.MISSAO_CONCLUIDA: twist = self.logica_missao_concluida()
            self.cmd_vel_pub.publish(twist)
        except Exception as e:
            self.get_logger().error(f"Erro na máquina de estados: {str(e)}")

    # --- Funções de Transição de Estado ---
    def mudar_estado(self, novo_estado: EstadoRobo):
        if self.estado_atual != novo_estado:
            self.get_logger().info(f"Mudando de estado: {self.estado_atual.name} -> {novo_estado.name}")
            self.estado_atual = novo_estado
            self.consecutive_right_turns = 0 # Reseta contagem de giros
            
            # NOVO: Inicia o timer ao entrar no estado de coleta
            if novo_estado == EstadoRobo.COLETANDO_BANDEIRA:
                self.tempo_inicio_coleta = self.get_clock().now().nanoseconds / 1e9

    # --- Lógicas de Estado ---
    def logica_aguardando_comando(self): return Twist()

    def logica_explorando(self):
        if self.bandeira_visivel and self.posicao_bandeira_relativa is not None:
            self.mudar_estado(EstadoRobo.BANDEIRA_DETECTADA)
            return Twist()
        
        twist = Twist()
        if self.obstaculo_a_frente:
            twist.linear.x = 0.0
            if self.min_dist_lado_direito_fisico > (self.min_dist_lado_esquerdo_fisico + self.margem_decisao_lateral_desvio):
                twist.angular.z = -self.velocidade_giro_desvio_obstaculo
            elif self.min_dist_lado_esquerdo_fisico > (self.min_dist_lado_direito_fisico + self.margem_decisao_lateral_desvio):
                twist.angular.z = self.velocidade_giro_desvio_obstaculo
            else: # Aleatório se o espaço for similar
                twist.angular.z = random.choice([-1, 1]) * self.velocidade_giro_desvio_obstaculo
        else:
            twist.linear.x = 0.25 # velocidade_linear_exploracao
            twist.angular.z = 0.0
        return twist

    def logica_bandeira_detectada(self):
        if self.bandeira_visivel and self.posicao_bandeira_relativa is not None:
            self.mudar_estado(EstadoRobo.NAVEGANDO_PARA_BANDEIRA)
        else: 
            self.mudar_estado(EstadoRobo.EXPLORANDO)
        return Twist()

    def logica_navegando_para_bandeira(self):
        if not self.bandeira_visivel or self.posicao_bandeira_relativa is None:
            self.mudar_estado(EstadoRobo.EXPLORANDO)
            return Twist()

        dist_vis, ang_cam = self.posicao_bandeira_relativa
        twist = Twist()
        tempo_atual = self.get_clock().now().nanoseconds / 1e9

        # --- DESVIO DE OBSTÁCULO ---
        if self.obstaculo_a_frente:
            self.get_logger().warn("Navegando: Obstáculo à frente. Executando desvio inteligente...")
            twist.linear.x = 0.0
            if self.min_dist_lado_direito_fisico > (self.min_dist_lado_esquerdo_fisico + self.margem_decisao_lateral_desvio):
                twist.angular.z = -self.velocidade_giro_desvio_obstaculo
            elif self.min_dist_lado_esquerdo_fisico > (self.min_dist_lado_direito_fisico + self.margem_decisao_lateral_desvio):
                twist.angular.z = self.velocidade_giro_desvio_obstaculo
            else:
                twist.angular.z = random.choice([-1, 1]) * self.velocidade_giro_desvio_obstaculo

            self.ultimo_desvio_nav_tempo = tempo_atual
            self.em_avanco_pos_desvio_nav = True
            return twist

        # --- AVANÇO FORÇADO APÓS O DESVIO ---
        if self.em_avanco_pos_desvio_nav:
            if (tempo_atual - self.ultimo_desvio_nav_tempo) < self.duracao_avanco_pos_desvio_nav:
                twist.linear.x = 0.15  # avança um pouco
                twist.angular.z = 0.0
                self.get_logger().debug("Avançando após desvio para afastar do obstáculo (Navegando)...")
                return twist
            else:
                self.get_logger().debug("Avanço pós-desvio finalizado (Navegando). Retomando navegação.")
                self.em_avanco_pos_desvio_nav = False

        # --- CONTROLE NORMAL DE NAVEGAÇÃO ---
        twist.angular.z = -self.Kp_angular_nav * ang_cam
        if abs(ang_cam) < self.limiar_angulo_para_avancar_nav:
            if dist_vis < self.limiar_distancia_posicionamento:
                self.mudar_estado(EstadoRobo.POSICIONANDO_PARA_COLETA)
                return Twist()
            else:
                twist.linear.x = max(self.velocidade_min_aproximacao_nav, self.Kp_linear_nav * dist_vis)
                twist.linear.x = min(twist.linear.x, self.velocidade_max_aproximacao_nav)
        else:
            twist.linear.x = 0.0

        twist.angular.z = np.clip(twist.angular.z, -self.velocidade_max_angular_nav, self.velocidade_max_angular_nav)
        return twist


        dist_vis, ang_cam = self.posicao_bandeira_relativa
        twist = Twist()
        
        if self.obstaculo_a_frente:
            self.get_logger().warn("Navegando: Obstáculo IMEDIATO. Desviando...")
            twist.linear.x = 0.0
            if self.min_dist_lado_direito_fisico > self.min_dist_lado_esquerdo_fisico:
                twist.angular.z = -self.velocidade_giro_desvio_obstaculo
            else:
                twist.angular.z = self.velocidade_giro_desvio_obstaculo
            return twist

        twist.angular.z = -self.Kp_angular_nav * ang_cam
        if abs(ang_cam) < self.limiar_angulo_para_avancar_nav:
            if dist_vis < self.limiar_distancia_posicionamento:
                self.mudar_estado(EstadoRobo.POSICIONANDO_PARA_COLETA)
                return Twist()
            else:
                twist.linear.x = max(self.velocidade_min_aproximacao_nav, self.Kp_linear_nav * dist_vis)
                twist.linear.x = min(twist.linear.x, self.velocidade_max_aproximacao_nav)
        else:
            twist.linear.x = 0.0
        
        twist.angular.z = np.clip(twist.angular.z, -self.velocidade_max_angular_nav, self.velocidade_max_angular_nav)
        return twist

    def logica_posicionando_para_coleta(self):
        if not self.bandeira_visivel or self.posicao_bandeira_relativa is None:
            self.mudar_estado(EstadoRobo.EXPLORANDO)
            return Twist()

        dist_vis, ang_cam = self.posicao_bandeira_relativa
        erro_dist = dist_vis - self.distancia_final_ideal
        erro_ang = ang_cam

        # --- Lógica de Transição (MODIFICADO) ---
        if abs(erro_dist) < self.tolerancia_distancia and abs(erro_ang) < self.tolerancia_angulo:
            self.get_logger().info(f">>> POSICIONAMENTO FINAL CONCLUÍDO! DistVis: {dist_vis:.2f}m, Angulo: {ang_cam:.2f}rad")
            self.mudar_estado(EstadoRobo.COLETANDO_BANDEIRA) # Mudar para coletar
            return Twist() 

        twist = Twist()
        twist.angular.z = -self.Kp_pos_ang * erro_ang
        if abs(erro_ang) < (self.tolerancia_angulo * 2.0): 
            twist.linear.x = self.Kp_pos_lin * erro_dist
        else:
            twist.linear.x = 0.0
        
        twist.linear.x = np.clip(twist.linear.x, -self.velocidade_max_linear_pos, self.velocidade_max_linear_pos)
        twist.angular.z = np.clip(twist.angular.z, -self.velocidade_max_angular_pos, self.velocidade_max_angular_pos)
        return twist

    # --- Lógicas de Estado (NOVAS) ---
    def logica_coletando_bandeira(self):
        self.get_logger().info(f"Coletando bandeira... (Aguardando {self.duracao_coleta_s}s)")
        
        tempo_atual = self.get_clock().now().nanoseconds / 1e9
        if (tempo_atual - self.tempo_inicio_coleta) > self.duracao_coleta_s:
            self.get_logger().info("Coleta finalizada. Iniciando retorno à base.")
            self.mudar_estado(EstadoRobo.RETORNANDO_A_BASE)
            
        return Twist() # Robô fica parado

    def logica_retornando_a_base(self):
        twist = Twist()
        tempo_atual = self.get_clock().now().nanoseconds / 1e9

        if self.odom_atual is None or self.posicao_inicial is None:
            self.get_logger().warn("Retornando: Odometria inicial ou atual não disponível. Aguardando...")
            return twist

        # --- DESVIO DE OBSTÁCULO ---
        if self.obstaculo_a_frente:
            self.get_logger().warn("Retornando: Obstáculo à frente. Executando desvio inteligente...")
            twist.linear.x = 0.0
            if self.min_dist_lado_direito_fisico > (self.min_dist_lado_esquerdo_fisico + self.margem_decisao_lateral_desvio):
                twist.angular.z = -self.velocidade_giro_desvio_obstaculo
            elif self.min_dist_lado_esquerdo_fisico > (self.min_dist_lado_direito_fisico + self.margem_decisao_lateral_desvio):
                twist.angular.z = self.velocidade_giro_desvio_obstaculo
            else:
                twist.angular.z = random.choice([-1, 1]) * self.velocidade_giro_desvio_obstaculo

            self.ultimo_desvio_tempo = tempo_atual
            self.em_avanco_pos_desvio = True
            return twist

        # --- AVANÇO FORÇADO APÓS O DESVIO ---
        if self.em_avanco_pos_desvio:
            if (tempo_atual - self.ultimo_desvio_tempo) < self.duracao_avanco_pos_desvio:
                twist.linear.x = 0.15  # avança um pouco
                twist.angular.z = 0.0
                self.get_logger().debug("Avançando após desvio para afastar do obstáculo...")
                return twist
            else:
                self.get_logger().debug("Avanço pós-desvio finalizado. Retomando navegação.")
                self.em_avanco_pos_desvio = False

        # --- NAVEGAÇÃO NORMAL PARA BASE ---
        pos_atual = self.odom_atual.pose.pose.position
        pos_base = self.posicao_inicial.pose.pose.position
        distancia_ate_base = math.sqrt((pos_base.x - pos_atual.x)**2 + (pos_base.y - pos_atual.y)**2)

        if distancia_ate_base < self.limiar_distancia_base:
            self.get_logger().info("CHEGOU À BASE!")
            self.mudar_estado(EstadoRobo.MISSAO_CONCLUIDA)
            return Twist()

        yaw_atual = self.euler_from_quaternion(self.odom_atual.pose.pose.orientation)
        angulo_para_base = math.atan2(pos_base.y - pos_atual.y, pos_base.x - pos_atual.x)
        erro_angulo = angulo_para_base - yaw_atual

        # Ajuste do erro de ângulo para [-pi, pi]
        if erro_angulo > math.pi:
            erro_angulo -= 2 * math.pi
        elif erro_angulo < -math.pi:
            erro_angulo += 2 * math.pi

        twist.angular.z = self.Kp_angular_base * erro_angulo
        if abs(erro_angulo) < self.limiar_angulo_retorno_base:
            twist.linear.x = self.Kp_linear_base * distancia_ate_base
            twist.linear.x = min(twist.linear.x, self.velocidade_max_retorno)
        else:
            twist.linear.x = 0.0

        return twist


    def logica_missao_concluida(self):
        self.get_logger().info("MISSÃO CONCLUÍDA! O robô irá parar.")
        return Twist() # Para o robô

# --- Função Principal ---
def main(args=None):
    rclpy.init(args=args)
    controle_robo_node = ControleRobo()
    try:
        rclpy.spin(controle_robo_node)
    except KeyboardInterrupt:
        controle_robo_node.get_logger().info("Nó de Controle encerrado (Ctrl+C).")
    except Exception as e:
        controle_robo_node.get_logger().error(f"Erro: {str(e)}")
    finally:
        try:
            stop_twist = Twist()
            controle_robo_node.cmd_vel_pub.publish(stop_twist)
            controle_robo_node.get_logger().info("Comando de parada final enviado.")
        except Exception as e:
            controle_robo_node.get_logger().warn(f"Não foi possível enviar comando de parada: {e}")
        
        controle_robo_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()