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

# --- Enumeração dos Estados do Robô ---
class EstadoRobo(Enum):
    AGUARDANDO_COMANDO = auto()
    EXPLORANDO = auto()
    BANDEIRA_DETECTADA = auto()
    NAVEGANDO_PARA_BANDEIRA = auto()
    POSICIONANDO_PARA_COLETA = auto()
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
        self.limiar_distancia_posicionamento = 0.2  # metros 
        self.limiar_angulo_para_posicionamento_final = 0.1 # radianos 
        self.distancia_final_ideal = 0.25 # metros
        self.tolerancia_distancia = 0.05  # metros
        self.tolerancia_angulo = 0.05     # radianos

        self.estado_atual = EstadoRobo.EXPLORANDO

        # --- Variáveis de Percepção ---
        self.obstaculo_a_frente = False
        self.bandeira_visivel = False
        self.posicao_bandeira_relativa = None

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
        self.velocidade_max_aproximacao_nav = 0.15
        self.velocidade_min_aproximacao_nav = 0.04
        self.velocidade_max_angular_nav = 0.5

        # Para POSICIONANDO_PARA_COLETA
        self.Kp_pos_ang = 0.3
        self.Kp_pos_lin = 0.08
        self.velocidade_max_linear_pos = 0.05
        self.velocidade_max_angular_pos = 0.2

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

        # --- Detecção Frontal com validação melhorada ---
        angulo_abertura_frontal_graus = 30 
        if num_ranges == 360:
            offset_frontal = angulo_abertura_frontal_graus
        else: 
            offset_frontal = int(num_ranges * (angulo_abertura_frontal_graus / 360.0))

        indices_frente = list(range(num_ranges - offset_frontal, num_ranges)) + \
                           list(range(0, offset_frontal + 1))
        
        distancias_frente_raw = []
        for i in indices_frente:
            if 0 <= i < num_ranges and msg.ranges[i] != float('inf') and msg.ranges[i] != float('nan') and msg.ranges[i] > self.min_valid_range:
                distancias_frente_raw.append(msg.ranges[i])
        
        if distancias_frente_raw:
            self.distancia_lidar_frontal = min(distancias_frente_raw)
            
            # Contagem consistente de obstáculos
            if self.distancia_lidar_frontal < self.limiar_obstaculo_frontal:
                self.obstacle_count += 1
                if self.obstacle_count >= self.consistent_obstacle_threshold:
                    if not self.obstaculo_a_frente:
                        self.get_logger().info(f'OBSTÁCULO FRONTAL detectado a {self.distancia_lidar_frontal:.2f}m.')
                        self.last_obstacle_time = self.get_clock().now().nanoseconds / 1e9
                    self.obstaculo_a_frente = True
            else:
                self.obstacle_count = 0
                if self.obstaculo_a_frente:
                    self.get_logger().info('Caminho FRONTAL livre.')
                self.obstaculo_a_frente = False
        else:
            self.distancia_lidar_frontal = float('inf')
            self.obstaculo_a_frente = False

        # --- Detecção Lateral ---
        angulo_inicio_setor_lateral_graus = 20
        angulo_fim_setor_lateral_graus = 75

        # Setor ESQUERDO FÍSICO
        idx_inicio_esq_fisico = angulo_inicio_setor_lateral_graus
        idx_fim_esq_fisico = angulo_fim_setor_lateral_graus
        
        indices_setor_esquerdo_fisico = []
        if 0 <= idx_inicio_esq_fisico < num_ranges and 0 <= idx_fim_esq_fisico < num_ranges and idx_inicio_esq_fisico <= idx_fim_esq_fisico:
            indices_setor_esquerdo_fisico = list(range(idx_inicio_esq_fisico, idx_fim_esq_fisico + 1))
        
        distancias_setor_esquerdo_raw = []
        for i in indices_setor_esquerdo_fisico:
            if msg.ranges[i] != float('inf') and msg.ranges[i] != float('nan') and msg.ranges[i] > self.min_valid_range:
                distancias_setor_esquerdo_raw.append(msg.ranges[i])
        
        if distancias_setor_esquerdo_raw:
            self.min_dist_lado_esquerdo_fisico = min(distancias_setor_esquerdo_raw)
        else:
            self.min_dist_lado_esquerdo_fisico = float('inf')

        # Setor DIREITO FÍSICO
        idx_inicio_dir_fisico = num_ranges - angulo_fim_setor_lateral_graus
        idx_fim_dir_fisico = num_ranges - angulo_inicio_setor_lateral_graus
        
        indices_setor_direito_fisico = []
        if 0 <= idx_inicio_dir_fisico < num_ranges and 0 <= idx_fim_dir_fisico < num_ranges and idx_inicio_dir_fisico <= idx_fim_dir_fisico:
            indices_setor_direito_fisico = list(range(idx_inicio_dir_fisico, idx_fim_dir_fisico + 1))

        distancias_setor_direito_raw = []
        for i in indices_setor_direito_fisico:
            if msg.ranges[i] != float('inf') and msg.ranges[i] != float('nan') and msg.ranges[i] > self.min_valid_range:
                distancias_setor_direito_raw.append(msg.ranges[i])

        if distancias_setor_direito_raw:
            self.min_dist_lado_direito_fisico = min(distancias_setor_direito_raw)
        else:
            self.min_dist_lado_direito_fisico = float('inf')

    # --- Callbacks de Câmera ---
    def camera_callback(self, msg: Image):
        try:
            cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Erro ao converter imagem com CvBridge: {e}")
            self.bandeira_visivel = False
            self.posicao_bandeira_relativa = None
            self.area_bandeira_detectada = 0.0
            return

        mask_bandeira = cv2.inRange(cv_image_bgr, self.cor_bandeira_bgr_inferior, self.cor_bandeira_bgr_superior)

        try:
            mask_msg = self.bridge.cv2_to_imgmsg(mask_bandeira, encoding="mono8")
            self.mask_publisher.publish(mask_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Erro ao converter/publicar máscara de debug: {e}')

        height, width = cv_image_bgr.shape[:2]
        center_x_image = width // 2

        contours, _ = cv2.findContours(mask_bandeira, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            maior_contorno = max(contours, key=cv2.contourArea)
            self.area_bandeira_detectada = cv2.contourArea(maior_contorno)
            limiar_area_minima = 50 

            if self.area_bandeira_detectada > limiar_area_minima:
                M = cv2.moments(maior_contorno)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    
                    angulo_horizontal_rad = 0.0
                    if (width / 2.0) > 1e-3:
                        angulo_horizontal_rad = (float(cX - center_x_image) / (width / 2.0)) * (1.57 / 2.0)

                    # Estimativa de Distância Visual
                    _x_rect, _y_rect, w_rect_pixels, h_rect_pixels = cv2.boundingRect(maior_contorno)
                    distancia_visual_calculada_m = float('inf')
                    if h_rect_pixels > 0:
                        distancia_visual_calculada_m = (self.ALTURA_REAL_BANDEIRA_M * self.distancia_focal_px) / h_rect_pixels
                    
                    if not self.bandeira_visivel:
                         self.get_logger().info(f"BANDEIRA VISÍVEL! (Cor BGR, Área: {self.area_bandeira_detectada:.1f}, DistVis~{distancia_visual_calculada_m:.2f}m, Angulo~{angulo_horizontal_rad:.2f}rad)")
                    
                    self.bandeira_visivel = True
                    self.posicao_bandeira_relativa = (distancia_visual_calculada_m, angulo_horizontal_rad)
                else:
                    self.bandeira_visivel = False
                    self.posicao_bandeira_relativa = None
                    self.area_bandeira_detectada = 0.0
            else:
                if self.bandeira_visivel: self.get_logger().info("Bandeira não mais proeminente (área pequena).")
                self.bandeira_visivel = False
                self.posicao_bandeira_relativa = None
                self.area_bandeira_detectada = 0.0
        else:
            if self.bandeira_visivel: self.get_logger().info("Bandeira perdida de vista (sem contornos).")
            self.bandeira_visivel = False
            self.posicao_bandeira_relativa = None
            self.area_bandeira_detectada = 0.0

    def imu_callback(self, msg: Imu): pass
    def odom_callback(self, msg: Odometry): pass

    # --- Loop Principal da Máquina de Estados ---
    def loop_maquina_estados(self):
        try:
            twist = Twist() 

            if self.estado_atual == EstadoRobo.AGUARDANDO_COMANDO:
                twist = self.logica_aguardando_comando()
            elif self.estado_atual == EstadoRobo.EXPLORANDO:
                twist = self.logica_explorando()
            elif self.estado_atual == EstadoRobo.BANDEIRA_DETECTADA:
                twist = self.logica_bandeira_detectada()
            elif self.estado_atual == EstadoRobo.NAVEGANDO_PARA_BANDEIRA:
                twist = self.logica_navegando_para_bandeira()
            elif self.estado_atual == EstadoRobo.POSICIONANDO_PARA_COLETA:
                twist = self.logica_posicionando_para_coleta()
            elif self.estado_atual == EstadoRobo.MISSAO_CONCLUIDA:
                twist = self.logica_missao_concluida()
            
            self.cmd_vel_pub.publish(twist)
        except Exception as e:
            self.get_logger().error(f"Erro na máquina de estado {str(e)}")
            return Twist()

    # --- Funções de Transição de Estado ---
    def mudar_estado(self, novo_estado: EstadoRobo):
        if self.estado_atual != novo_estado:
            self.get_logger().info(f"Mudando de estado: {self.estado_atual.name} -> {novo_estado.name}")
            self.estado_atual = novo_estado
            # Reseta contagem de giros consecutivos ao mudar de estado
            self.consecutive_right_turns = 0

    # --- Lógicas de Estado ---
    def logica_aguardando_comando(self):
        return Twist()

    # --- Lógicas de Exploranção ---
    def logica_explorando(self):
        twist = Twist()
        velocidade_linear_exploracao = 0.15 
        # Verifica se a bandeira foi detectada
        if self.bandeira_visivel and self.posicao_bandeira_relativa is not None:
            self.get_logger().info("Explorando: Bandeira detectada! Mudando para BANDEIRA_DETECTADA.")
            self.mudar_estado(EstadoRobo.BANDEIRA_DETECTADA)
            return Twist()
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Melhora a lógica de desvio de obstáculo
        if (self.obstaculo_a_frente and 
            (current_time - self.last_obstacle_time) > self.obstacle_avoidance_timeout):
            self.get_logger().warn("STUCK DETECTADO! Tentar manobra de fuga.")
            twist.linear.x = -0.1
            twist.angular.z = 0.5
            self.last_obstacle_time = current_time  # Reseta o timer
            return twist
            
        elif self.obstaculo_a_frente:
            self.get_logger().info("Explorando: Obstáculo frontal detectado. Decidindo direção inteligente do giro...")
            twist.linear.x = 0.0
            
            # Decisão de desvio com base no espaço lateral
            if (self.min_dist_lado_direito_fisico > 
                (self.min_dist_lado_esquerdo_fisico + self.margem_decisao_lateral_desvio)):
                self.get_logger().info(f"--> EXPLORANDO: Mais espaço à DIREITA FÍSICA ({self.min_dist_lado_direito_fisico:.2f}m > {self.min_dist_lado_esquerdo_fisico:.2f}m + margem). Virando para DIREITA.")
                twist.angular.z = -self.velocidade_giro_desvio_obstaculo
                self.consecutive_right_turns += 1
            elif (self.min_dist_lado_esquerdo_fisico > 
                  (self.min_dist_lado_direito_fisico + self.margem_decisao_lateral_desvio)):
                self.get_logger().info(f"--> EXPLORANDO: Mais espaço à ESQUERDA FÍSICA ({self.min_dist_lado_esquerdo_fisico:.2f}m > {self.min_dist_lado_direito_fisico:.2f}m + margem). Virando para ESQUERDA.")
                twist.angular.z = self.velocidade_giro_desvio_obstaculo
                self.consecutive_right_turns = 0
            else:
                # Decisão aleatória quando os espaços são similares
                if self.consecutive_right_turns >= self.max_consecutive_turns:
                    self.get_logger().info("EXPLORANDO: Muitos giros consecutivos à DIREITA. Virando para ESQUERDA.")
                    twist.angular.z = self.velocidade_giro_desvio_obstaculo
                    self.consecutive_right_turns = 0
                else:
                    turn_direction = random.choice([-1, 1])  
                    self.get_logger().info(f"--> EXPLORANDO: Espaço lateral similar. Virando {'DIREITA' if turn_direction == -1 else 'ESQUERDA'} (aleatório).")
                    twist.angular.z = turn_direction * self.velocidade_giro_desvio_obstaculo
                    if turn_direction == -1:
                        self.consecutive_right_turns += 1
                    else:
                        self.consecutive_right_turns = 0
        else:
            twist.linear.x = velocidade_linear_exploracao
            twist.angular.z = 0.0
            self.consecutive_right_turns = 0
            
        return twist

    # --- Lógica quando a bandeira é detectada ---
    def logica_bandeira_detectada(self):
        twist = Twist() 
        self.get_logger().info(f"Bandeira Detectada! Posição relativa (DistVis, AngCam): {self.posicao_bandeira_relativa}")
        
        if self.posicao_bandeira_relativa is not None and self.bandeira_visivel:
            self.mudar_estado(EstadoRobo.NAVEGANDO_PARA_BANDEIRA)
        else: 
            self.get_logger().warn("Bandeira detectada mas info inválida/perdida rapidamente. Voltando a EXPLORAR.")
            self.bandeira_visivel = False
            self.posicao_bandeira_relativa = None
            self.mudar_estado(EstadoRobo.EXPLORANDO)
        return twist

    # --- Lógica de Navegação para a Bandeira ---
    def logica_navegando_para_bandeira(self):
        twist = Twist()
        
        if not self.bandeira_visivel or self.posicao_bandeira_relativa is None:
            self.get_logger().warn("Navegando: Bandeira perdida/inválida. Voltando a EXPLORAR.")
            self.mudar_estado(EstadoRobo.EXPLORANDO)
            return twist

        distancia_visual_bandeira, angulo_bandeira_camera = self.posicao_bandeira_relativa
        self.get_logger().info(f"Navegando: Angulo_CAM: {angulo_bandeira_camera:.2f}rad, Dist_VISUAL: {distancia_visual_bandeira:.2f}m, Obst_Imediato: {self.obstaculo_a_frente}")

        # Desvio de obstáculo imediato
        if self.obstaculo_a_frente:
            self.get_logger().warn("Navegando: Obstáculo IMEDIATO. Desviando com prioridade (escolha inteligente)...")
            twist.linear.x = 0.0

            if (self.min_dist_lado_direito_fisico > 
                (self.min_dist_lado_esquerdo_fisico + self.margem_decisao_lateral_desvio)):
                self.get_logger().info(f"--> NAVEGANDO: Mais espaço à DIREITA FÍSICA ({self.min_dist_lado_direito_fisico:.2f}m). Virando para DIREITA.")
                twist.angular.z = -self.velocidade_giro_desvio_obstaculo
            elif (self.min_dist_lado_esquerdo_fisico > 
                  (self.min_dist_lado_direito_fisico + self.margem_decisao_lateral_desvio)):
                self.get_logger().info(f"--> NAVEGANDO: Mais espaço à ESQUERDA FÍSICA ({self.min_dist_lado_esquerdo_fisico:.2f}m). Virando para ESQUERDA.")
                twist.angular.z = self.velocidade_giro_desvio_obstaculo
            else:
                # Decisão aleatória quando os espaços são similares
                turn_direction = random.choice([-1, 1])
                self.get_logger().info(f"--> NAVEGANDO: Espaço lateral similar. Virando {'DIREITA' if turn_direction == -1 else 'ESQUERDA'} (aleatório).")
                twist.angular.z = turn_direction * self.velocidade_giro_desvio_obstaculo
            return twist

        # Navegação normal até a bandeira
        twist.angular.z = -self.Kp_angular_nav * angulo_bandeira_camera

        if abs(angulo_bandeira_camera) < self.limiar_angulo_para_avancar_nav: 
            self.get_logger().debug("Navegando: Bem alinhado. Verificando distância VISUAL para progredir/posicionar.")
            
            if distancia_visual_bandeira < self.limiar_distancia_posicionamento:
                if abs(angulo_bandeira_camera) < self.limiar_angulo_para_posicionamento_final:
                    self.get_logger().info(f">>> Alvo próximo (DistVis {distancia_visual_bandeira:.2f}m) e BEM ALINHADO. Transição para POSICIONANDO.")
                    self.mudar_estado(EstadoRobo.POSICIONANDO_PARA_COLETA)
                    return Twist()
                else:
                    self.get_logger().warn(f"Navegando: Perto (DistVis {distancia_visual_bandeira:.2f}m), mas não PERFEITAMENTE alinhado (CAM {angulo_bandeira_camera:.2f}rad). Ajustando ângulo.")
                    twist.linear.x = 0.0
            else: 
                if distancia_visual_bandeira != float('inf') and distancia_visual_bandeira > 0:
                    twist.linear.x = max(self.velocidade_min_aproximacao_nav, self.Kp_linear_nav * distancia_visual_bandeira)
                    twist.linear.x = min(twist.linear.x, self.velocidade_max_aproximacao_nav)
                else: 
                    twist.linear.x = self.velocidade_min_aproximacao_nav if distancia_visual_bandeira > 0 else 0.0 
        else:
            self.get_logger().debug("Navegando: Desalinhado com a bandeira. Priorizando giro (linear.x = 0).")
            twist.linear.x = 0.0

        twist.angular.z = np.clip(twist.angular.z, -self.velocidade_max_angular_nav, self.velocidade_max_angular_nav)
        
        return twist

    # --- Lógica de Posicionamento para Coleta ---
    def logica_posicionando_para_coleta(self):
        twist = Twist()
        self.get_logger().info("Posicionando para coleta...")

        if not self.bandeira_visivel or self.posicao_bandeira_relativa is None:
            self.get_logger().warn("Posicionando: Bandeira perdida. Voltando a EXPLORAR.")
            self.mudar_estado(EstadoRobo.EXPLORANDO)
            return twist

        distancia_atual_visual, angulo_atual_camera = self.posicao_bandeira_relativa
        self.get_logger().debug(f"Posicionando: Angulo_CAM: {angulo_atual_camera:.2f}rad, Dist_VISUAL: {distancia_atual_visual:.2f}m")

        erro_distancia = distancia_atual_visual - self.distancia_final_ideal
        erro_angulo = angulo_atual_camera

        if abs(erro_distancia) < self.tolerancia_distancia and abs(erro_angulo) < self.tolerancia_angulo:
            self.get_logger().info(f">>> POSICIONAMENTO FINAL CONCLUÍDO! DistVis: {distancia_atual_visual:.2f}m, Angulo: {angulo_atual_camera:.2f}rad")
            self.mudar_estado(EstadoRobo.MISSAO_CONCLUIDA)
            return Twist() 

        twist.angular.z = -self.Kp_pos_ang * erro_angulo 
        
        if abs(erro_angulo) < (self.tolerancia_angulo * 2.0): 
            if distancia_atual_visual != float('inf') and distancia_atual_visual > 0:
                 twist.linear.x = self.Kp_pos_lin * erro_distancia
            else:
                 twist.linear.x = 0.0
        else:
            twist.linear.x = 0.0 
        
        twist.linear.x = np.clip(twist.linear.x, -self.velocidade_max_linear_pos, self.velocidade_max_linear_pos)
        twist.angular.z = np.clip(twist.angular.z, -self.velocidade_max_angular_pos, self.velocidade_max_angular_pos)
        
        return twist
    # --- Lógica de Missão Concluída ---
    def logica_missao_concluida(self):
        self.get_logger().info("MISSÃO CONCLUÍDA!")
        return Twist()

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
            controle_robo_node.get_logger().info("Comando de parada enviado.")
        except Exception as e:
            controle_robo_node.get_logger().warn(f"Não foi possível enviar comando de parada: {e}")
        
        controle_robo_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
