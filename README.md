# Trabalho 1 - Programação de Robôs Móveis (SSC0712)
### Universidade de São Paulo (USP)
### Instituto de Ciências Matemáticas e de Computação (ICMC)

Projeto de familiarização com *ROS2 Humble* 🐢 e com simulação em *Gazebo Fortress* através da modelagem em Python de um robô terrestre autônomo que tem por objetivo capturar uma bandeira

## 📌 Orientador
	Prof. Dr. Matheus Machado dos Santos

## ✏️ Desenvolvido por
	Ana Rita Marega Gonçalves	Estudante de Ciências de Computação		N°USP: 15746365
	Beatriz Aparecida Diniz		Estudante de Ciências de Computação		N°USP: 11925430
 	Felipe de Oliveira Gomes	Estudante de Engenharia de Computação		N°USP: 14613841
	Gabriel dos Santos Alves	Estudante de Engenharia de Computação		N°USP: 14614032

## 📂 Repositório-base 

	https://github.com/matheusbg8/prm

## ⬇️ Instalação e compilação do Workspace

	cd ~/
 	git clone https://github.com/Lipe-Gomes/trabalho1_prm.git
	cd trabalho1_prm
	colcon build --symlink-install --packages-select prm

## 💻 Execute o ambiente de simulção *Gazebo*

  	source ~/trabalho1_prm/install/local_setup.bash
  	ros2 launch prm inicia_simulacao.launch.py

## 🔧 Em um outro terminal, carregue o robô

 	source ~/trabalho1_prm/install/local_setup.bash
 	ros2 launch prm carrega_robo.launch.py

## 🚩 Em um terceiro terminal, execute o algoritmo de *Catch the Flag!*, baseado no algoritmo de navegação A*

 	source ~/trabalho1_prm/install/local_setup.bash
  	ros2 run prm controle_robo

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](./LICENSE) para mais detalhes.
