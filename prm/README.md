# PRM - Programação de Robôs Móveis

**Disciplina SSC0712**  
Oferecida para os cursos de Engenharia de Computação e áreas afins na **USP São Carlos**

Este repositório contém o material da disciplina *Programação de Robôs Móveis*, focada no desenvolvimento de soluções em robótica móvel utilizando **ROS 2 Humble** e o simulador **Gazebo Fortress**.

## 📦 Tecnologias utilizadas

- ROS 2 Humble
- Gazebo Fortress
- Python
- RViz / Gazebo GUI
- [teleop_twist_keyboard](https://github.com/ros2/teleop_twist_keyboard)

---

## 🚀 Como utilizar o pacote

### 1. Clonar o repositório

Acesse a pasta `src` do seu workspace ROS 2:

```bash
cd ~/ros2_ws/src/
git clone https://github.com/matheusbg8/prm.git
````

### 2. Instalar dependências

Instale as dependências do pacote com:

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

> Certifique-se de ter rodado previamente `sudo rosdep init` e `rosdep update`, se for a primeira vez usando o `rosdep`.

### 3. Compilar o workspace

Certifique-se de estar na **raiz do seu workspace** (geralmente `~/ros2_ws`) antes de compilar:

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select prm
```

### 4. Atualizar o ambiente do terminal

```bash
source install/local_setup.bash
```

---

## 🧪 Executando a simulação

### 1. Iniciar o mundo no Gazebo

```bash
ros2 launch prm inicia_simulacao.launch.py
```

### 2. Carregar o robô no ambiente

Em um **novo terminal** (não se esqueça de `source install/local_setup.bash`):

```bash
ros2 launch prm carrega_robo.launch.py
```

### 3. Controle automático (demonstração)

Em outro terminal:

```bash
ros2 run prm controle_robo
```

### 4. **Controle manual (alternativa ao passo 3)**

Você pode controlar o robô usando o teclado, como alternativa ao controle automático:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

#### Instalar `teleop_twist_keyboard` (caso não esteja disponível)

```bash
sudo apt install ros-humble-teleop-twist-keyboard
```

> **Importante**: execute **o passo 3 *ou* o passo 4**, dependendo se deseja usar o controle automático ou manual.
