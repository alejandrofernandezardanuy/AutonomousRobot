TurtleBot3: Autonomous Navigation & Docking
Este proyecto implementa un sistema de navegación autónoma por fases para el robot TurtleBot3 Burger utilizando ROS2.

Características principales

Navegación por Waypoints: El robot sigue una ruta predefinida (Puntos A -> C -> D -> F -> Puerta) con control proporcional.


Evasión de Obstáculos (Bug2): Algoritmo reactivo que detecta obstáculos mediante LiDAR y realiza seguimiento de paredes (Wall Following).


Detección de Estación: Identificación de una estación de carga (4 pilares de 40x40 cm) mediante clustering de puntos LiDAR.


Aparcamiento de Precisión: Maniobra de docking final con un error inferior a 3 cm.

Arquitectura del Sistema
El sistema se divide en 5 módulos independientes coordinados por un nodo central:


navigation.py: Control de movimiento hacia objetivos.


obstacle_avoidance.py: Gestión de obstáculos con algoritmo Bug2.


station_detector.py: Localización geométrica de la estación.


docking_controller.py: Maniobra final de aparcamiento.


mission_logger.py: Registro de datos en tiempo real (CSV).

Resultados

Precisión: Error medio de posición por debajo de los 12 cm con SLAM activo.


Fiabilidad: Detección de la estación confirmada en el 100% de las pruebas.


Tiempo: Misión completa realizada en aproximadamente 5 minutos.
