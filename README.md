# Autonomous Navigation

A ROS2 Python package for autonomous robot navigation and docking on charging station, implementing a complete mission with waypoint navigation, obstacle avoidance, station detection, and precision docking.

For a video demonstration, see the documentation.

## Features

- **Waypoint Navigation**: Follows predefined waypoints with proportional control for orientation and heading correction.
- **Obstacle Avoidance**: Reactive Bug2 algorithm for safe navigation around obstacles.
- **Station Detection**: Detects square stations (40x40 cm) using laser scan data.
- **Docking Controller**: Precision docking with obstacle avoidance.
- **Mission Phases**: 
  - Phase I: Navigate to specific points (A → C → D → F → Q).
  - Phase II: Exploration loop with station detection, return to base, and docking.
- **Localization**: Uses SLAM Toolbox with odometry fallback.
- **Logging**: CSV mission logs and map export (PGM + YAML).

## Installation

1. Ensure ROS2 Humble (or compatible) is installed.
2. Clone this repository into your ROS2 workspace `src/` directory.
3. Install dependencies: `rosdep install --from-paths src --ignore-src -r -y`
4. Build: `colcon build --packages-select autonomous_nav`
5. Source: `source install/setup.bash`

## Usage

Launch the main mission node:

```bash
ros2 launch autonomous_nav mission_launch.py
```

Or run individual nodes for debugging:

```bash
ros2 run autonomous_nav mission_node
```

## Dependencies

- rclpy
- geometry_msgs
- nav_msgs
- sensor_msgs

## License

TODO: License declaration</content>
<parameter name="filePath">/home/alex/Documentos/robo/Entrega Robotica Autonomous Navigation/autonomous_nav/README.md
