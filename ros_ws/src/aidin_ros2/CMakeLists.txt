cmake_minimum_required(VERSION 3.8)
project(aidin_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(lxros REQUIRED)


include_directories(include)

add_executable(aidin_ft_sensor_node src/aidin_ros2_node.cpp)
ament_target_dependencies(aidin_ft_sensor_node std_srvs sensor_msgs lxros rclcpp geometry_msgs)

install(TARGETS
  aidin_ft_sensor_node
  DESTINATION lib/${PROJECT_NAME}
)
#ament_python_install_package(python/${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  #aidin_ros2/ft_sensor_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
