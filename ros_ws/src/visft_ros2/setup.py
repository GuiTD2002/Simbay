from setuptools import setup

package_name = 'visft_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS2 driver for VIS_FT force-torque sensor',
    license='MIT',
    entry_points={
        'console_scripts': [
            'visft_node = visft_ros2.visft_node:main',
        ],
    },
)
