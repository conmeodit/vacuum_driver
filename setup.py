from glob import glob

from setuptools import find_packages, setup


package_name = 'vacuum_driver'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/config', glob('config/*.yaml')),
        (f'share/{package_name}/rviz', glob('rviz/*.rviz')),
        (f'share/{package_name}/urdf', glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cuong',
    maintainer_email='cuong@todo.todo',
    description='Vacuum robot Webots communication driver.',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'pure_driver = vacuum_driver.pure_driver:main',
            'autonomous_cleaning_node = vacuum_driver.autonomous_cleaning_node:main',
            'slam_session_manager_node = vacuum_driver.slam_session_manager_node:main',
        ],
    },
)
