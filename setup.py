import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'imitation_nav_training'

logs_files = [f for f in glob('logs/*') if os.path.isfile(f)]
logs_files += [f for f in glob('logs/result/*') if os.path.isfile(f)]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['imitation_nav_training', 'imitation_nav_training.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'logs'), logs_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kyo',
    maintainer_email='s21c1135sc@s.chibakoudai.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_collector_node = imitation_nav_training.data_collector_node:main',
            'controller_node = imitation_nav_training.controller_node:main',
            'augment_node = imitation_nav_training.augment.gamma_augment:main',
            'train.py = imitation_nav_training.train_model:main',
        ],
    },
)
