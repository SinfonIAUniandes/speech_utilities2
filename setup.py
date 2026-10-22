import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'speech_utilities2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Personas del robot y fragmentos compartidos del system prompt.
        (os.path.join('share', package_name, 'data'), glob('data/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotica',
    maintainer_email='si.escobar@uniandes.edu.co',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'microphone_node = speech_utilities2.microphone_node:main',
            'conversation_node = speech_utilities2.conversation_node:main',
            'radio_node = speech_utilities2.radio_node:main',
        ],
    },
)
