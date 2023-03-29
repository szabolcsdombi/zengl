from setuptools import setup

setup(
    name='assets',
    version='0.1.0',
    packages=['assets'],
    package_data={
        'assets': [
            'assets/crate-texture.bin',
            'assets/cube-model.bin',
            'assets/food-model.bin',
        ],
    },
    include_package_data=True,
)
