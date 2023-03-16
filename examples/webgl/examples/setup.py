from setuptools import setup

setup(
    name='examples',
    version='0.1.0',
    packages=['examples'],
    package_data={
        'examples': [
            'assets/crate-texture.bin',
            'assets/cube-model.bin',
            'assets/food-model.bin',
        ],
    },
    include_package_data=True,
)
