import os
import sys

from setuptools import Extension, setup

extra_compile_args = []
extra_link_args = []

stubs = {
    'packages': ['zengl-stubs'],
    'package_data': {'zengl-stubs': ['__init__.pyi']},
    'include_package_data': True,
}

if sys.platform.startswith('linux'):
    extra_compile_args = ['-Wno-write-strings', '-Wno-narrowing']

if sys.platform.startswith('darwin'):
    extra_compile_args = ['-std=c++11', '-Wno-writable-strings', '-Wno-c++11-narrowing']

if os.getenv('ZENGL_COVERAGE'):
    extra_compile_args += ['-O0', '--coverage']
    extra_link_args += ['-O0', '--coverage']

if os.getenv('ZENGL_NO_STUBS'):
    stubs = {}

ext = Extension(
    name='zengl',
    sources=['zengl.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

with open('README.md') as readme:
    long_description = readme.read()

setup(
    name='zengl',
    version='1.13.0',
    ext_modules=[ext],
    py_modules=['_zengl'],
    license='MIT',
    python_requires='>=3.6',
    platforms=['any'],
    description='Self-Contained OpenGL Rendering Pipelines for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Szabolcs Dombi',
    author_email='cprogrammer1994@gmail.com',
    url='https://github.com/szabolcsdombi/zengl/',
    project_urls={
        'Documentation': 'https://zengl.readthedocs.io/',
        'Source': 'https://github.com/szabolcsdombi/zengl/',
        'Bug Tracker': 'https://github.com/szabolcsdombi/zengl/issues/',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Games/Entertainment',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    keywords=[
        'OpenGL',
        'rendering',
        'graphics',
        'visualization',
    ],
    **stubs,
)
