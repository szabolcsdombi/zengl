import os
import sys

from setuptools import Extension, setup

extra_compile_args = []
extra_link_args = []
define_macros = []

stubs = {
    'packages': ['zengl-stubs'],
    'package_data': {'zengl-stubs': ['__init__.pyi']},
    'include_package_data': True,
}

if sys.platform.startswith('linux'):
    extra_compile_args = []

if sys.platform.startswith('darwin'):
    extra_compile_args = ['-Wno-writable-strings']

if os.getenv('PYODIDE'):
    import re

    with open('zengl.js') as f:
        zengl_js = re.sub(r'\s+', ' ', f.read(), flags=re.M)

    define_macros = [
        ('ZENGL_JS', f'"{zengl_js}"'),
        ('WEB', None),
    ]
    stubs = {}

if os.getenv('ZENGL_COVERAGE'):
    extra_compile_args += ['-O0', '--coverage']
    extra_link_args += ['-O0', '--coverage']

if os.getenv('ZENGL_WARNINGS'):
    if sys.platform.startswith('linux'):
        extra_compile_args += [
            '-g3', '-Wall', '-Wextra', '-Wconversion', '-Wdouble-promotion', '-Wno-unused-parameter',
            '-Wno-cast-function-type', '-fsanitize=undefined', '-Werror',
        ]

if os.getenv('ZENGL_NO_STUBS'):
    stubs = {}

ext = Extension(
    name='zengl',
    sources=['zengl.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
)

with open('README.md') as readme:
    long_description = readme.read()

setup(
    name='zengl',
    version='1.16.0',
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
