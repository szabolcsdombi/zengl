import os
import re
import sys
import sysconfig

from setuptools import Extension, setup

extra_compile_args = []
extra_link_args = []
define_macros = []

if sys.platform.startswith('darwin'):
    extra_compile_args += ['-Wno-writable-strings']

if os.getenv('PYODIDE') or str(sysconfig.get_config_var('HOST_GNU_TYPE')).startswith('wasm'):
    with open('zengl.js') as zengl_js:
        code = re.sub(r'\s+', ' ', zengl_js.read())
        define_macros += [('EXTERN_GL', f'"{code}"')]

if os.getenv('ZENGL_COVERAGE'):
    extra_compile_args += ['-O0', '--coverage']
    extra_link_args += ['-O0', '--coverage']

if os.getenv('ZENGL_WARNINGS'):
    if sys.platform.startswith('linux'):
        extra_compile_args += [
            '-g3',
            '-Wall',
            '-Wextra',
            '-Wconversion',
            '-Wdouble-promotion',
            '-Wno-unused-parameter',
            '-Wno-cast-function-type',
            '-Werror',
        ]
        extra_link_args += [
            '-fsanitize=undefined',
        ]

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
    version='2.6.0',
    ext_modules=[ext],
    py_modules=['_zengl'],
    license='MIT',
    platforms=['any'],
    description='OpenGL Pipelines for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Szabolcs Dombi',
    author_email='szabolcs@szabolcsdombi.com',
    url='https://github.com/szabolcsdombi/zengl/',
    project_urls={
        'Documentation': 'https://zengl.readthedocs.io/',
        'Source': 'https://github.com/szabolcsdombi/zengl/',
        'Bug Tracker': 'https://github.com/szabolcsdombi/zengl/issues/',
    },
    keywords=[
        'OpenGL',
        'rendering',
        'graphics',
        'shader',
        'gpu',
    ],
    packages=['zengl-stubs'],
    package_data={'zengl-stubs': ['__init__.pyi']},
    include_package_data=True,
)
