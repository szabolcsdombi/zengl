import os
import re
import sys

from setuptools import Extension, setup

py_limited_api = True
extra_compile_args = []
extra_link_args = []
define_macros = []
cmdclass = {}

stubs = {
    'packages': ['zengl-stubs'],
    'package_data': {'zengl-stubs': ['__init__.pyi']},
    'include_package_data': True,
}

if sys.hexversion < 0x030B0000:
    py_limited_api = False

if sys.platform.startswith('darwin'):
    extra_compile_args += ['-Wno-writable-strings']

if os.getenv('PYODIDE'):
    with open('zengl.js') as zengl_js:
        code = re.sub(r'\s+', ' ', zengl_js.read())
        define_macros += [('EXTERN_GL', f'"{code}"')]

    py_limited_api = False
    stubs = {}

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

if py_limited_api:
    from wheel.bdist_wheel import bdist_wheel

    class bdist_wheel_abi3(bdist_wheel):
        def get_tag(self):
            python, abi, plat = super().get_tag()

            if python.startswith('cp'):
                return 'cp311', 'abi3', plat

            return python, abi, plat

    cmdclass = {'cmdclass': {'bdist_wheel': bdist_wheel_abi3}}
    define_macros += [('Py_LIMITED_API', 0x030B0000)]

ext = Extension(
    name='zengl',
    sources=['zengl.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
    py_limited_api=py_limited_api,
)

with open('README.md') as readme:
    long_description = readme.read()

setup(
    name='zengl',
    version='2.5.0',
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
    **cmdclass,
    **stubs,
)
