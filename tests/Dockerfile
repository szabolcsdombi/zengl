FROM python:3.9.7
RUN apt-get update && apt-get install -y xvfb libgl1-mesa-dev libx11-dev
RUN pip install -U pip wheel setuptools &&\
    pip install glcontext numpy pytest pycodestyle pyflakes isort sphinx sphinx_rtd_theme
WORKDIR /app
COPY . .
RUN python -m pycodestyle --max-line-length=120 .
RUN python -m pyflakes .
RUN python -m isort -c -o zengl -p window -p assets -p utils --src=examples/example_project .
RUN python setup.py sdist
RUN python setup.py build_sphinx
RUN find dist/zengl-*.tar.gz | xargs pip install
CMD xvfb-run -s "-screen 0 1x1x24" python -m pytest tests
