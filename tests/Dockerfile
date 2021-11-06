FROM python:3.9.7
RUN apt-get update && apt-get install -y xvfb libgl1-mesa-dev libx11-dev
RUN pip install -U pip wheel setuptools && pip install glcontext numpy pytest
WORKDIR /app
COPY . .
RUN python setup.py sdist
RUN find dist/zengl-*.tar.gz | xargs pip install
CMD xvfb-run -s "-screen 0 1x1x24" python -m pytest tests
