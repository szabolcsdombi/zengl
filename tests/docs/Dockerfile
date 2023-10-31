FROM python:3.12.0
RUN pip install -U pip wheel setuptools &&\
    pip install sphinx furo
WORKDIR /app
COPY . .
CMD sphinx-build docs build
