FROM python:3.11.2
RUN pip install -U pip wheel setuptools &&\
    pip install sphinx furo
WORKDIR /app
COPY . .
CMD sphinx-build docs build
