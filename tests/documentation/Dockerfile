FROM python:3.11.2
RUN pip install -U pip wheel setuptools &&\
    pip install sphinx furo
WORKDIR /app
COPY . .
CMD python setup.py build_sphinx
