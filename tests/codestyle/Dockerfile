FROM python:3.11.2
RUN pip install -U pip wheel setuptools &&\
    pip install pytest pycodestyle flake8 isort
WORKDIR /app
COPY . .
RUN git init
CMD python -m pycodestyle --exclude=proxy.py,webgl.py --max-line-length=120 . &&\
    python -m flake8 --exclude=proxy.py,webgl.py --max-line-length=120 . &&\
    python -m isort -c --line-length=120 --multi-line=5 .
