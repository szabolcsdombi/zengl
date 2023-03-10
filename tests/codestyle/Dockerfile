FROM python:3.11.2
RUN pip install -U pip wheel setuptools &&\
    pip install pytest pycodestyle flake8 isort
WORKDIR /app
COPY . .
RUN git init
CMD python -m pycodestyle --exclude=proxy.py,webgl.py --max-line-length=120 . &&\
    python -m flake8 --exclude=proxy.py,webgl.py --max-line-length=120 . &&\
    python -m isort -c -o zengl -p window -p assets -p utils --gitignore --line-length=120 --multi-line=5 \
    --src=examples/example_project --src=examples/shapes_from_glsl .
