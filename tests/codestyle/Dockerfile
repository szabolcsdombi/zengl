FROM python:3.11.2
RUN pip install -U pip wheel setuptools &&\
    pip install pytest pycodestyle flake8 isort
WORKDIR /app
COPY . .
RUN git init
CMD python -m pycodestyle --max-line-length=120 . &&\
    python -m flake8 --max-line-length=120 . &&\
    python -m isort -c -o zengl -p window -p assets -p utils --gitignore --line-length=120 --multi-line=5 \
    --src=examples/example_project --src=examples/shapes_from_glsl .
