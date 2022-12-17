FROM python:3.11.1
RUN apt-get update && apt-get install -y xvfb libgl1-mesa-dev libx11-dev gcovr
RUN pip install -U pip wheel setuptools &&\
    pip install glcontext pytest numpy pyopengl
WORKDIR /app
COPY . .
RUN python setup.py build_ext --inplace && cp build/temp.*/zengl.gcno .
CMD xvfb-run python -m pytest tests && gcovr
