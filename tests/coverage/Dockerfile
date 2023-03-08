FROM ubuntu:22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip libgl1-mesa-dev libegl1-mesa-dev libx11-dev gcovr
RUN pip install -U pip wheel setuptools &&\
    pip install https://github.com/moderngl/glcontext/archive/refs/tags/2.4.0.zip numpy pytest pyopengl
WORKDIR /app
COPY . .
ENV ZENGL_COVERAGE=yes
RUN python3 setup.py build_ext --inplace && cp build/temp.*/zengl.gcno .
CMD python3 -X dev -m pytest -s -vvv tests && gcovr
