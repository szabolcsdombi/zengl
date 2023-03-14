FROM python:3.10.2
RUN python3 -m pip install pyodide-build
RUN git clone https://github.com/emscripten-core/emsdk.git &&\
    /emsdk/emsdk install `pyodide config get emscripten_version` &&\
    /emsdk/emsdk activate `pyodide config get emscripten_version`

WORKDIR /zengl/
COPY _zengl.py zengl.cpp setup.py pyproject.toml MANIFEST.in README.md LICENSE /zengl/
RUN echo 'source /emsdk/emsdk_env.sh && ZENGL_NO_STUBS=yes pyodide build --exports PyInit_zengl' | bash

WORKDIR /web/
RUN cp /zengl/dist/*.whl /web/zengl.whl
COPY examples/webgl/public/* /web/
CMD python -m http.server --bind 0.0.0.0 8000
