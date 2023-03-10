FROM python:3.10.2
RUN python3 -m pip install pyodide-build
RUN git clone https://github.com/emscripten-core/emsdk.git &&\
    /emsdk/emsdk install `pyodide config get emscripten_version` &&\
    /emsdk/emsdk activate `pyodide config get emscripten_version`

WORKDIR /zengl/
COPY _zengl.py zengl.cpp setup.py pyproject.toml MANIFEST.in README.md LICENSE examples/webgl/webgl.py /zengl/
RUN sed -i -r '/(packages|package_data|include_package_data)=/d' setup.py && cat webgl.py >> _zengl.py
RUN echo 'source /emsdk/emsdk_env.sh && pyodide build' | bash

WORKDIR /web/
RUN cp /zengl/dist/*.whl /web/zengl.whl
COPY examples/webgl/public/* /web/
CMD python -m http.server --bind 0.0.0.0 8000
