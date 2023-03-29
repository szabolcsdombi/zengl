FROM python:3.10.2
RUN python3 -m pip install pyodide-build
RUN git clone https://github.com/emscripten-core/emsdk.git &&\
    /emsdk/emsdk install `pyodide config get emscripten_version` &&\
    /emsdk/emsdk activate `pyodide config get emscripten_version`

RUN wget -q https://f003.backblazeb2.com/file/zengl-data/examples/web-example-assets.tar.gz

COPY _zengl.py zengl.c setup.py pyproject.toml MANIFEST.in README.md LICENSE /zengl/
RUN echo 'source /emsdk/emsdk_env.sh && ZENGL_NO_STUBS=yes pyodide build /zengl/ --exports PyInit_zengl' | bash

COPY examples/webgl/assets /assets/
RUN tar -zxf web-example-assets.tar.gz -C /assets/assets/
RUN echo 'source /emsdk/emsdk_env.sh && pyodide build /assets/' | bash

COPY examples/webgl/webgl /webgl/
RUN echo 'source /emsdk/emsdk_env.sh && pyodide build /webgl/' | bash

WORKDIR /web/
RUN cp /zengl/dist/zengl-1.12.0-cp310-cp310-emscripten_3_1_27_wasm32.whl /web/
RUN cp /webgl/dist/webgl-0.1.0-cp310-cp310-emscripten_3_1_27_wasm32.whl /web/
RUN cp /assets/dist/assets-0.1.0-py3-none-any.whl /web/
COPY examples/webgl/public/* /web/
CMD python -m http.server --bind 0.0.0.0 8000
