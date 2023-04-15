FROM python:3.11.2
ENV EMSDK=/opt/emsdk EMSDK_NODE=/opt/emsdk/node/14.18.2_64bit/bin/node \
    PATH=/opt/emsdk:/opt/emsdk/upstream/emscripten:/opt/emsdk/node/14.18.2_64bit/bin:$PATH
RUN git clone https://github.com/emscripten-core/emsdk.git $EMSDK &&\
    emsdk install 3.1.32 && emsdk activate 3.1.32 && pip install pyodide-build==0.23.1

RUN wget -q https://f003.backblazeb2.com/file/zengl-data/examples/web-example-assets.tar.gz

COPY _zengl.py zengl.c setup.py pyproject.toml MANIFEST.in README.md LICENSE /zengl/
RUN ZENGL_NO_STUBS=yes pyodide build /zengl/ --exports PyInit_zengl

COPY examples/webgl/assets /assets/
RUN tar -zxf web-example-assets.tar.gz -C /assets/assets/
RUN pyodide build /assets/

COPY examples/webgl/webgl /webgl/
RUN pyodide build /webgl/

WORKDIR /web/
RUN cp /dist/*.whl /web/
COPY examples/webgl/public/* /web/
CMD python -m http.server --bind 0.0.0.0 8000
