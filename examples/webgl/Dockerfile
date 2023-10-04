FROM python:3.11.3
ENV EMSDK=/opt/emsdk EMSDK_NODE=/opt/emsdk/node/16.20.0_64bit/bin/node \
    PATH=/opt/emsdk:/opt/emsdk/upstream/emscripten:/opt/emsdk/node/16.20.0_64bit/bin:$PATH
RUN git clone https://github.com/emscripten-core/emsdk.git $EMSDK &&\
    emsdk install 3.1.45 && emsdk activate 3.1.45 && pip install pyodide-build==0.24.1

RUN wget -q https://f003.backblazeb2.com/file/zengl-data/examples/web-example-assets.tar.gz

COPY _zengl.py zengl.c setup.py pyproject.toml MANIFEST.in README.md LICENSE /zengl/
RUN ZENGL_NO_STUBS=yes pyodide build /zengl/ -o /dist/

COPY examples/webgl/assets /assets/
RUN tar -zxf web-example-assets.tar.gz -C /assets/assets/
RUN pyodide build /assets/ -o /dist/

COPY examples/webgl/zengl_webgl /zengl_webgl/
RUN pyodide build /zengl_webgl/ -o /dist/

COPY examples/webgl/zengl_canvas /zengl_canvas/
RUN pyodide build /zengl_canvas/ -o /dist/

WORKDIR /web/
RUN cp /dist/*.whl /web/
COPY examples/webgl/zengl_webgl/zengl_webgl.js examples/webgl/zengl_canvas/zengl_canvas.js /web/
COPY examples/webgl/public/* /web/
CMD python -m http.server --bind 0.0.0.0 8000
