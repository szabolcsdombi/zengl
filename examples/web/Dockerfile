FROM python:3.11.3
ENV EMSDK=/opt/emsdk EMSDK_NODE=/opt/emsdk/node/16.20.0_64bit/bin/node \
    PATH=/opt/emsdk:/opt/emsdk/upstream/emscripten:/opt/emsdk/node/16.20.0_64bit/bin:$PATH
RUN git clone https://github.com/emscripten-core/emsdk.git $EMSDK &&\
    emsdk install 3.1.45 && emsdk activate 3.1.45 && pip install pyodide-build==0.24.1 &&\
    python -c "from pyodide_build.build_env import init_environment; init_environment()"

COPY setup.py MANIFEST.in README.md zengl_web.js zengl_web.c _zengl.py /zengl/
RUN pyodide build /zengl -o /web/

COPY examples/*.py /examples/
RUN touch /examples/glwindow.py
RUN cd /examples/ && tar -zcf /web/examples.tar.gz *

COPY examples/web/public /web
CMD python -m http.server -d web
