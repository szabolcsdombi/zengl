const zenglCanvasSetup = (pyodide, canvas) => {
  canvas.addEventListener('mousemove', (evt) => {
    state.mx = evt.x;
    state.my = evt.y;
  });

  canvas.addEventListener('keydown', (evt) => {
    state.keys.add(evt.key);
  });

  canvas.addEventListener('keyup', (evt) => {
    state.keys.delete(evt.key);
  });

  const state = {
    width: canvas.width,
    height: canvas.height,
    prevMx: 0,
    prevMy: 0,
    prevKeys: new Set(),
    mx: 0,
    my: 0,
    keys: new Set(),
    timeRef: null,
    timeElapsed: 0,
    frame: 0,
  };

  const render = (timestamp) => {
    state.timeElapsed = (timestamp - state.timeRef) * 1e-3;

    const callback = pyodide.globals.get('render');
    if (callback !== undefined) {
      callback();
    }

    state.prevMx = state.mx;
    state.prevMy = state.my;
    state.prevKeys = state.keys;
    state.keys = new Set(state.keys);
    state.frame += 1;

    requestAnimationFrame(render);
  };

  requestAnimationFrame((timestamp) => {
    state.timeRef = timestamp;
    requestAnimationFrame(render);
  });

  const wasm = pyodide._module;

  const textEncoder = new TextEncoder('utf-8');
  const textDecoder = new TextDecoder('utf-8');

  const getString = (ptr) => {
    const length = wasm.HEAPU8.subarray(ptr).findIndex((c) => c === 0);
    return textDecoder.decode(wasm.HEAPU8.subarray(ptr, ptr + length));
  };

  pyodide._module.mergeLibSymbols({
    zengl_get_frame() {
      return state.frame;
    },
    zengl_get_time() {
      return state.timeElapsed;
    },
    zengl_get_mouse(mouse) {
      wasm.HEAP32[mouse / 4] = state.mx;
      wasm.HEAP32[mouse / 4 + 1] = state.my;
    },
    zengl_get_mouse_delta(mouse) {
      wasm.HEAP32[mouse / 4] = state.mx - state.prevMx;
      wasm.HEAP32[mouse / 4 + 1] = state.my - state.prevMy;
    },
    zengl_get_size(size) {
      wasm.HEAP32[size / 4] = state.width;
      wasm.HEAP32[size / 4 + 1] = state.height;
    },
    zengl_get_key(key) {
      const keyName = getString(key);
      return (state.keys.has(keyName) ? 1 : 0) | (state.prevKeys.has(keyName) ? 2 : 0);
    },
  });
};
