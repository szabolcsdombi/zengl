const runPython = async (code) => {
  const pyodide = await loadPyodide();
  await pyodide.loadPackage([
    'zengl-1.16.0-cp311-cp311-emscripten_3_1_45_wasm32.whl',
  ], { messageCallback() {} });
  const response = await fetch('examples.tar.gz');
  const buffer = await response.arrayBuffer();
  await pyodide.unpackArchive(buffer, 'gztar');
  pyodide.runPython(code);
};
