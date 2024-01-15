(wasm, gl) => {
  const textEncoder = new TextEncoder('utf-8');
  const textDecoder = new TextDecoder('utf-8');

  const getString = (ptr) => {
    const length = wasm.HEAPU8.subarray(ptr).findIndex((c) => c === 0);
    return textDecoder.decode(wasm.HEAPU8.subarray(ptr, ptr + length));
  };

  const setString = (ptr, text) => {
    const raw = textEncoder.encode(text);
    wasm.HEAPU8.set(raw, ptr);
    wasm.HEAPU8[raw.byteLength] = 0;
  };

  const typedArray = (type, ptr, size) => {
    switch (type) {
      case 0x1400: return wasm.HEAP8.subarray(ptr, ptr + size);
      case 0x1401: return wasm.HEAPU8.subarray(ptr, ptr + size);
      case 0x1402: return wasm.HEAP16.subarray(ptr >> 1, (ptr >> 1) + (size << 1));
      case 0x1403: return wasm.HEAPU16.subarray(ptr >> 1, (ptr >> 1) + (size << 1));
      case 0x1404: return wasm.HEAP32.subarray(ptr >> 2, (ptr >> 2) + (size << 2));
      case 0x1405: return wasm.HEAPU32.subarray(ptr >> 2, (ptr >> 2) + (size << 2));
      case 0x1406: return wasm.HEAPF32.subarray(ptr >> 2, (ptr >> 2) + (size << 2));
      case 0x84FA: return wasm.HEAPU32.subarray(ptr >> 2, (ptr >> 2) + (size << 2));
    };
  };

  const componentCount = (format) => {
    switch (format) {
      case 0x1903: return 1;
      case 0x8227: return 2;
      case 0x1908: return 4;
      case 0x8D94: return 1;
      case 0x8228: return 2;
      case 0x1902: return 1;
      case 0x84F9: return 1;
    }
  };

  const glo = new Map();
  let glid = 1;
  glo[0] = null;

  return {
    zengl_glCullFace(mode) {
      gl.cullFace(mode);
    },
    zengl_glClear(mask) {
      gl.clear(mask);
    },
    zengl_glTexParameteri(target, pname, param) {
      gl.texParameteri(target, pname, param);
    },
    zengl_glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels) {
      gl.texImage2D(target, level, internalformat, width, height, border, format, type, null);
    },
    zengl_glDepthMask(flag) {
      gl.depthMask(flag);
    },
    zengl_glDisable(cap) {
      if (cap !== 0x8D69 && cap !== 0x8642 && cap !== 0x884F && cap !== 0x8DB9) {
        gl.disable(cap);
      }
    },
    zengl_glEnable(cap) {
      if (cap !== 0x8D69 && cap !== 0x8642 && cap !== 0x884F && cap !== 0x8DB9) {
        gl.enable(cap);
      }
    },
    zengl_glFlush() {
      gl.flush();
    },
    zengl_glDepthFunc(func) {
      gl.depthFunc(func);
    },
    zengl_glReadBuffer(src) {
      gl.readBuffer(src);
    },
    zengl_glReadPixels(x, y, width, height, format, type, pixels) {
      const data = typedArray(type, pixels, width * height * componentCount(format));
      gl.readPixels(x, y, width, height, format, type, data);
    },
    zengl_glGetError() {
      return gl.getError();
    },
    zengl_glGetIntegerv(pname, data) {
      const value = gl.getParameter(pname);
      wasm.HEAP32[data >> 2] = Math.min(value, 0x7ffffff);
    },
    zengl_glGetString(pname) {
      return wasm.allocateUTF8(gl.getParameter(pname));
    },
    zengl_glViewport(x, y, width, height) {
      gl.viewport(x, y, width, height);
    },
    zengl_glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels) {
      const data = typedArray(type, pixels, width * height * componentCount(format));
      gl.texSubImage2D(target, level, xoffset, yoffset, width, height, format, type, data);
    },
    zengl_glBindTexture(target, texture) {
      gl.bindTexture(target, glo[texture]);
    },
    zengl_glDeleteTextures(n, textures) {
      const texture = wasm.HEAP32[textures >> 2];
      gl.deleteTexture(glo[texture]);
      glo.delete(texture);
    },
    zengl_glGenTextures(n, textures) {
      const texture = glid++;
      glo[texture] = gl.createTexture();
      wasm.HEAP32[textures >> 2] = texture;
    },
    zengl_glTexImage3D(target, level, internalformat, width, height, depth, border, format, type, pixels) {
      gl.texImage3D(target, level, internalformat, width, height, depth, border, format, type, null);
    },
    zengl_glTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels) {
      const data = typedArray(type, pixels, width * height * depth * componentCount(format));
      gl.texSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, data);
    },
    zengl_glActiveTexture(texture) {
      gl.activeTexture(texture);
    },
    zengl_glBlendFuncSeparate(sfactorRGB, dfactorRGB, sfactorAlpha, dfactorAlpha) {
      gl.blendFuncSeparate(sfactorRGB, dfactorRGB, sfactorAlpha, dfactorAlpha);
    },
    zengl_glGenQueries(n, ids) {
      gl.getExtension('EXT_disjoint_timer_query_webgl2');
      const query = glid++;
      glo[query] = gl.createQuery();
      wasm.HEAP32[ids >> 2] = query;
    },
    zengl_glBeginQuery(target, id) {
      gl.beginQuery(target, glo[id]);
    },
    zengl_glEndQuery(target) {
      gl.endQuery(target);
    },
    zengl_glGetQueryObjectuiv(id, pname, params) {
      wasm.HEAP32[params >> 2] = gl.getQueryParameter(glo[id], pname);
    },
    zengl_glBindBuffer(target, buffer) {
      gl.bindBuffer(target, glo[buffer]);
    },
    zengl_glDeleteBuffers(n, buffers) {
      const buffer = wasm.HEAP32[buffers >> 2];
      gl.deleteBuffer(glo[buffer]);
      glo.delete(buffer);
    },
    zengl_glGenBuffers(n, buffers) {
      const buffer = glid++;
      glo[buffer] = gl.createBuffer();
      wasm.HEAP32[buffers >> 2] = buffer;
    },
    zengl_glBufferData(target, size, data, usage) {
      gl.bufferData(target, size, usage);
    },
    zengl_glBufferSubData(target, offset, size, data) {
      gl.bufferSubData(target, offset, wasm.HEAPU8.subarray(data, data + size));
    },
    zengl_glGetBufferSubData(target, offset, size, data) {
      gl.getBufferSubData(target, offset, wasm.HEAPU8.subarray(data, data + size));
    },
    zengl_glBlendEquationSeparate(modeRGB, modeAlpha) {
      gl.blendEquationSeparate(modeRGB, modeAlpha);
    },
    zengl_glDrawBuffers(n, bufs) {
      gl.drawBuffers(wasm.HEAP32.subarray(bufs >> 2, (bufs >> 2) + n));
    },
    zengl_glStencilOpSeparate(face, sfail, dpfail, dppass) {
      gl.stencilOpSeparate(face, sfail, dpfail, dppass);
    },
    zengl_glStencilFuncSeparate(face, func, ref, mask) {
      gl.stencilFuncSeparate(face, func, ref, mask);
    },
    zengl_glStencilMaskSeparate(face, mask) {
      gl.stencilMaskSeparate(face, mask);
    },
    zengl_glAttachShader(program, shader) {
      gl.attachShader(glo[program], glo[shader]);
    },
    zengl_glCompileShader(shader) {
      gl.compileShader(glo[shader]);
    },
    zengl_glCreateProgram() {
      const program = glid++;
      glo[program] = gl.createProgram();
      return program;
    },
    zengl_glCreateShader(type) {
      const shader = glid++;
      glo[shader] = gl.createShader(type);
      return shader;
    },
    zengl_glDeleteProgram(program) {
      gl.deleteProgram(glo[program]);
      glo.delete(program);
    },
    zengl_glDeleteShader(shader) {
      gl.deleteShader(glo[shader]);
      glo.delete(shader);
    },
    zengl_glEnableVertexAttribArray(index) {
      gl.enableVertexAttribArray(index);
    },
    zengl_glGetActiveAttrib(program, index, bufSize, length, size, type, name) {
      const info = gl.getActiveAttrib(glo[program], index);
      setString(name, info.name);
      wasm.HEAP32[size >> 2] = info.size;
      wasm.HEAP32[type >> 2] = info.type;
    },
    zengl_glGetActiveUniform(program, index, bufSize, length, size, type, name) {
      const info = gl.getActiveUniform(glo[program], index);
      setString(name, info.name);
      wasm.HEAP32[size >> 2] = info.size;
      wasm.HEAP32[type >> 2] = info.type;
    },
    zengl_glGetAttribLocation(program, name) {
      return gl.getAttribLocation(glo[program], getString(name));
    },
    zengl_glGetProgramiv(program, pname, params) {
      if (pname === 0x8B84) {
        wasm.HEAP32[params >> 2] = gl.getProgramInfoLog(glo[program]).length + 1;
      } else {
        wasm.HEAP32[params >> 2] = gl.getProgramParameter(glo[program], pname);
      }
    },
    zengl_glGetProgramInfoLog(program, bufSize, length, infoLog) {
      setString(infoLog, gl.getProgramInfoLog(glo[program]));
    },
    zengl_glGetShaderiv(shader, pname, params) {
      if (pname === 0x8B84) {
        wasm.HEAP32[params >> 2] = gl.getShaderInfoLog(glo[shader]).length + 1;
      } else {
        wasm.HEAP32[params >> 2] = gl.getShaderParameter(glo[shader], pname);
      }
    },
    zengl_glGetShaderInfoLog(shader, bufSize, length, infoLog) {
      setString(infoLog, gl.getShaderInfoLog(glo[shader]));
    },
    zengl_glGetUniformLocation(program, name) {
      const uniform = gl.getUniformLocation(glo[program], getString(name));
      if (uniform !== null) {
        const location = glid++;
        glo[location] = uniform;
        return location;
      }
      return -1;
    },
    zengl_glLinkProgram(program) {
      gl.linkProgram(glo[program]);
    },
    zengl_glShaderSource(shader, count, string, length) {
      gl.shaderSource(glo[shader], getString(wasm.HEAP32[string >> 2]));
    },
    zengl_glUseProgram(program) {
      gl.useProgram(glo[program]);
    },
    zengl_glUniform1i(location, v0) {
      gl.uniform1i(glo[location], v0);
    },
    zengl_glUniform1fv(location, count, value) {
      gl.uniform1fv(glo[location], wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count));
    },
    zengl_glUniform2fv(location, count, value) {
      gl.uniform2fv(glo[location], wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 2));
    },
    zengl_glUniform3fv(location, count, value) {
      gl.uniform3fv(glo[location], wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 3));
    },
    zengl_glUniform4fv(location, count, value) {
      gl.uniform4fv(glo[location], wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 4));
    },
    zengl_glUniform1iv(location, count, value) {
      gl.uniform1iv(glo[location], wasm.HEAP32.subarray(value >> 2, (value >> 2) + count));
    },
    zengl_glUniform2iv(location, count, value) {
      gl.uniform2iv(glo[location], wasm.HEAP32.subarray(value >> 2, (value >> 2) + count * 2));
    },
    zengl_glUniform3iv(location, count, value) {
      gl.uniform3iv(glo[location], wasm.HEAP32.subarray(value >> 2, (value >> 2) + count * 3));
    },
    zengl_glUniform4iv(location, count, value) {
      gl.uniform4iv(glo[location], wasm.HEAP32.subarray(value >> 2, (value >> 2) + count * 4));
    },
    zengl_glUniformMatrix2fv(location, count, transpose, value) {
      gl.uniformMatrix2fv(glo[location], transpose, wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 4));
    },
    zengl_glUniformMatrix3fv(location, count, transpose, value) {
      gl.uniformMatrix3fv(glo[location], transpose, wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 9));
    },
    zengl_glUniformMatrix4fv(location, count, transpose, value) {
      gl.uniformMatrix4fv(glo[location], transpose, wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 16));
    },
    zengl_glVertexAttribPointer(index, size, type, normalized, stride, pointer) {
      gl.vertexAttribPointer(index, size, type, normalized, stride, pointer);
    },
    zengl_glUniformMatrix2x3fv(location, count, transpose, value) {
      gl.uniformMatrix2x3fv(glo[location], transpose, wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 6));
    },
    zengl_glUniformMatrix3x2fv(location, count, transpose, value) {
      gl.uniformMatrix3x2fv(glo[location], transpose, wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 6));
    },
    zengl_glUniformMatrix2x4fv(location, count, transpose, value) {
      gl.uniformMatrix2x4fv(glo[location], transpose, wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 8));
    },
    zengl_glUniformMatrix4x2fv(location, count, transpose, value) {
      gl.uniformMatrix4x2fv(glo[location], transpose, wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 8));
    },
    zengl_glUniformMatrix3x4fv(location, count, transpose, value) {
      gl.uniformMatrix3x4fv(glo[location], transpose, wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 12));
    },
    zengl_glUniformMatrix4x3fv(location, count, transpose, value) {
      gl.uniformMatrix4x3fv(glo[location], transpose, wasm.HEAPF32.subarray(value >> 2, (value >> 2) + count * 12));
    },
    zengl_glBindBufferRange(target, index, buffer, offset, size) {
      gl.bindBufferRange(target, index, glo[buffer], offset, size);
    },
    zengl_glVertexAttribIPointer(index, size, type, stride, pointer) {
      gl.vertexAttribIPointer(index, size, type, stride, pointer);
    },
    zengl_glUniform1uiv(location, count, value) {
      gl.uniform1uiv(glo[location], wasm.HEAPU32.subarray(value >> 2, (value >> 2) + count));
    },
    zengl_glUniform2uiv(location, count, value) {
      gl.uniform2uiv(glo[location], wasm.HEAPU32.subarray(value >> 2, (value >> 2) + count * 2));
    },
    zengl_glUniform3uiv(location, count, value) {
      gl.uniform3uiv(glo[location], wasm.HEAPU32.subarray(value >> 2, (value >> 2) + count * 3));
    },
    zengl_glUniform4uiv(location, count, value) {
      gl.uniform4uiv(glo[location], wasm.HEAPU32.subarray(value >> 2, (value >> 2) + count * 4));
    },
    zengl_glClearBufferiv(buffer, drawbuffer, value) {
      gl.clearBufferiv(buffer, drawbuffer, wasm.HEAP32.subarray(value >> 2, (value >> 2) + 4));
    },
    zengl_glClearBufferuiv(buffer, drawbuffer, value) {
      gl.clearBufferuiv(buffer, drawbuffer, wasm.HEAPU32.subarray(value >> 2, (value >> 2) + 4));
    },
    zengl_glClearBufferfv(buffer, drawbuffer, value) {
      gl.clearBufferfv(buffer, drawbuffer, wasm.HEAPF32.subarray(value >> 2, (value >> 2) + 4));
    },
    zengl_glClearBufferfi(buffer, drawbuffer, depth, stencil) {
      gl.clearBufferfi(buffer, drawbuffer, depth, stencil);
    },
    zengl_glBindRenderbuffer(target, renderbuffer) {
      gl.bindRenderbuffer(target, glo[renderbuffer]);
    },
    zengl_glDeleteRenderbuffers(n, renderbuffers) {
      const renderbuffer = wasm.HEAP32[renderbuffers >> 2];
      gl.deleteRenderbuffer(glo[renderbuffer]);
      glo.delete(renderbuffer);
    },
    zengl_glGenRenderbuffers(n, renderbuffers) {
      const renderbuffer = glid++;
      glo[renderbuffer] = gl.createRenderbuffer();
      wasm.HEAP32[renderbuffers >> 2] = renderbuffer;
    },
    zengl_glBindFramebuffer(target, framebuffer) {
      gl.bindFramebuffer(target, glo[framebuffer]);
    },
    zengl_glDeleteFramebuffers(n, framebuffers) {
      const framebuffer = wasm.HEAP32[framebuffers >> 2];
      gl.deleteFramebuffer(glo[framebuffer]);
      glo.delete(framebuffer);
    },
    zengl_glGenFramebuffers(n, framebuffers) {
      const framebuffer = glid++;
      glo[framebuffer] = gl.createFramebuffer();
      wasm.HEAP32[framebuffers >> 2] = framebuffer;
    },
    zengl_glFramebufferTexture2D(target, attachment, textarget, texture, level) {
      gl.framebufferTexture2D(target, attachment, textarget, glo[texture], level);
    },
    zengl_glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer) {
      gl.framebufferRenderbuffer(target, attachment, renderbuffertarget, glo[renderbuffer]);
    },
    zengl_glGenerateMipmap(target) {
      gl.generateMipmap(target);
    },
    zengl_glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter) {
      gl.blitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
    },
    zengl_glRenderbufferStorageMultisample(target, samples, internalformat, width, height) {
      gl.renderbufferStorageMultisample(target, samples, internalformat, width, height);
    },
    zengl_glFramebufferTextureLayer(target, attachment, texture, level, layer) {
      gl.framebufferTextureLayer(target, attachment, glo[texture], level, layer);
    },
    zengl_glBindVertexArray(array) {
      gl.bindVertexArray(glo[array]);
    },
    zengl_glDeleteVertexArrays(n, arrays) {
      const array = wasm.HEAP32[arrays >> 2];
      gl.deleteVertexArray(glo[array]);
      glo.delete(array);
    },
    zengl_glGenVertexArrays(n, arrays) {
      const array = glid++;
      glo[array] = gl.createVertexArray();
      wasm.HEAP32[arrays >> 2] = array;
    },
    zengl_glDrawArraysInstanced(mode, first, count, instancecount) {
      gl.drawArraysInstanced(mode, first, count, instancecount);
    },
    zengl_glDrawElementsInstanced(mode, count, type, indices, instancecount) {
      gl.drawElementsInstanced(mode, count, type, indices, instancecount);
    },
    zengl_glCopyBufferSubData(readTarget, writeTarget, readOffset, writeOffset, size) {
      gl.copyBufferSubData(readTarget, writeTarget, readOffset, writeOffset, size);
    },
    zengl_glGetUniformBlockIndex(program, uniformBlockName) {
      return gl.getUniformBlockIndex(glo[program], getString(uniformBlockName));
    },
    zengl_glGetActiveUniformBlockiv(program, uniformBlockIndex, pname, params) {
      wasm.HEAP32[params >> 2] = gl.getActiveUniformBlockParameter(glo[program], uniformBlockIndex, pname);
    },
    zengl_glGetActiveUniformBlockName(program, uniformBlockIndex, bufSize, length, uniformBlockName) {
      setString(uniformBlockName, gl.getActiveUniformBlockName(glo[program], uniformBlockIndex));
    },
    zengl_glUniformBlockBinding(program, uniformBlockIndex, uniformBlockBinding) {
      gl.uniformBlockBinding(glo[program], uniformBlockIndex, uniformBlockBinding);
    },
    zengl_glFenceSync(condition, flags) {
      const sync = glid++;
      glo[sync] = gl.fenceSync(condition, flags);
      return sync;
    },
    zengl_glDeleteSync(sync) {
      gl.deleteSync(glo[sync]);
      glo.delete(sync);
    },
    zengl_glClientWaitSync(sync, flags, timeout) {
      gl.clientWaitSync(glo[sync], flags, gl.MAX_CLIENT_WAIT_TIMEOUT_WEBGL);
    },
    zengl_glGenSamplers(count, samplers) {
      const sampler = glid++;
      glo[sampler] = gl.createSampler();
      wasm.HEAP32[samplers >> 2] = sampler;
    },
    zengl_glDeleteSamplers(count, samplers) {
      const sampler = wasm.HEAP32[samplers >> 2];
      gl.deleteSampler(glo[sampler]);
      glo.delete(sampler);
    },
    zengl_glBindSampler(unit, sampler) {
      gl.bindSampler(unit, glo[sampler]);
    },
    zengl_glSamplerParameteri(sampler, pname, param) {
      gl.samplerParameteri(glo[sampler], pname, param);
    },
    zengl_glSamplerParameterf(sampler, pname, param) {
      gl.samplerParameterf(glo[sampler], pname, param);
    },
    zengl_glVertexAttribDivisor(index, divisor) {
      gl.vertexAttribDivisor(index, divisor);
    },
  };
}
