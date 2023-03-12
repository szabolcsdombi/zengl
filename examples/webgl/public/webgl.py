import ctypes
from itertools import count

import js
from pyodide.ffi import create_proxy


def webgl(gl):
    glstr = {}
    glid = iter(count(1))
    glo = {0: None}

    void = None
    proc = ctypes.CFUNCTYPE
    GLboolean = ctypes.c_ubyte
    GLenum = ctypes.c_uint32
    GLbitfield = ctypes.c_uint32
    GLuint = ctypes.c_uint32
    GLint = ctypes.c_int32
    GLfloat = ctypes.c_float
    GLsizei = ctypes.c_int32
    GLuint64 = ctypes.c_uint64
    void_star = ctypes.c_void_p
    GLsync = ctypes.c_void_p
    GLubyte_star = ctypes.c_void_p
    GLint_star = ctypes.POINTER(ctypes.c_int32)
    GLsizei_star = ctypes.POINTER(ctypes.c_int32)
    GLuint_star = ctypes.POINTER(ctypes.c_uint32)
    GLenum_star = ctypes.POINTER(ctypes.c_uint32)
    GLfloat_star = ctypes.POINTER(ctypes.c_float)
    GLchar_star_star = ctypes.POINTER(ctypes.c_char_p)
    GLchar_star = ctypes.POINTER(ctypes.c_ubyte)
    GLintptr = ctypes.c_uint32
    GLsizeiptr = ctypes.c_uint32

    def addr(func):
        return ctypes.cast(func, ctypes.c_void_p).value, func

    def glCullFace(mode: GLenum):
        gl.cullFace(mode)

    def glClear(mask: GLbitfield):
        gl.clear(mask)

    def glTexParameteri(target: GLenum, pname: GLenum, param: GLint):
        gl.texParameteri(target, pname, param)

    def glTexImage2D(target: GLenum, level: GLint, internalformat: GLint, width: GLsizei, height: GLsizei, border: GLint, format: GLenum, type: GLenum, pixels: void_star):
        size = width * height * 4
        if pixels:
            data = js.Uint8Array.new(ctypes.cast(pixels, ctypes.POINTER(ctypes.c_ubyte * size)).contents)
        else:
            data = js.Uint8Array.new(bytearray(size))
        gl.texImage2D(target, level, internalformat, width, height, border, format, type, data)

    def glDepthMask(flag: GLboolean):
        gl.depthMask(flag)

    def glDisable(cap: GLenum):
        if cap not in (0x8D69, 0x8642, 0x884F, 0x8DB9):
            gl.disable(cap)

    def glEnable(cap: GLenum):
        if cap not in (0x8D69, 0x8642, 0x884F, 0x8DB9):
            gl.enable(cap)

    def glFlush():
        gl.flush()

    def glDepthFunc(func: GLenum):
        gl.depthFunc(func)

    def glReadBuffer(src: GLenum):
        gl.readBuffer(src)

    def glReadPixels(x: GLint, y: GLint, width: GLsizei, height: GLsizei, format: GLenum, type: GLenum, pixels: void_star):
        # return gl['glReadPixels'](x, y, width, height, format, type, pixels)
        raise NotImplementedError('glReadPixels')

    def glGetError():
        return gl.getError()

    def glGetIntegerv(pname: GLenum, data: GLint_star):
        data[0] = gl.getParameter(pname, data)

    def glGetString(name: GLenum):
        glstr[name] = gl.getParameter(name).encode()
        return ctypes.cast(ctypes.c_char_p(glstr[name]), ctypes.c_void_p).value

    def glViewport(x: GLint, y: GLint, width: GLsizei, height: GLsizei):
        gl.viewport(x, y, width, height)

    def glTexSubImage2D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, width: GLsizei, height: GLsizei, format: GLenum, type: GLenum, pixels: void_star):
        size = width * height * 4
        if pixels:
            data = js.Uint8Array.new(ctypes.cast(pixels, ctypes.POINTER(ctypes.c_ubyte * size)).contents)
        else:
            data = js.Uint8Array.new(bytearray(size))
        gl.texSubImage2D(target, level, xoffset, yoffset, width, height, format, type, data)

    def glBindTexture(target: GLenum, texture: GLuint):
        gl.bindTexture(target, glo[texture])

    def glDeleteTextures(n: GLsizei, textures: GLuint_star):
        gl.deleteTexture(glo[textures[0]])
        del glo[textures[0]]

    def glGenTextures(n: GLsizei, textures: GLuint_star):
        textures[0] = next(glid)
        glo[textures[0]] = gl.createTexture()

    def glTexImage3D(target: GLenum, level: GLint, internalformat: GLint, width: GLsizei, height: GLsizei, depth: GLsizei, border: GLint, format: GLenum, type: GLenum, pixels: void_star):
        size = width * height * depth * 4
        if pixels:
            data = js.Uint8Array.new(ctypes.cast(pixels, ctypes.POINTER(ctypes.c_ubyte * size)).contents)
        else:
            data = js.Uint8Array.new(bytearray(size))
        gl.texImage3D(target, level, internalformat, width, height, depth, border, format, type, data)

    def glTexSubImage3D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, zoffset: GLint, width: GLsizei, height: GLsizei, depth: GLsizei, format: GLenum, type: GLenum, pixels: void_star):
        size = width * height * depth * 4
        if pixels:
            data = js.Uint8Array.new(ctypes.cast(pixels, ctypes.POINTER(ctypes.c_ubyte * size)).contents)
        else:
            data = js.Uint8Array.new(bytearray(size))
        gl.texSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, data)

    def glActiveTexture(texture: GLenum):
        gl.activeTexture(texture)

    def glBlendFuncSeparate(sfactorRGB: GLenum, dfactorRGB: GLenum, sfactorAlpha: GLenum, dfactorAlpha: GLenum):
        gl.blendFuncSeparate(sfactorRGB, dfactorRGB, sfactorAlpha, dfactorAlpha)

    def glGenQueries(n: GLsizei, ids: GLuint_star):
        ids[0] = next(glid)
        glo[ids[0]] = gl.createQuery()
        gl.getExtension('EXT_disjoint_timer_query')

    def glBeginQuery(target: GLenum, id: GLuint):
        pass

    def glEndQuery(target: GLenum):
        pass

    def glGetQueryObjectuiv(id: GLuint, pname: GLenum, params: GLuint_star):
        params[0] = 0

    def glBindBuffer(target: GLenum, buffer: GLuint):
        gl.bindBuffer(target, glo[buffer])

    def glDeleteBuffers(n: GLsizei, buffers: GLuint_star):
        gl.deleteBuffer(glo[buffers[0]])
        del glo[buffers[0]]

    def glGenBuffers(n: GLsizei, buffers: GLuint_star):
        buffers[0] = next(glid)
        glo[buffers[0]] = gl.createBuffer()

    def glBufferData(target: GLenum, size: GLsizeiptr, data: void_star, usage: GLenum):
        if data:
            content = js.Uint8Array.new(ctypes.cast(data, ctypes.POINTER(ctypes.c_ubyte * size)).contents)
        else:
            content = js.Uint8Array.new(bytearray(size))
        gl.bufferData(target, content, usage)

    def glBufferSubData(target: GLenum, offset: GLintptr, size: GLsizeiptr, data: void_star):
        content = js.Uint8Array.new(ctypes.cast(data, ctypes.POINTER(ctypes.c_ubyte * size)).contents)
        gl.bufferSubData(target, offset, content)

    def glUnmapBuffer(target: GLenum):
        return 0

    def glBlendEquationSeparate(modeRGB: GLenum, modeAlpha: GLenum):
        gl.blendEquationSeparate(modeRGB, modeAlpha)

    def glDrawBuffers(n: GLsizei, bufs: GLenum_star):
        gl.drawBuffers([bufs[i] for i in range(n)])

    def glStencilOpSeparate(face: GLenum, sfail: GLenum, dpfail: GLenum, dppass: GLenum):
        gl.stencilOpSeparate(face, sfail, dpfail, dppass)

    def glStencilFuncSeparate(face: GLenum, func: GLenum, ref: GLint, mask: GLuint):
        gl.stencilFuncSeparate(face, func, ref, mask)

    def glStencilMaskSeparate(face: GLenum, mask: GLuint):
        gl.stencilMaskSeparate(face, mask)

    def glAttachShader(program: GLuint, shader: GLuint):
        gl.attachShader(glo[program], glo[shader])

    def glCompileShader(shader: GLuint):
        gl.compileShader(glo[shader])

    def glCreateProgram():
        res = next(glid)
        glo[res] = gl.createProgram()
        return res

    def glCreateShader(type: GLenum):
        res = next(glid)
        glo[res] = gl.createShader(type)
        return res

    def glDeleteProgram(program: GLuint):
        gl.deleteProgram(glo[program])
        del glo[program]

    def glDeleteShader(shader: GLuint):
        gl.deleteShader(glo[shader])
        del glo[shader]

    def glEnableVertexAttribArray(index: GLuint):
        gl.enableVertexAttribArray(index)

    def glGetActiveAttrib(program: GLuint, index: GLuint, bufSize: GLsizei, length: GLsizei_star, size: GLint_star, type: GLenum_star, name: GLchar_star):
        info = gl.getActiveAttrib(glo[program], index)
        name_raw = info.name.encode()
        name_ptr = ctypes.cast(name, ctypes.POINTER(ctypes.c_ubyte))
        for i in range(len(name_raw)):
            name_ptr[i] = name_raw[i]
        name_ptr[len(name_raw)] = 0
        length[0] = len(name_raw)
        size[0] = info.size
        type[0] = info.type

    def glGetActiveUniform(program: GLuint, index: GLuint, bufSize: GLsizei, length: GLsizei_star, size: GLint_star, type: GLenum_star, name: GLchar_star):
        info = gl.getActiveUniform(glo[program], index)
        name_raw = info.name.encode()
        name_ptr = ctypes.cast(name, ctypes.POINTER(ctypes.c_ubyte))
        for i in range(len(name_raw)):
            name_ptr[i] = name_raw[i]
        name_ptr[len(name_raw)] = 0
        length[0] = len(name_raw)
        size[0] = info.size
        type[0] = info.type

    def glGetAttribLocation(program: GLuint, name: GLchar_star):
        return gl.getAttribLocation(glo[program], ctypes.cast(name, ctypes.c_char_p).value.decode())

    def glGetProgramiv(program: GLuint, pname: GLenum, params: GLint_star):
        if pname == 0x8B84:
            params[0] = len(gl.getShaderInfoLog(glo[program]).encode())
        else:
            params[0] = gl.getProgramParameter(glo[program], pname)

    def glGetProgramInfoLog(program: GLuint, bufSize: GLsizei, length: GLsizei_star, infoLog: GLchar_star):
        msg = gl.getProgramInfoLog(glo[program]).encode()
        ptr = ctypes.cast(infoLog, ctypes.POINTER(ctypes.c_ubyte))
        for i in range(len(msg)):
            ptr[i] = msg[i]
        ptr[len(msg)] = 0
        length[0] = len(msg)

    def glGetShaderiv(shader: GLuint, pname: GLenum, params: GLint_star):
        if pname == 0x8B84:
            params[0] = len(gl.getShaderInfoLog(glo[shader]).encode())
        else:
            params[0] = gl.getShaderParameter(glo[shader], pname)

    def glGetShaderInfoLog(shader: GLuint, bufSize: GLsizei, length: GLsizei_star, infoLog: GLchar_star):
        msg = gl.getShaderInfoLog(glo[shader]).encode()
        ptr = ctypes.cast(infoLog, ctypes.POINTER(ctypes.c_ubyte))
        for i in range(len(msg)):
            ptr[i] = msg[i]
        ptr[len(msg)] = 0
        length[0] = len(msg)

    def glGetUniformLocation(program: GLuint, name: GLchar_star):
        location = next(glid)
        glo[location] = gl.getUniformLocation(glo[program], ctypes.cast(name, ctypes.c_char_p).value.decode())
        return location

    def glLinkProgram(program: GLuint):
        gl.linkProgram(glo[program])

    def glShaderSource(shader: GLuint, count: GLsizei, string: GLchar_star_star, length: GLint_star):
        gl.shaderSource(glo[shader], string[0].strip().decode())

    def glUseProgram(program: GLuint):
        gl.useProgram(glo[program])

    def glUniform1i(location: GLint, v0: GLint):
        gl.uniform1i(glo[location], v0)

    def glUniform1fv(location: GLint, count: GLsizei, value: GLfloat_star):
        gl.uniform1fv(glo[location], js.Float32Array.new([value[i] for i in range(count)]))

    def glUniform2fv(location: GLint, count: GLsizei, value: GLfloat_star):
        gl.uniform2fv(glo[location], js.Float32Array.new([value[i] for i in range(count * 2)]))

    def glUniform3fv(location: GLint, count: GLsizei, value: GLfloat_star):
        gl.uniform3fv(glo[location], js.Float32Array.new([value[i] for i in range(count * 3)]))

    def glUniform4fv(location: GLint, count: GLsizei, value: GLfloat_star):
        gl.uniform4fv(glo[location], js.Float32Array.new([value[i] for i in range(count * 4)]))

    def glUniform1iv(location: GLint, count: GLsizei, value: GLint_star):
        gl.uniform1iv(glo[location], js.Int32Array.new([value[i] for i in range(count)]))

    def glUniform2iv(location: GLint, count: GLsizei, value: GLint_star):
        gl.uniform2iv(glo[location], js.Int32Array.new([value[i] for i in range(count * 2)]))

    def glUniform3iv(location: GLint, count: GLsizei, value: GLint_star):
        gl.uniform3iv(glo[location], js.Int32Array.new([value[i] for i in range(count * 3)]))

    def glUniform4iv(location: GLint, count: GLsizei, value: GLint_star):
        gl.uniform4iv(glo[location], js.Int32Array.new([value[i] for i in range(count * 4)]))

    def glUniformMatrix2fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        gl.uniformMatrix2fv(glo[location], transpose, js.Float32Array.new([value[i] for i in range(count * 4)]))

    def glUniformMatrix3fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        gl.uniformMatrix3fv(glo[location], transpose, js.Float32Array.new([value[i] for i in range(count * 9)]))

    def glUniformMatrix4fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        gl.uniformMatrix4fv(glo[location], transpose, js.Float32Array.new([value[i] for i in range(count * 16)]))

    def glVertexAttribPointer(index: GLuint, size: GLint, type: GLenum, normalized: GLboolean, stride: GLsizei, pointer: void_star):
        gl.vertexAttribPointer(index, size, type, normalized, stride, pointer)

    def glUniformMatrix2x3fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        gl.uniformMatrix2x3fv(glo[location], transpose, js.Float32Array.new([value[i] for i in range(count * 6)]))

    def glUniformMatrix3x2fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        gl.uniformMatrix3x2fv(glo[location], transpose, js.Float32Array.new([value[i] for i in range(count * 6)]))

    def glUniformMatrix2x4fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        gl.uniformMatrix2x4fv(glo[location], transpose, js.Float32Array.new([value[i] for i in range(count * 8)]))

    def glUniformMatrix4x2fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        gl.uniformMatrix4x2fv(glo[location], transpose, js.Float32Array.new([value[i] for i in range(count * 8)]))

    def glUniformMatrix3x4fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        gl.uniformMatrix3x4fv(glo[location], transpose, js.Float32Array.new([value[i] for i in range(count * 12)]))

    def glUniformMatrix4x3fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        gl.uniformMatrix4x3fv(glo[location], transpose, js.Float32Array.new([value[i] for i in range(count * 12)]))

    def glBindBufferRange(target: GLenum, index: GLuint, buffer: GLuint, offset: GLintptr, size: GLsizeiptr):
        gl.bindBufferRange(target, index, glo[buffer], offset, size)

    def glVertexAttribIPointer(index: GLuint, size: GLint, type: GLenum, stride: GLsizei, pointer: void_star):
        gl.vertexAttribIPointer(index, size, type, stride, pointer)

    def glUniform1uiv(location: GLint, count: GLsizei, value: GLuint_star):
        gl.uniform1uiv(glo[location], js.UInt32Array.new([value[i] for i in range(count)]))

    def glUniform2uiv(location: GLint, count: GLsizei, value: GLuint_star):
        gl.uniform2uiv(glo[location], js.UInt32Array.new([value[i] for i in range(count * 2)]))

    def glUniform3uiv(location: GLint, count: GLsizei, value: GLuint_star):
        gl.uniform3uiv(glo[location], js.UInt32Array.new([value[i] for i in range(count * 3)]))

    def glUniform4uiv(location: GLint, count: GLsizei, value: GLuint_star):
        gl.uniform4uiv(glo[location], js.UInt32Array.new([value[i] for i in range(count * 4)]))

    def glClearBufferiv(buffer: GLenum, drawbuffer: GLint, value: GLint_star):
        gl.clearBufferiv(buffer, drawbuffer, js.Int32Array.new([value[0], value[1], value[2], value[3]]))

    def glClearBufferuiv(buffer: GLenum, drawbuffer: GLint, value: GLuint_star):
        gl.clearBufferuiv(buffer, drawbuffer, js.UInt32Array.new([value[0], value[1], value[2], value[3]]))

    def glClearBufferfv(buffer: GLenum, drawbuffer: GLint, value: GLfloat_star):
        gl.clearBufferfv(buffer, drawbuffer, js.Float32Array.new([value[0], value[1], value[2], value[3]]))

    def glClearBufferfi(buffer: GLenum, drawbuffer: GLint, depth: GLfloat, stencil: GLint):
        gl.clearBufferfi(buffer, drawbuffer, depth, stencil)

    def glBindRenderbuffer(target: GLenum, renderbuffer: GLuint):
        gl.bindRenderbuffer(target, glo[renderbuffer])

    def glDeleteRenderbuffers(n: GLsizei, renderbuffers: GLuint_star):
        gl.deleteRenderbuffer(glo[renderbuffers[0]])
        del glo[renderbuffers[0]]

    def glGenRenderbuffers(n: GLsizei, renderbuffers: GLuint_star):
        renderbuffers[0] = next(glid)
        glo[renderbuffers[0]] = gl.createRenderbuffer()

    def glBindFramebuffer(target: GLenum, framebuffer: GLuint):
        gl.bindFramebuffer(target, glo[framebuffer])

    def glDeleteFramebuffers(n: GLsizei, framebuffers: GLuint_star):
        gl.deleteFramebuffer(glo[framebuffers[0]])
        del glo[framebuffers[0]]

    def glGenFramebuffers(n: GLsizei, framebuffers: GLuint_star):
        framebuffers[0] = next(glid)
        glo[framebuffers[0]] = gl.createFramebuffer()

    def glFramebufferTexture2D(target: GLenum, attachment: GLenum, textarget: GLenum, texture: GLuint, level: GLint):
        gl.framebufferTexture2D(target, attachment, textarget, glo[texture], level)

    def glFramebufferRenderbuffer(target: GLenum, attachment: GLenum, renderbuffertarget: GLenum, renderbuffer: GLuint):
        gl.framebufferRenderbuffer(target, attachment, renderbuffertarget, glo[renderbuffer])

    def glGenerateMipmap(target: GLenum):
        gl.generateMipmap(target)

    def glBlitFramebuffer(srcX0: GLint, srcY0: GLint, srcX1: GLint, srcY1: GLint, dstX0: GLint, dstY0: GLint, dstX1: GLint, dstY1: GLint, mask: GLbitfield, filter: GLenum):
        gl.blitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter)

    def glRenderbufferStorageMultisample(target: GLenum, samples: GLsizei, internalformat: GLenum, width: GLsizei, height: GLsizei):
        gl.renderbufferStorageMultisample(target, samples, internalformat, width, height)

    def glFramebufferTextureLayer(target: GLenum, attachment: GLenum, texture: GLuint, level: GLint, layer: GLint):
        gl.framebufferTextureLayer(target, attachment, texture, level, layer)

    def glMapBufferRange(target: GLenum, offset: GLintptr, length: GLsizeiptr, access: GLbitfield):
        return 0

    def glBindVertexArray(array: GLuint):
        gl.bindVertexArray(glo[array])

    def glDeleteVertexArrays(n: GLsizei, arrays: GLuint_star):
        gl.deleteVertexArray(glo[arrays[0]])
        del glo[arrays[0]]

    def glGenVertexArrays(n: GLsizei, arrays: GLuint_star):
        arrays[0] = next(glid)
        glo[arrays[0]] = gl.createVertexArray()

    def glDrawArraysInstanced(mode: GLenum, first: GLint, count: GLsizei, instancecount: GLsizei):
        gl.drawArraysInstanced(mode, first, count, instancecount)

    def glDrawElementsInstanced(mode: GLenum, count: GLsizei, type: GLenum, indices: void_star, instancecount: GLsizei):
        gl.drawElementsInstanced(mode, count, type, indices, instancecount)

    def glGetUniformBlockIndex(program: GLuint, uniformBlockName: GLchar_star):
        return gl.getUniformBlockIndex(glo[program], ctypes.cast(uniformBlockName, ctypes.c_char_p).value.decode())

    def glGetActiveUniformBlockiv(program: GLuint, uniformBlockIndex: GLuint, pname: GLenum, params: GLint_star):
        params[0] = gl.getActiveUniformBlockParameter(glo[program], uniformBlockIndex, pname)

    def glGetActiveUniformBlockName(program: GLuint, uniformBlockIndex: GLuint, bufSize: GLsizei, length: GLsizei_star, uniformBlockName: GLchar_star):
        name_raw = gl.getActiveUniformBlockName(glo[program], uniformBlockIndex).encode()
        name_ptr = ctypes.cast(uniformBlockName, ctypes.POINTER(ctypes.c_ubyte))
        for i in range(len(name_raw)):
            name_ptr[i] = name_raw[i]
        name_ptr[len(name_raw)] = 0
        length[0] = len(name_raw)

    def glUniformBlockBinding(program: GLuint, uniformBlockIndex: GLuint, uniformBlockBinding: GLuint):
        gl.uniformBlockBinding(glo[program], uniformBlockIndex, uniformBlockBinding)

    def glFenceSync(condition: GLenum, flags: GLbitfield):
        sync = next(glid)
        glo[sync] = gl.fenceSync(condition, flags)
        return sync

    def glDeleteSync(sync: GLsync):
        gl.deleteSync(glo[sync.value])
        del glo[sync.value]

    def glClientWaitSync(sync: GLsync, flags: GLbitfield, timeout: GLuint64):
        return gl.clientWaitSync(sync, flags, timeout)

    def glGenSamplers(count: GLsizei, samplers: GLuint_star):
        samplers[0] = next(glid)
        glo[samplers[0]] = gl.createSampler()

    def glDeleteSamplers(count: GLsizei, samplers: GLuint_star):
        gl.deleteSampler(glo[samplers[0]])
        del glo[samplers[0]]

    def glBindSampler(unit: GLuint, sampler: GLuint):
        gl.bindSampler(unit, glo[sampler])

    def glSamplerParameteri(sampler: GLuint, pname: GLenum, param: GLint):
        gl.samplerParameteri(sampler, pname, param)

    def glSamplerParameterf(sampler: GLuint, pname: GLenum, param: GLfloat):
        gl.samplerParameterf(sampler, pname, param)

    def glVertexAttribDivisor(index: GLuint, divisor: GLuint):
        gl.vertexAttribDivisor(index, divisor)

    return {
        'glCullFace': addr(proc(void, GLenum)(glCullFace)),
        'glClear': addr(proc(void, GLbitfield)(glClear)),
        'glTexParameteri': addr(proc(void, GLenum, GLenum, GLint)(glTexParameteri)),
        'glTexImage2D': addr(proc(void, GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, void_star)(glTexImage2D)),
        'glDepthMask': addr(proc(void, GLboolean)(glDepthMask)),
        'glDisable': addr(proc(void, GLenum)(glDisable)),
        'glEnable': addr(proc(void, GLenum)(glEnable)),
        'glFlush': addr(proc(void)(glFlush)),
        'glDepthFunc': addr(proc(void, GLenum)(glDepthFunc)),
        'glReadBuffer': addr(proc(void, GLenum)(glReadBuffer)),
        'glReadPixels': addr(proc(void, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void_star)(glReadPixels)),
        'glGetError': addr(proc(GLenum)(glGetError)),
        'glGetIntegerv': addr(proc(void, GLenum, GLint_star)(glGetIntegerv)),
        'glGetString': addr(proc(GLubyte_star, GLenum)(glGetString)),
        'glViewport': addr(proc(void, GLint, GLint, GLsizei, GLsizei)(glViewport)),
        'glTexSubImage2D': addr(proc(void, GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void_star)(glTexSubImage2D)),
        'glBindTexture': addr(proc(void, GLenum, GLuint)(glBindTexture)),
        'glDeleteTextures': addr(proc(void, GLsizei, GLuint_star)(glDeleteTextures)),
        'glGenTextures': addr(proc(void, GLsizei, GLuint_star)(glGenTextures)),
        'glTexImage3D': addr(proc(void, GLenum, GLint, GLint, GLsizei, GLsizei, GLsizei, GLint, GLenum, GLenum, void_star)(glTexImage3D)),
        'glTexSubImage3D': addr(proc(void, GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei, GLenum, GLenum, void_star)(glTexSubImage3D)),
        'glActiveTexture': addr(proc(void, GLenum)(glActiveTexture)),
        'glBlendFuncSeparate': addr(proc(void, GLenum, GLenum, GLenum, GLenum)(glBlendFuncSeparate)),
        'glGenQueries': addr(proc(void, GLsizei, GLuint_star)(glGenQueries)),
        'glBeginQuery': addr(proc(void, GLenum, GLuint)(glBeginQuery)),
        'glEndQuery': addr(proc(void, GLenum)(glEndQuery)),
        'glGetQueryObjectuiv': addr(proc(void, GLuint, GLenum, GLuint_star)(glGetQueryObjectuiv)),
        'glBindBuffer': addr(proc(void, GLenum, GLuint)(glBindBuffer)),
        'glDeleteBuffers': addr(proc(void, GLsizei, GLuint_star)(glDeleteBuffers)),
        'glGenBuffers': addr(proc(void, GLsizei, GLuint_star)(glGenBuffers)),
        'glBufferData': addr(proc(void, GLenum, GLsizeiptr, void_star, GLenum)(glBufferData)),
        'glBufferSubData': addr(proc(void, GLenum, GLintptr, GLsizeiptr, void_star)(glBufferSubData)),
        'glUnmapBuffer': addr(proc(GLboolean, GLenum)(glUnmapBuffer)),
        'glBlendEquationSeparate': addr(proc(void, GLenum, GLenum)(glBlendEquationSeparate)),
        'glDrawBuffers': addr(proc(void, GLsizei, GLenum_star)(glDrawBuffers)),
        'glStencilOpSeparate': addr(proc(void, GLenum, GLenum, GLenum, GLenum)(glStencilOpSeparate)),
        'glStencilFuncSeparate': addr(proc(void, GLenum, GLenum, GLint, GLuint)(glStencilFuncSeparate)),
        'glStencilMaskSeparate': addr(proc(void, GLenum, GLuint)(glStencilMaskSeparate)),
        'glAttachShader': addr(proc(void, GLuint, GLuint)(glAttachShader)),
        'glCompileShader': addr(proc(void, GLuint)(glCompileShader)),
        'glCreateProgram': addr(proc(GLuint)(glCreateProgram)),
        'glCreateShader': addr(proc(GLuint, GLenum)(glCreateShader)),
        'glDeleteProgram': addr(proc(void, GLuint)(glDeleteProgram)),
        'glDeleteShader': addr(proc(void, GLuint)(glDeleteShader)),
        'glEnableVertexAttribArray': addr(proc(void, GLuint)(glEnableVertexAttribArray)),
        'glGetActiveAttrib': addr(proc(void, GLuint, GLuint, GLsizei, GLsizei_star, GLint_star, GLenum_star, GLchar_star)(glGetActiveAttrib)),
        'glGetActiveUniform': addr(proc(void, GLuint, GLuint, GLsizei, GLsizei_star, GLint_star, GLenum_star, GLchar_star)(glGetActiveUniform)),
        'glGetAttribLocation': addr(proc(GLint, GLuint, GLchar_star)(glGetAttribLocation)),
        'glGetProgramiv': addr(proc(void, GLuint, GLenum, GLint_star)(glGetProgramiv)),
        'glGetProgramInfoLog': addr(proc(void, GLuint, GLsizei, GLsizei_star, GLchar_star)(glGetProgramInfoLog)),
        'glGetShaderiv': addr(proc(void, GLuint, GLenum, GLint_star)(glGetShaderiv)),
        'glGetShaderInfoLog': addr(proc(void, GLuint, GLsizei, GLsizei_star, GLchar_star)(glGetShaderInfoLog)),
        'glGetUniformLocation': addr(proc(GLint, GLuint, GLchar_star)(glGetUniformLocation)),
        'glLinkProgram': addr(proc(void, GLuint)(glLinkProgram)),
        'glShaderSource': addr(proc(void, GLuint, GLsizei, GLchar_star_star, GLint_star)(glShaderSource)),
        'glUseProgram': addr(proc(void, GLuint)(glUseProgram)),
        'glUniform1i': addr(proc(void, GLint, GLint)(glUniform1i)),
        'glUniform1fv': addr(proc(void, GLint, GLsizei, GLfloat_star)(glUniform1fv)),
        'glUniform2fv': addr(proc(void, GLint, GLsizei, GLfloat_star)(glUniform2fv)),
        'glUniform3fv': addr(proc(void, GLint, GLsizei, GLfloat_star)(glUniform3fv)),
        'glUniform4fv': addr(proc(void, GLint, GLsizei, GLfloat_star)(glUniform4fv)),
        'glUniform1iv': addr(proc(void, GLint, GLsizei, GLint_star)(glUniform1iv)),
        'glUniform2iv': addr(proc(void, GLint, GLsizei, GLint_star)(glUniform2iv)),
        'glUniform3iv': addr(proc(void, GLint, GLsizei, GLint_star)(glUniform3iv)),
        'glUniform4iv': addr(proc(void, GLint, GLsizei, GLint_star)(glUniform4iv)),
        'glUniformMatrix2fv': addr(proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix2fv)),
        'glUniformMatrix3fv': addr(proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix3fv)),
        'glUniformMatrix4fv': addr(proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix4fv)),
        'glVertexAttribPointer': addr(proc(void, GLuint, GLint, GLenum, GLboolean, GLsizei, void_star)(glVertexAttribPointer)),
        'glUniformMatrix2x3fv': addr(proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix2x3fv)),
        'glUniformMatrix3x2fv': addr(proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix3x2fv)),
        'glUniformMatrix2x4fv': addr(proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix2x4fv)),
        'glUniformMatrix4x2fv': addr(proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix4x2fv)),
        'glUniformMatrix3x4fv': addr(proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix3x4fv)),
        'glUniformMatrix4x3fv': addr(proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix4x3fv)),
        'glBindBufferRange': addr(proc(void, GLenum, GLuint, GLuint, GLintptr, GLsizeiptr)(glBindBufferRange)),
        'glVertexAttribIPointer': addr(proc(void, GLuint, GLint, GLenum, GLsizei, void_star)(glVertexAttribIPointer)),
        'glUniform1uiv': addr(proc(void, GLint, GLsizei, GLuint_star)(glUniform1uiv)),
        'glUniform2uiv': addr(proc(void, GLint, GLsizei, GLuint_star)(glUniform2uiv)),
        'glUniform3uiv': addr(proc(void, GLint, GLsizei, GLuint_star)(glUniform3uiv)),
        'glUniform4uiv': addr(proc(void, GLint, GLsizei, GLuint_star)(glUniform4uiv)),
        'glClearBufferiv': addr(proc(void, GLenum, GLint, GLint_star)(glClearBufferiv)),
        'glClearBufferuiv': addr(proc(void, GLenum, GLint, GLuint_star)(glClearBufferuiv)),
        'glClearBufferfv': addr(proc(void, GLenum, GLint, GLfloat_star)(glClearBufferfv)),
        'glClearBufferfi': addr(proc(void, GLenum, GLint, GLfloat, GLint)(glClearBufferfi)),
        'glBindRenderbuffer': addr(proc(void, GLenum, GLuint)(glBindRenderbuffer)),
        'glDeleteRenderbuffers': addr(proc(void, GLsizei, GLuint_star)(glDeleteRenderbuffers)),
        'glGenRenderbuffers': addr(proc(void, GLsizei, GLuint_star)(glGenRenderbuffers)),
        'glBindFramebuffer': addr(proc(void, GLenum, GLuint)(glBindFramebuffer)),
        'glDeleteFramebuffers': addr(proc(void, GLsizei, GLuint_star)(glDeleteFramebuffers)),
        'glGenFramebuffers': addr(proc(void, GLsizei, GLuint_star)(glGenFramebuffers)),
        'glFramebufferTexture2D': addr(proc(void, GLenum, GLenum, GLenum, GLuint, GLint)(glFramebufferTexture2D)),
        'glFramebufferRenderbuffer': addr(proc(void, GLenum, GLenum, GLenum, GLuint)(glFramebufferRenderbuffer)),
        'glGenerateMipmap': addr(proc(void, GLenum)(glGenerateMipmap)),
        'glBlitFramebuffer': addr(proc(void, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum)(glBlitFramebuffer)),
        'glRenderbufferStorageMultisample': addr(proc(void, GLenum, GLsizei, GLenum, GLsizei, GLsizei)(glRenderbufferStorageMultisample)),
        'glFramebufferTextureLayer': addr(proc(void, GLenum, GLenum, GLuint, GLint, GLint)(glFramebufferTextureLayer)),
        'glMapBufferRange': addr(proc(void_star, GLenum, GLintptr, GLsizeiptr, GLbitfield)(glMapBufferRange)),
        'glBindVertexArray': addr(proc(void, GLuint)(glBindVertexArray)),
        'glDeleteVertexArrays': addr(proc(void, GLsizei, GLuint_star)(glDeleteVertexArrays)),
        'glGenVertexArrays': addr(proc(void, GLsizei, GLuint_star)(glGenVertexArrays)),
        'glDrawArraysInstanced': addr(proc(void, GLenum, GLint, GLsizei, GLsizei)(glDrawArraysInstanced)),
        'glDrawElementsInstanced': addr(proc(void, GLenum, GLsizei, GLenum, void_star, GLsizei)(glDrawElementsInstanced)),
        'glGetUniformBlockIndex': addr(proc(GLuint, GLuint, GLchar_star)(glGetUniformBlockIndex)),
        'glGetActiveUniformBlockiv': addr(proc(void, GLuint, GLuint, GLenum, GLint_star)(glGetActiveUniformBlockiv)),
        'glGetActiveUniformBlockName': addr(proc(void, GLuint, GLuint, GLsizei, GLsizei_star, GLchar_star)(glGetActiveUniformBlockName)),
        'glUniformBlockBinding': addr(proc(void, GLuint, GLuint, GLuint)(glUniformBlockBinding)),
        'glFenceSync': addr(proc(GLsync, GLenum, GLbitfield)(glFenceSync)),
        'glDeleteSync': addr(proc(void, GLsync)(glDeleteSync)),
        'glClientWaitSync': addr(proc(GLenum, GLsync, GLbitfield, GLuint64)(glClientWaitSync)),
        'glGenSamplers': addr(proc(void, GLsizei, GLuint_star)(glGenSamplers)),
        'glDeleteSamplers': addr(proc(void, GLsizei, GLuint_star)(glDeleteSamplers)),
        'glBindSampler': addr(proc(void, GLuint, GLuint)(glBindSampler)),
        'glSamplerParameteri': addr(proc(void, GLuint, GLenum, GLint)(glSamplerParameteri)),
        'glSamplerParameterf': addr(proc(void, GLuint, GLenum, GLfloat)(glSamplerParameterf)),
        'glVertexAttribDivisor': addr(proc(void, GLuint, GLuint)(glVertexAttribDivisor)),
    }


class PyodideCanvas:
    def __init__(self, size=(1280, 720)):
        self.size = size
        self.aspect = self.size[0] / self.size[1]
        self.canvas = js.document.createElement('canvas')
        self.canvas.width, self.canvas.height = size
        self.gl = self.canvas.getContext('webgl2', js.JSON.parse('{"antialias": false}'))
        self.mouse = (0, 0)

        def mousemove(evt):
            self.mouse = (evt.x, evt.y)

        js.document.body.appendChild(self.canvas)
        self.canvas.addEventListener('mousemove', create_proxy(mousemove))
        # js.window.gl = self.gl
        self.webgl = webgl(self.gl)
        self.time = 0.0

    def update(self):
        self.time += 1.0 / 60.0

    def load_opengl_function(self, name):
        return self.webgl.get(name, (0, None))[0]
