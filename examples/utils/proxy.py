import ctypes


class DefaultLoader:
    def __init__(self):
        import ctypes
        if hasattr(ctypes, 'WinDLL'):
            lib = ctypes.WinDLL('Opengl32.dll')
            proc = ctypes.cast(lib.wglGetProcAddress, ctypes.CFUNCTYPE(ctypes.c_ulonglong, ctypes.c_char_p))
        else:
            lib = ctypes.CDLL('libGL.so')
            proc = ctypes.cast(lib.glXGetProcAddress, ctypes.CFUNCTYPE(ctypes.c_ulonglong, ctypes.c_char_p))
        self.load_opengl_function = lambda name: proc(name.encode()) or ctypes.cast(lib[name], ctypes.c_void_p).value


def make_gl():
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
    GLchar_star = ctypes.c_char_p
    GLsync = ctypes.c_void_p
    GLubyte_star = ctypes.c_void_p
    GLint_star = ctypes.POINTER(ctypes.c_int32)
    GLsizei_star = ctypes.POINTER(ctypes.c_int32)
    GLuint_star = ctypes.POINTER(ctypes.c_uint32)
    GLenum_star = ctypes.POINTER(ctypes.c_uint32)
    GLfloat_star = ctypes.POINTER(ctypes.c_float)
    GLchar_star_star = ctypes.POINTER(ctypes.c_char_p)
    GLintptr = ctypes.c_uint32
    GLsizeiptr = ctypes.c_uint32

    loader = DefaultLoader()
    load = loader.load_opengl_function

    return {
        'glCullFace': ctypes.cast(load('glCullFace'), proc(void, GLenum)),
        'glClear': ctypes.cast(load('glClear'), proc(void, GLbitfield)),
        'glTexParameteri': ctypes.cast(load('glTexParameteri'), proc(void, GLenum, GLenum, GLint)),
        'glTexImage2D': ctypes.cast(load('glTexImage2D'), proc(void, GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, void_star)),
        'glDepthMask': ctypes.cast(load('glDepthMask'), proc(void, GLboolean)),
        'glDisable': ctypes.cast(load('glDisable'), proc(void, GLenum)),
        'glEnable': ctypes.cast(load('glEnable'), proc(void, GLenum)),
        'glFlush': ctypes.cast(load('glFlush'), proc(void)),
        'glDepthFunc': ctypes.cast(load('glDepthFunc'), proc(void, GLenum)),
        'glReadBuffer': ctypes.cast(load('glReadBuffer'), proc(void, GLenum)),
        'glReadPixels': ctypes.cast(load('glReadPixels'), proc(void, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void_star)),
        'glGetError': ctypes.cast(load('glGetError'), proc(GLenum)),
        'glGetIntegerv': ctypes.cast(load('glGetIntegerv'), proc(void, GLenum, GLint_star)),
        'glGetString': ctypes.cast(load('glGetString'), proc(GLubyte_star, GLenum)),
        'glViewport': ctypes.cast(load('glViewport'), proc(void, GLint, GLint, GLsizei, GLsizei)),
        'glTexSubImage2D': ctypes.cast(load('glTexSubImage2D'), proc(void, GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void_star)),
        'glBindTexture': ctypes.cast(load('glBindTexture'), proc(void, GLenum, GLuint)),
        'glDeleteTextures': ctypes.cast(load('glDeleteTextures'), proc(void, GLsizei, GLuint_star)),
        'glGenTextures': ctypes.cast(load('glGenTextures'), proc(void, GLsizei, GLuint_star)),
        'glTexImage3D': ctypes.cast(load('glTexImage3D'), proc(void, GLenum, GLint, GLint, GLsizei, GLsizei, GLsizei, GLint, GLenum, GLenum, void_star)),
        'glTexSubImage3D': ctypes.cast(load('glTexSubImage3D'), proc(void, GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei, GLenum, GLenum, void_star)),
        'glActiveTexture': ctypes.cast(load('glActiveTexture'), proc(void, GLenum)),
        'glBlendFuncSeparate': ctypes.cast(load('glBlendFuncSeparate'), proc(void, GLenum, GLenum, GLenum, GLenum)),
        'glGenQueries': ctypes.cast(load('glGenQueries'), proc(void, GLsizei, GLuint_star)),
        'glBeginQuery': ctypes.cast(load('glBeginQuery'), proc(void, GLenum, GLuint)),
        'glEndQuery': ctypes.cast(load('glEndQuery'), proc(void, GLenum)),
        'glGetQueryObjectuiv': ctypes.cast(load('glGetQueryObjectuiv'), proc(void, GLuint, GLenum, GLuint_star)),
        'glBindBuffer': ctypes.cast(load('glBindBuffer'), proc(void, GLenum, GLuint)),
        'glDeleteBuffers': ctypes.cast(load('glDeleteBuffers'), proc(void, GLsizei, GLuint_star)),
        'glGenBuffers': ctypes.cast(load('glGenBuffers'), proc(void, GLsizei, GLuint_star)),
        'glBufferData': ctypes.cast(load('glBufferData'), proc(void, GLenum, GLsizeiptr, void_star, GLenum)),
        'glBufferSubData': ctypes.cast(load('glBufferSubData'), proc(void, GLenum, GLintptr, GLsizeiptr, void_star)),
        'glUnmapBuffer': ctypes.cast(load('glUnmapBuffer'), proc(GLboolean, GLenum)),
        'glBlendEquationSeparate': ctypes.cast(load('glBlendEquationSeparate'), proc(void, GLenum, GLenum)),
        'glDrawBuffers': ctypes.cast(load('glDrawBuffers'), proc(void, GLsizei, GLenum_star)),
        'glStencilOpSeparate': ctypes.cast(load('glStencilOpSeparate'), proc(void, GLenum, GLenum, GLenum, GLenum)),
        'glStencilFuncSeparate': ctypes.cast(load('glStencilFuncSeparate'), proc(void, GLenum, GLenum, GLint, GLuint)),
        'glStencilMaskSeparate': ctypes.cast(load('glStencilMaskSeparate'), proc(void, GLenum, GLuint)),
        'glAttachShader': ctypes.cast(load('glAttachShader'), proc(void, GLuint, GLuint)),
        'glCompileShader': ctypes.cast(load('glCompileShader'), proc(void, GLuint)),
        'glCreateProgram': ctypes.cast(load('glCreateProgram'), proc(GLuint)),
        'glCreateShader': ctypes.cast(load('glCreateShader'), proc(GLuint, GLenum)),
        'glDeleteProgram': ctypes.cast(load('glDeleteProgram'), proc(void, GLuint)),
        'glDeleteShader': ctypes.cast(load('glDeleteShader'), proc(void, GLuint)),
        'glEnableVertexAttribArray': ctypes.cast(load('glEnableVertexAttribArray'), proc(void, GLuint)),
        'glGetActiveAttrib': ctypes.cast(load('glGetActiveAttrib'), proc(void, GLuint, GLuint, GLsizei, GLsizei_star, GLint_star, GLenum_star, GLchar_star)),
        'glGetActiveUniform': ctypes.cast(load('glGetActiveUniform'), proc(void, GLuint, GLuint, GLsizei, GLsizei_star, GLint_star, GLenum_star, GLchar_star)),
        'glGetAttribLocation': ctypes.cast(load('glGetAttribLocation'), proc(GLint, GLuint, GLchar_star)),
        'glGetProgramiv': ctypes.cast(load('glGetProgramiv'), proc(void, GLuint, GLenum, GLint_star)),
        'glGetProgramInfoLog': ctypes.cast(load('glGetProgramInfoLog'), proc(void, GLuint, GLsizei, GLsizei_star, GLchar_star)),
        'glGetShaderiv': ctypes.cast(load('glGetShaderiv'), proc(void, GLuint, GLenum, GLint_star)),
        'glGetShaderInfoLog': ctypes.cast(load('glGetShaderInfoLog'), proc(void, GLuint, GLsizei, GLsizei_star, GLchar_star)),
        'glGetUniformLocation': ctypes.cast(load('glGetUniformLocation'), proc(GLint, GLuint, GLchar_star)),
        'glLinkProgram': ctypes.cast(load('glLinkProgram'), proc(void, GLuint)),
        'glShaderSource': ctypes.cast(load('glShaderSource'), proc(void, GLuint, GLsizei, GLchar_star_star, GLint_star)),
        'glUseProgram': ctypes.cast(load('glUseProgram'), proc(void, GLuint)),
        'glUniform1i': ctypes.cast(load('glUniform1i'), proc(void, GLint, GLint)),
        'glUniform1fv': ctypes.cast(load('glUniform1fv'), proc(void, GLint, GLsizei, GLfloat_star)),
        'glUniform2fv': ctypes.cast(load('glUniform2fv'), proc(void, GLint, GLsizei, GLfloat_star)),
        'glUniform3fv': ctypes.cast(load('glUniform3fv'), proc(void, GLint, GLsizei, GLfloat_star)),
        'glUniform4fv': ctypes.cast(load('glUniform4fv'), proc(void, GLint, GLsizei, GLfloat_star)),
        'glUniform1iv': ctypes.cast(load('glUniform1iv'), proc(void, GLint, GLsizei, GLint_star)),
        'glUniform2iv': ctypes.cast(load('glUniform2iv'), proc(void, GLint, GLsizei, GLint_star)),
        'glUniform3iv': ctypes.cast(load('glUniform3iv'), proc(void, GLint, GLsizei, GLint_star)),
        'glUniform4iv': ctypes.cast(load('glUniform4iv'), proc(void, GLint, GLsizei, GLint_star)),
        'glUniformMatrix2fv': ctypes.cast(load('glUniformMatrix2fv'), proc(void, GLint, GLsizei, GLboolean, GLfloat_star)),
        'glUniformMatrix3fv': ctypes.cast(load('glUniformMatrix3fv'), proc(void, GLint, GLsizei, GLboolean, GLfloat_star)),
        'glUniformMatrix4fv': ctypes.cast(load('glUniformMatrix4fv'), proc(void, GLint, GLsizei, GLboolean, GLfloat_star)),
        'glVertexAttribPointer': ctypes.cast(load('glVertexAttribPointer'), proc(void, GLuint, GLint, GLenum, GLboolean, GLsizei, void_star)),
        'glUniformMatrix2x3fv': ctypes.cast(load('glUniformMatrix2x3fv'), proc(void, GLint, GLsizei, GLboolean, GLfloat_star)),
        'glUniformMatrix3x2fv': ctypes.cast(load('glUniformMatrix3x2fv'), proc(void, GLint, GLsizei, GLboolean, GLfloat_star)),
        'glUniformMatrix2x4fv': ctypes.cast(load('glUniformMatrix2x4fv'), proc(void, GLint, GLsizei, GLboolean, GLfloat_star)),
        'glUniformMatrix4x2fv': ctypes.cast(load('glUniformMatrix4x2fv'), proc(void, GLint, GLsizei, GLboolean, GLfloat_star)),
        'glUniformMatrix3x4fv': ctypes.cast(load('glUniformMatrix3x4fv'), proc(void, GLint, GLsizei, GLboolean, GLfloat_star)),
        'glUniformMatrix4x3fv': ctypes.cast(load('glUniformMatrix4x3fv'), proc(void, GLint, GLsizei, GLboolean, GLfloat_star)),
        'glBindBufferRange': ctypes.cast(load('glBindBufferRange'), proc(void, GLenum, GLuint, GLuint, GLintptr, GLsizeiptr)),
        'glVertexAttribIPointer': ctypes.cast(load('glVertexAttribIPointer'), proc(void, GLuint, GLint, GLenum, GLsizei, void_star)),
        'glUniform1uiv': ctypes.cast(load('glUniform1uiv'), proc(void, GLint, GLsizei, GLuint_star)),
        'glUniform2uiv': ctypes.cast(load('glUniform2uiv'), proc(void, GLint, GLsizei, GLuint_star)),
        'glUniform3uiv': ctypes.cast(load('glUniform3uiv'), proc(void, GLint, GLsizei, GLuint_star)),
        'glUniform4uiv': ctypes.cast(load('glUniform4uiv'), proc(void, GLint, GLsizei, GLuint_star)),
        'glClearBufferiv': ctypes.cast(load('glClearBufferiv'), proc(void, GLenum, GLint, GLint_star)),
        'glClearBufferuiv': ctypes.cast(load('glClearBufferuiv'), proc(void, GLenum, GLint, GLuint_star)),
        'glClearBufferfv': ctypes.cast(load('glClearBufferfv'), proc(void, GLenum, GLint, GLfloat_star)),
        'glClearBufferfi': ctypes.cast(load('glClearBufferfi'), proc(void, GLenum, GLint, GLfloat, GLint)),
        'glBindRenderbuffer': ctypes.cast(load('glBindRenderbuffer'), proc(void, GLenum, GLuint)),
        'glDeleteRenderbuffers': ctypes.cast(load('glDeleteRenderbuffers'), proc(void, GLsizei, GLuint_star)),
        'glGenRenderbuffers': ctypes.cast(load('glGenRenderbuffers'), proc(void, GLsizei, GLuint_star)),
        'glBindFramebuffer': ctypes.cast(load('glBindFramebuffer'), proc(void, GLenum, GLuint)),
        'glDeleteFramebuffers': ctypes.cast(load('glDeleteFramebuffers'), proc(void, GLsizei, GLuint_star)),
        'glGenFramebuffers': ctypes.cast(load('glGenFramebuffers'), proc(void, GLsizei, GLuint_star)),
        'glFramebufferTexture2D': ctypes.cast(load('glFramebufferTexture2D'), proc(void, GLenum, GLenum, GLenum, GLuint, GLint)),
        'glFramebufferRenderbuffer': ctypes.cast(load('glFramebufferRenderbuffer'), proc(void, GLenum, GLenum, GLenum, GLuint)),
        'glGenerateMipmap': ctypes.cast(load('glGenerateMipmap'), proc(void, GLenum)),
        'glBlitFramebuffer': ctypes.cast(load('glBlitFramebuffer'), proc(void, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum)),
        'glRenderbufferStorageMultisample': ctypes.cast(load('glRenderbufferStorageMultisample'), proc(void, GLenum, GLsizei, GLenum, GLsizei, GLsizei)),
        'glFramebufferTextureLayer': ctypes.cast(load('glFramebufferTextureLayer'), proc(void, GLenum, GLenum, GLuint, GLint, GLint)),
        'glMapBufferRange': ctypes.cast(load('glMapBufferRange'), proc(void_star, GLenum, GLintptr, GLsizeiptr, GLbitfield)),
        'glBindVertexArray': ctypes.cast(load('glBindVertexArray'), proc(void, GLuint)),
        'glDeleteVertexArrays': ctypes.cast(load('glDeleteVertexArrays'), proc(void, GLsizei, GLuint_star)),
        'glGenVertexArrays': ctypes.cast(load('glGenVertexArrays'), proc(void, GLsizei, GLuint_star)),
        'glDrawArraysInstanced': ctypes.cast(load('glDrawArraysInstanced'), proc(void, GLenum, GLint, GLsizei, GLsizei)),
        'glDrawElementsInstanced': ctypes.cast(load('glDrawElementsInstanced'), proc(void, GLenum, GLsizei, GLenum, void_star, GLsizei)),
        'glGetUniformBlockIndex': ctypes.cast(load('glGetUniformBlockIndex'), proc(GLuint, GLuint, GLchar_star)),
        'glGetActiveUniformBlockiv': ctypes.cast(load('glGetActiveUniformBlockiv'), proc(void, GLuint, GLuint, GLenum, GLint_star)),
        'glGetActiveUniformBlockName': ctypes.cast(load('glGetActiveUniformBlockName'), proc(void, GLuint, GLuint, GLsizei, GLsizei_star, GLchar_star)),
        'glUniformBlockBinding': ctypes.cast(load('glUniformBlockBinding'), proc(void, GLuint, GLuint, GLuint)),
        'glFenceSync': ctypes.cast(load('glFenceSync'), proc(GLsync, GLenum, GLbitfield)),
        'glDeleteSync': ctypes.cast(load('glDeleteSync'), proc(void, GLsync)),
        'glClientWaitSync': ctypes.cast(load('glClientWaitSync'), proc(GLenum, GLsync, GLbitfield, GLuint64)),
        'glGenSamplers': ctypes.cast(load('glGenSamplers'), proc(void, GLsizei, GLuint_star)),
        'glDeleteSamplers': ctypes.cast(load('glDeleteSamplers'), proc(void, GLsizei, GLuint_star)),
        'glBindSampler': ctypes.cast(load('glBindSampler'), proc(void, GLuint, GLuint)),
        'glSamplerParameteri': ctypes.cast(load('glSamplerParameteri'), proc(void, GLuint, GLenum, GLint)),
        'glSamplerParameterf': ctypes.cast(load('glSamplerParameterf'), proc(void, GLuint, GLenum, GLfloat)),
        'glVertexAttribDivisor': ctypes.cast(load('glVertexAttribDivisor'), proc(void, GLuint, GLuint)),
    }


def make_wrapper(gl):
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
    GLchar_star = ctypes.c_char_p
    GLsync = ctypes.c_void_p
    GLubyte_star = ctypes.c_void_p
    GLint_star = ctypes.POINTER(ctypes.c_int32)
    GLsizei_star = ctypes.POINTER(ctypes.c_int32)
    GLuint_star = ctypes.POINTER(ctypes.c_uint32)
    GLenum_star = ctypes.POINTER(ctypes.c_uint32)
    GLfloat_star = ctypes.POINTER(ctypes.c_float)
    GLchar_star_star = ctypes.POINTER(ctypes.c_char_p)
    GLintptr = ctypes.c_uint32
    GLsizeiptr = ctypes.c_uint32

    def glCullFace(mode: GLenum):
        print('glCullFace')
        return gl['glCullFace'](mode)
    def glClear(mask: GLbitfield):
        print('glClear')
        return gl['glClear'](mask)
    def glTexParameteri(target: GLenum, pname: GLenum, param: GLint):
        print('glTexParameteri')
        return gl['glTexParameteri'](target, pname, param)
    def glTexImage2D(target: GLenum, level: GLint, internalformat: GLint, width: GLsizei, height: GLsizei, border: GLint, format: GLenum, type: GLenum, pixels: void_star):
        print('glTexImage2D')
        return gl['glTexImage2D'](target, level, internalformat, width, height, border, format, type, pixels)
    def glDepthMask(flag: GLboolean):
        print('glDepthMask')
        return gl['glDepthMask'](flag)
    def glDisable(cap: GLenum):
        print('glDisable')
        return gl['glDisable'](cap)
    def glEnable(cap: GLenum):
        print('glEnable')
        return gl['glEnable'](cap)
    def glFlush():
        print('glFlush')
        return gl['glFlush']()
    def glDepthFunc(func: GLenum):
        print('glDepthFunc')
        return gl['glDepthFunc'](func)
    def glReadBuffer(src: GLenum):
        print('glReadBuffer')
        return gl['glReadBuffer'](src)
    def glReadPixels(x: GLint, y: GLint, width: GLsizei, height: GLsizei, format: GLenum, type: GLenum, pixels: void_star):
        print('glReadPixels')
        return gl['glReadPixels'](x, y, width, height, format, type, pixels)
    def glGetError():
        print('glGetError')
        return gl['glGetError']()
    def glGetIntegerv(pname: GLenum, data: GLint_star):
        print('glGetIntegerv')
        return gl['glGetIntegerv'](pname, data)
    def glGetString(name: GLenum):
        print('glGetString')
        return gl['glGetString'](name)
    def glViewport(x: GLint, y: GLint, width: GLsizei, height: GLsizei):
        print('glViewport')
        return gl['glViewport'](x, y, width, height)
    def glTexSubImage2D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, width: GLsizei, height: GLsizei, format: GLenum, type: GLenum, pixels: void_star):
        print('glTexSubImage2D')
        return gl['glTexSubImage2D'](target, level, xoffset, yoffset, width, height, format, type, pixels)
    def glBindTexture(target: GLenum, texture: GLuint):
        print('glBindTexture')
        return gl['glBindTexture'](target, texture)
    def glDeleteTextures(n: GLsizei, textures: GLuint_star):
        print('glDeleteTextures')
        return gl['glDeleteTextures'](n, textures)
    def glGenTextures(n: GLsizei, textures: GLuint_star):
        print('glGenTextures')
        return gl['glGenTextures'](n, textures)
    def glTexImage3D(target: GLenum, level: GLint, internalformat: GLint, width: GLsizei, height: GLsizei, depth: GLsizei, border: GLint, format: GLenum, type: GLenum, pixels: void_star):
        print('glTexImage3D')
        return gl['glTexImage3D'](target, level, internalformat, width, height, depth, border, format, type, pixels)
    def glTexSubImage3D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, zoffset: GLint, width: GLsizei, height: GLsizei, depth: GLsizei, format: GLenum, type: GLenum, pixels: void_star):
        print('glTexSubImage3D')
        return gl['glTexSubImage3D'](target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels)
    def glActiveTexture(texture: GLenum):
        print('glActiveTexture')
        return gl['glActiveTexture'](texture)
    def glBlendFuncSeparate(sfactorRGB: GLenum, dfactorRGB: GLenum, sfactorAlpha: GLenum, dfactorAlpha: GLenum):
        print('glBlendFuncSeparate')
        return gl['glBlendFuncSeparate'](sfactorRGB, dfactorRGB, sfactorAlpha, dfactorAlpha)
    def glGenQueries(n: GLsizei, ids: GLuint_star):
        print('glGenQueries')
        return gl['glGenQueries'](n, ids)
    def glBeginQuery(target: GLenum, id: GLuint):
        print('glBeginQuery')
        return gl['glBeginQuery'](target, id)
    def glEndQuery(target: GLenum):
        print('glEndQuery')
        return gl['glEndQuery'](target)
    def glGetQueryObjectuiv(id: GLuint, pname: GLenum, params: GLuint_star):
        print('glGetQueryObjectuiv')
        return gl['glGetQueryObjectuiv'](id, pname, params)
    def glBindBuffer(target: GLenum, buffer: GLuint):
        print('glBindBuffer')
        return gl['glBindBuffer'](target, buffer)
    def glDeleteBuffers(n: GLsizei, buffers: GLuint_star):
        print('glDeleteBuffers')
        return gl['glDeleteBuffers'](n, buffers)
    def glGenBuffers(n: GLsizei, buffers: GLuint_star):
        print('glGenBuffers')
        return gl['glGenBuffers'](n, buffers)
    def glBufferData(target: GLenum, size: GLsizeiptr, data: void_star, usage: GLenum):
        print('glBufferData')
        return gl['glBufferData'](target, size, data, usage)
    def glBufferSubData(target: GLenum, offset: GLintptr, size: GLsizeiptr, data: void_star):
        print('glBufferSubData')
        return gl['glBufferSubData'](target, offset, size, data)
    def glUnmapBuffer(target: GLenum):
        print('glUnmapBuffer')
        return gl['glUnmapBuffer'](target)
    def glBlendEquationSeparate(modeRGB: GLenum, modeAlpha: GLenum):
        print('glBlendEquationSeparate')
        return gl['glBlendEquationSeparate'](modeRGB, modeAlpha)
    def glDrawBuffers(n: GLsizei, bufs: GLenum_star):
        print('glDrawBuffers')
        return gl['glDrawBuffers'](n, bufs)
    def glStencilOpSeparate(face: GLenum, sfail: GLenum, dpfail: GLenum, dppass: GLenum):
        print('glStencilOpSeparate')
        return gl['glStencilOpSeparate'](face, sfail, dpfail, dppass)
    def glStencilFuncSeparate(face: GLenum, func: GLenum, ref: GLint, mask: GLuint):
        print('glStencilFuncSeparate')
        return gl['glStencilFuncSeparate'](face, func, ref, mask)
    def glStencilMaskSeparate(face: GLenum, mask: GLuint):
        print('glStencilMaskSeparate')
        return gl['glStencilMaskSeparate'](face, mask)
    def glAttachShader(program: GLuint, shader: GLuint):
        print('glAttachShader')
        return gl['glAttachShader'](program, shader)
    def glCompileShader(shader: GLuint):
        print('glCompileShader')
        return gl['glCompileShader'](shader)
    def glCreateProgram():
        print('glCreateProgram')
        return gl['glCreateProgram']()
    def glCreateShader(type: GLenum):
        print('glCreateShader')
        return gl['glCreateShader'](type)
    def glDeleteProgram(program: GLuint):
        print('glDeleteProgram')
        return gl['glDeleteProgram'](program)
    def glDeleteShader(shader: GLuint):
        print('glDeleteShader')
        return gl['glDeleteShader'](shader)
    def glEnableVertexAttribArray(index: GLuint):
        print('glEnableVertexAttribArray')
        return gl['glEnableVertexAttribArray'](index)
    def glGetActiveAttrib(program: GLuint, index: GLuint, bufSize: GLsizei, length: GLsizei_star, size: GLint_star, type: GLenum_star, name: GLchar_star):
        print('glGetActiveAttrib')
        return gl['glGetActiveAttrib'](program, index, bufSize, length, size, type, name)
    def glGetActiveUniform(program: GLuint, index: GLuint, bufSize: GLsizei, length: GLsizei_star, size: GLint_star, type: GLenum_star, name: GLchar_star):
        print('glGetActiveUniform')
        return gl['glGetActiveUniform'](program, index, bufSize, length, size, type, name)
    def glGetAttribLocation(program: GLuint, name: GLchar_star):
        print('glGetAttribLocation')
        return gl['glGetAttribLocation'](program, name)
    def glGetProgramiv(program: GLuint, pname: GLenum, params: GLint_star):
        print('glGetProgramiv')
        return gl['glGetProgramiv'](program, pname, params)
    def glGetProgramInfoLog(program: GLuint, bufSize: GLsizei, length: GLsizei_star, infoLog: GLchar_star):
        print('glGetProgramInfoLog')
        return gl['glGetProgramInfoLog'](program, bufSize, length, infoLog)
    def glGetShaderiv(shader: GLuint, pname: GLenum, params: GLint_star):
        print('glGetShaderiv')
        return gl['glGetShaderiv'](shader, pname, params)
    def glGetShaderInfoLog(shader: GLuint, bufSize: GLsizei, length: GLsizei_star, infoLog: GLchar_star):
        print('glGetShaderInfoLog')
        return gl['glGetShaderInfoLog'](shader, bufSize, length, infoLog)
    def glGetUniformLocation(program: GLuint, name: GLchar_star):
        print('glGetUniformLocation')
        return gl['glGetUniformLocation'](program, name)
    def glLinkProgram(program: GLuint):
        print('glLinkProgram')
        return gl['glLinkProgram'](program)
    def glShaderSource(shader: GLuint, count: GLsizei, string: GLchar_star_star, length: GLint_star):
        print('glShaderSource')
        return gl['glShaderSource'](shader, count, string, length)
    def glUseProgram(program: GLuint):
        print('glUseProgram')
        return gl['glUseProgram'](program)
    def glUniform1i(location: GLint, v0: GLint):
        print('glUniform1i')
        return gl['glUniform1i'](location, v0)
    def glUniform1fv(location: GLint, count: GLsizei, value: GLfloat_star):
        print('glUniform1fv')
        return gl['glUniform1fv'](location, count, value)
    def glUniform2fv(location: GLint, count: GLsizei, value: GLfloat_star):
        print('glUniform2fv')
        return gl['glUniform2fv'](location, count, value)
    def glUniform3fv(location: GLint, count: GLsizei, value: GLfloat_star):
        print('glUniform3fv')
        return gl['glUniform3fv'](location, count, value)
    def glUniform4fv(location: GLint, count: GLsizei, value: GLfloat_star):
        print('glUniform4fv')
        return gl['glUniform4fv'](location, count, value)
    def glUniform1iv(location: GLint, count: GLsizei, value: GLint_star):
        print('glUniform1iv')
        return gl['glUniform1iv'](location, count, value)
    def glUniform2iv(location: GLint, count: GLsizei, value: GLint_star):
        print('glUniform2iv')
        return gl['glUniform2iv'](location, count, value)
    def glUniform3iv(location: GLint, count: GLsizei, value: GLint_star):
        print('glUniform3iv')
        return gl['glUniform3iv'](location, count, value)
    def glUniform4iv(location: GLint, count: GLsizei, value: GLint_star):
        print('glUniform4iv')
        return gl['glUniform4iv'](location, count, value)
    def glUniformMatrix2fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        print('glUniformMatrix2fv')
        return gl['glUniformMatrix2fv'](location, count, transpose, value)
    def glUniformMatrix3fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        print('glUniformMatrix3fv')
        return gl['glUniformMatrix3fv'](location, count, transpose, value)
    def glUniformMatrix4fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        print('glUniformMatrix4fv')
        return gl['glUniformMatrix4fv'](location, count, transpose, value)
    def glVertexAttribPointer(index: GLuint, size: GLint, type: GLenum, normalized: GLboolean, stride: GLsizei, pointer: void_star):
        print('glVertexAttribPointer')
        return gl['glVertexAttribPointer'](index, size, type, normalized, stride, pointer)
    def glUniformMatrix2x3fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        print('glUniformMatrix2x3fv')
        return gl['glUniformMatrix2x3fv'](location, count, transpose, value)
    def glUniformMatrix3x2fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        print('glUniformMatrix3x2fv')
        return gl['glUniformMatrix3x2fv'](location, count, transpose, value)
    def glUniformMatrix2x4fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        print('glUniformMatrix2x4fv')
        return gl['glUniformMatrix2x4fv'](location, count, transpose, value)
    def glUniformMatrix4x2fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        print('glUniformMatrix4x2fv')
        return gl['glUniformMatrix4x2fv'](location, count, transpose, value)
    def glUniformMatrix3x4fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        print('glUniformMatrix3x4fv')
        return gl['glUniformMatrix3x4fv'](location, count, transpose, value)
    def glUniformMatrix4x3fv(location: GLint, count: GLsizei, transpose: GLboolean, value: GLfloat_star):
        print('glUniformMatrix4x3fv')
        return gl['glUniformMatrix4x3fv'](location, count, transpose, value)
    def glBindBufferRange(target: GLenum, index: GLuint, buffer: GLuint, offset: GLintptr, size: GLsizeiptr):
        print('glBindBufferRange')
        return gl['glBindBufferRange'](target, index, buffer, offset, size)
    def glVertexAttribIPointer(index: GLuint, size: GLint, type: GLenum, stride: GLsizei, pointer: void_star):
        print('glVertexAttribIPointer')
        return gl['glVertexAttribIPointer'](index, size, type, stride, pointer)
    def glUniform1uiv(location: GLint, count: GLsizei, value: GLuint_star):
        print('glUniform1uiv')
        return gl['glUniform1uiv'](location, count, value)
    def glUniform2uiv(location: GLint, count: GLsizei, value: GLuint_star):
        print('glUniform2uiv')
        return gl['glUniform2uiv'](location, count, value)
    def glUniform3uiv(location: GLint, count: GLsizei, value: GLuint_star):
        print('glUniform3uiv')
        return gl['glUniform3uiv'](location, count, value)
    def glUniform4uiv(location: GLint, count: GLsizei, value: GLuint_star):
        print('glUniform4uiv')
        return gl['glUniform4uiv'](location, count, value)
    def glClearBufferiv(buffer: GLenum, drawbuffer: GLint, value: GLint_star):
        print('glClearBufferiv')
        return gl['glClearBufferiv'](buffer, drawbuffer, value)
    def glClearBufferuiv(buffer: GLenum, drawbuffer: GLint, value: GLuint_star):
        print('glClearBufferuiv')
        return gl['glClearBufferuiv'](buffer, drawbuffer, value)
    def glClearBufferfv(buffer: GLenum, drawbuffer: GLint, value: GLfloat_star):
        print('glClearBufferfv')
        return gl['glClearBufferfv'](buffer, drawbuffer, value)
    def glClearBufferfi(buffer: GLenum, drawbuffer: GLint, depth: GLfloat, stencil: GLint):
        print('glClearBufferfi')
        return gl['glClearBufferfi'](buffer, drawbuffer, depth, stencil)
    def glBindRenderbuffer(target: GLenum, renderbuffer: GLuint):
        print('glBindRenderbuffer')
        return gl['glBindRenderbuffer'](target, renderbuffer)
    def glDeleteRenderbuffers(n: GLsizei, renderbuffers: GLuint_star):
        print('glDeleteRenderbuffers')
        return gl['glDeleteRenderbuffers'](n, renderbuffers)
    def glGenRenderbuffers(n: GLsizei, renderbuffers: GLuint_star):
        print('glGenRenderbuffers')
        return gl['glGenRenderbuffers'](n, renderbuffers)
    def glBindFramebuffer(target: GLenum, framebuffer: GLuint):
        print('glBindFramebuffer')
        return gl['glBindFramebuffer'](target, framebuffer)
    def glDeleteFramebuffers(n: GLsizei, framebuffers: GLuint_star):
        print('glDeleteFramebuffers')
        return gl['glDeleteFramebuffers'](n, framebuffers)
    def glGenFramebuffers(n: GLsizei, framebuffers: GLuint_star):
        print('glGenFramebuffers')
        return gl['glGenFramebuffers'](n, framebuffers)
    def glFramebufferTexture2D(target: GLenum, attachment: GLenum, textarget: GLenum, texture: GLuint, level: GLint):
        print('glFramebufferTexture2D')
        return gl['glFramebufferTexture2D'](target, attachment, textarget, texture, level)
    def glFramebufferRenderbuffer(target: GLenum, attachment: GLenum, renderbuffertarget: GLenum, renderbuffer: GLuint):
        print('glFramebufferRenderbuffer')
        return gl['glFramebufferRenderbuffer'](target, attachment, renderbuffertarget, renderbuffer)
    def glGenerateMipmap(target: GLenum):
        print('glGenerateMipmap')
        return gl['glGenerateMipmap'](target)
    def glBlitFramebuffer(srcX0: GLint, srcY0: GLint, srcX1: GLint, srcY1: GLint, dstX0: GLint, dstY0: GLint, dstX1: GLint, dstY1: GLint, mask: GLbitfield, filter: GLenum):
        print('glBlitFramebuffer')
        return gl['glBlitFramebuffer'](srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter)
    def glRenderbufferStorageMultisample(target: GLenum, samples: GLsizei, internalformat: GLenum, width: GLsizei, height: GLsizei):
        print('glRenderbufferStorageMultisample')
        return gl['glRenderbufferStorageMultisample'](target, samples, internalformat, width, height)
    def glFramebufferTextureLayer(target: GLenum, attachment: GLenum, texture: GLuint, level: GLint, layer: GLint):
        print('glFramebufferTextureLayer')
        return gl['glFramebufferTextureLayer'](target, attachment, texture, level, layer)
    def glMapBufferRange(target: GLenum, offset: GLintptr, length: GLsizeiptr, access: GLbitfield):
        print('glMapBufferRange')
        return gl['glMapBufferRange'](target, offset, length, access)
    def glBindVertexArray(array: GLuint):
        print('glBindVertexArray')
        return gl['glBindVertexArray'](array)
    def glDeleteVertexArrays(n: GLsizei, arrays: GLuint_star):
        print('glDeleteVertexArrays')
        return gl['glDeleteVertexArrays'](n, arrays)
    def glGenVertexArrays(n: GLsizei, arrays: GLuint_star):
        print('glGenVertexArrays')
        return gl['glGenVertexArrays'](n, arrays)
    def glDrawArraysInstanced(mode: GLenum, first: GLint, count: GLsizei, instancecount: GLsizei):
        print('glDrawArraysInstanced')
        return gl['glDrawArraysInstanced'](mode, first, count, instancecount)
    def glDrawElementsInstanced(mode: GLenum, count: GLsizei, type: GLenum, indices: void_star, instancecount: GLsizei):
        print('glDrawElementsInstanced')
        return gl['glDrawElementsInstanced'](mode, count, type, indices, instancecount)
    def glGetUniformBlockIndex(program: GLuint, uniformBlockName: GLchar_star):
        print('glGetUniformBlockIndex')
        return gl['glGetUniformBlockIndex'](program, uniformBlockName)
    def glGetActiveUniformBlockiv(program: GLuint, uniformBlockIndex: GLuint, pname: GLenum, params: GLint_star):
        print('glGetActiveUniformBlockiv')
        return gl['glGetActiveUniformBlockiv'](program, uniformBlockIndex, pname, params)
    def glGetActiveUniformBlockName(program: GLuint, uniformBlockIndex: GLuint, bufSize: GLsizei, length: GLsizei_star, uniformBlockName: GLchar_star):
        print('glGetActiveUniformBlockName')
        return gl['glGetActiveUniformBlockName'](program, uniformBlockIndex, bufSize, length, uniformBlockName)
    def glUniformBlockBinding(program: GLuint, uniformBlockIndex: GLuint, uniformBlockBinding: GLuint):
        print('glUniformBlockBinding')
        return gl['glUniformBlockBinding'](program, uniformBlockIndex, uniformBlockBinding)
    def glFenceSync(condition: GLenum, flags: GLbitfield):
        print('glFenceSync')
        return gl['glFenceSync'](condition, flags)
    def glDeleteSync(sync: GLsync):
        print('glDeleteSync')
        return gl['glDeleteSync'](sync)
    def glClientWaitSync(sync: GLsync, flags: GLbitfield, timeout: GLuint64):
        print('glClientWaitSync')
        return gl['glClientWaitSync'](sync, flags, timeout)
    def glGenSamplers(count: GLsizei, samplers: GLuint_star):
        print('glGenSamplers')
        return gl['glGenSamplers'](count, samplers)
    def glDeleteSamplers(count: GLsizei, samplers: GLuint_star):
        print('glDeleteSamplers')
        return gl['glDeleteSamplers'](count, samplers)
    def glBindSampler(unit: GLuint, sampler: GLuint):
        print('glBindSampler')
        return gl['glBindSampler'](unit, sampler)
    def glSamplerParameteri(sampler: GLuint, pname: GLenum, param: GLint):
        print('glSamplerParameteri')
        return gl['glSamplerParameteri'](sampler, pname, param)
    def glSamplerParameterf(sampler: GLuint, pname: GLenum, param: GLfloat):
        print('glSamplerParameterf')
        return gl['glSamplerParameterf'](sampler, pname, param)
    def glVertexAttribDivisor(index: GLuint, divisor: GLuint):
        print('glVertexAttribDivisor')
        return gl['glVertexAttribDivisor'](index, divisor)

    return {
        'glCullFace': proc(void, GLenum)(glCullFace),
        'glClear': proc(void, GLbitfield)(glClear),
        'glTexParameteri': proc(void, GLenum, GLenum, GLint)(glTexParameteri),
        'glTexImage2D': proc(void, GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, void_star)(glTexImage2D),
        'glDepthMask': proc(void, GLboolean)(glDepthMask),
        'glDisable': proc(void, GLenum)(glDisable),
        'glEnable': proc(void, GLenum)(glEnable),
        'glFlush': proc(void)(glFlush),
        'glDepthFunc': proc(void, GLenum)(glDepthFunc),
        'glReadBuffer': proc(void, GLenum)(glReadBuffer),
        'glReadPixels': proc(void, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void_star)(glReadPixels),
        'glGetError': proc(GLenum)(glGetError),
        'glGetIntegerv': proc(void, GLenum, GLint_star)(glGetIntegerv),
        'glGetString': proc(GLubyte_star, GLenum)(glGetString),
        'glViewport': proc(void, GLint, GLint, GLsizei, GLsizei)(glViewport),
        'glTexSubImage2D': proc(void, GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void_star)(glTexSubImage2D),
        'glBindTexture': proc(void, GLenum, GLuint)(glBindTexture),
        'glDeleteTextures': proc(void, GLsizei, GLuint_star)(glDeleteTextures),
        'glGenTextures': proc(void, GLsizei, GLuint_star)(glGenTextures),
        'glTexImage3D': proc(void, GLenum, GLint, GLint, GLsizei, GLsizei, GLsizei, GLint, GLenum, GLenum, void_star)(glTexImage3D),
        'glTexSubImage3D': proc(void, GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei, GLenum, GLenum, void_star)(glTexSubImage3D),
        'glActiveTexture': proc(void, GLenum)(glActiveTexture),
        'glBlendFuncSeparate': proc(void, GLenum, GLenum, GLenum, GLenum)(glBlendFuncSeparate),
        'glGenQueries': proc(void, GLsizei, GLuint_star)(glGenQueries),
        'glBeginQuery': proc(void, GLenum, GLuint)(glBeginQuery),
        'glEndQuery': proc(void, GLenum)(glEndQuery),
        'glGetQueryObjectuiv': proc(void, GLuint, GLenum, GLuint_star)(glGetQueryObjectuiv),
        'glBindBuffer': proc(void, GLenum, GLuint)(glBindBuffer),
        'glDeleteBuffers': proc(void, GLsizei, GLuint_star)(glDeleteBuffers),
        'glGenBuffers': proc(void, GLsizei, GLuint_star)(glGenBuffers),
        'glBufferData': proc(void, GLenum, GLsizeiptr, void_star, GLenum)(glBufferData),
        'glBufferSubData': proc(void, GLenum, GLintptr, GLsizeiptr, void_star)(glBufferSubData),
        'glUnmapBuffer': proc(GLboolean, GLenum)(glUnmapBuffer),
        'glBlendEquationSeparate': proc(void, GLenum, GLenum)(glBlendEquationSeparate),
        'glDrawBuffers': proc(void, GLsizei, GLenum_star)(glDrawBuffers),
        'glStencilOpSeparate': proc(void, GLenum, GLenum, GLenum, GLenum)(glStencilOpSeparate),
        'glStencilFuncSeparate': proc(void, GLenum, GLenum, GLint, GLuint)(glStencilFuncSeparate),
        'glStencilMaskSeparate': proc(void, GLenum, GLuint)(glStencilMaskSeparate),
        'glAttachShader': proc(void, GLuint, GLuint)(glAttachShader),
        'glCompileShader': proc(void, GLuint)(glCompileShader),
        'glCreateProgram': proc(GLuint)(glCreateProgram),
        'glCreateShader': proc(GLuint, GLenum)(glCreateShader),
        'glDeleteProgram': proc(void, GLuint)(glDeleteProgram),
        'glDeleteShader': proc(void, GLuint)(glDeleteShader),
        'glEnableVertexAttribArray': proc(void, GLuint)(glEnableVertexAttribArray),
        'glGetActiveAttrib': proc(void, GLuint, GLuint, GLsizei, GLsizei_star, GLint_star, GLenum_star, GLchar_star)(glGetActiveAttrib),
        'glGetActiveUniform': proc(void, GLuint, GLuint, GLsizei, GLsizei_star, GLint_star, GLenum_star, GLchar_star)(glGetActiveUniform),
        'glGetAttribLocation': proc(GLint, GLuint, GLchar_star)(glGetAttribLocation),
        'glGetProgramiv': proc(void, GLuint, GLenum, GLint_star)(glGetProgramiv),
        'glGetProgramInfoLog': proc(void, GLuint, GLsizei, GLsizei_star, GLchar_star)(glGetProgramInfoLog),
        'glGetShaderiv': proc(void, GLuint, GLenum, GLint_star)(glGetShaderiv),
        'glGetShaderInfoLog': proc(void, GLuint, GLsizei, GLsizei_star, GLchar_star)(glGetShaderInfoLog),
        'glGetUniformLocation': proc(GLint, GLuint, GLchar_star)(glGetUniformLocation),
        'glLinkProgram': proc(void, GLuint)(glLinkProgram),
        'glShaderSource': proc(void, GLuint, GLsizei, GLchar_star_star, GLint_star)(glShaderSource),
        'glUseProgram': proc(void, GLuint)(glUseProgram),
        'glUniform1i': proc(void, GLint, GLint)(glUniform1i),
        'glUniform1fv': proc(void, GLint, GLsizei, GLfloat_star)(glUniform1fv),
        'glUniform2fv': proc(void, GLint, GLsizei, GLfloat_star)(glUniform2fv),
        'glUniform3fv': proc(void, GLint, GLsizei, GLfloat_star)(glUniform3fv),
        'glUniform4fv': proc(void, GLint, GLsizei, GLfloat_star)(glUniform4fv),
        'glUniform1iv': proc(void, GLint, GLsizei, GLint_star)(glUniform1iv),
        'glUniform2iv': proc(void, GLint, GLsizei, GLint_star)(glUniform2iv),
        'glUniform3iv': proc(void, GLint, GLsizei, GLint_star)(glUniform3iv),
        'glUniform4iv': proc(void, GLint, GLsizei, GLint_star)(glUniform4iv),
        'glUniformMatrix2fv': proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix2fv),
        'glUniformMatrix3fv': proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix3fv),
        'glUniformMatrix4fv': proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix4fv),
        'glVertexAttribPointer': proc(void, GLuint, GLint, GLenum, GLboolean, GLsizei, void_star)(glVertexAttribPointer),
        'glUniformMatrix2x3fv': proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix2x3fv),
        'glUniformMatrix3x2fv': proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix3x2fv),
        'glUniformMatrix2x4fv': proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix2x4fv),
        'glUniformMatrix4x2fv': proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix4x2fv),
        'glUniformMatrix3x4fv': proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix3x4fv),
        'glUniformMatrix4x3fv': proc(void, GLint, GLsizei, GLboolean, GLfloat_star)(glUniformMatrix4x3fv),
        'glBindBufferRange': proc(void, GLenum, GLuint, GLuint, GLintptr, GLsizeiptr)(glBindBufferRange),
        'glVertexAttribIPointer': proc(void, GLuint, GLint, GLenum, GLsizei, void_star)(glVertexAttribIPointer),
        'glUniform1uiv': proc(void, GLint, GLsizei, GLuint_star)(glUniform1uiv),
        'glUniform2uiv': proc(void, GLint, GLsizei, GLuint_star)(glUniform2uiv),
        'glUniform3uiv': proc(void, GLint, GLsizei, GLuint_star)(glUniform3uiv),
        'glUniform4uiv': proc(void, GLint, GLsizei, GLuint_star)(glUniform4uiv),
        'glClearBufferiv': proc(void, GLenum, GLint, GLint_star)(glClearBufferiv),
        'glClearBufferuiv': proc(void, GLenum, GLint, GLuint_star)(glClearBufferuiv),
        'glClearBufferfv': proc(void, GLenum, GLint, GLfloat_star)(glClearBufferfv),
        'glClearBufferfi': proc(void, GLenum, GLint, GLfloat, GLint)(glClearBufferfi),
        'glBindRenderbuffer': proc(void, GLenum, GLuint)(glBindRenderbuffer),
        'glDeleteRenderbuffers': proc(void, GLsizei, GLuint_star)(glDeleteRenderbuffers),
        'glGenRenderbuffers': proc(void, GLsizei, GLuint_star)(glGenRenderbuffers),
        'glBindFramebuffer': proc(void, GLenum, GLuint)(glBindFramebuffer),
        'glDeleteFramebuffers': proc(void, GLsizei, GLuint_star)(glDeleteFramebuffers),
        'glGenFramebuffers': proc(void, GLsizei, GLuint_star)(glGenFramebuffers),
        'glFramebufferTexture2D': proc(void, GLenum, GLenum, GLenum, GLuint, GLint)(glFramebufferTexture2D),
        'glFramebufferRenderbuffer': proc(void, GLenum, GLenum, GLenum, GLuint)(glFramebufferRenderbuffer),
        'glGenerateMipmap': proc(void, GLenum)(glGenerateMipmap),
        'glBlitFramebuffer': proc(void, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum)(glBlitFramebuffer),
        'glRenderbufferStorageMultisample': proc(void, GLenum, GLsizei, GLenum, GLsizei, GLsizei)(glRenderbufferStorageMultisample),
        'glFramebufferTextureLayer': proc(void, GLenum, GLenum, GLuint, GLint, GLint)(glFramebufferTextureLayer),
        'glMapBufferRange': proc(void_star, GLenum, GLintptr, GLsizeiptr, GLbitfield)(glMapBufferRange),
        'glBindVertexArray': proc(void, GLuint)(glBindVertexArray),
        'glDeleteVertexArrays': proc(void, GLsizei, GLuint_star)(glDeleteVertexArrays),
        'glGenVertexArrays': proc(void, GLsizei, GLuint_star)(glGenVertexArrays),
        'glDrawArraysInstanced': proc(void, GLenum, GLint, GLsizei, GLsizei)(glDrawArraysInstanced),
        'glDrawElementsInstanced': proc(void, GLenum, GLsizei, GLenum, void_star, GLsizei)(glDrawElementsInstanced),
        'glGetUniformBlockIndex': proc(GLuint, GLuint, GLchar_star)(glGetUniformBlockIndex),
        'glGetActiveUniformBlockiv': proc(void, GLuint, GLuint, GLenum, GLint_star)(glGetActiveUniformBlockiv),
        'glGetActiveUniformBlockName': proc(void, GLuint, GLuint, GLsizei, GLsizei_star, GLchar_star)(glGetActiveUniformBlockName),
        'glUniformBlockBinding': proc(void, GLuint, GLuint, GLuint)(glUniformBlockBinding),
        'glFenceSync': proc(GLsync, GLenum, GLbitfield)(glFenceSync),
        'glDeleteSync': proc(void, GLsync)(glDeleteSync),
        'glClientWaitSync': proc(GLenum, GLsync, GLbitfield, GLuint64)(glClientWaitSync),
        'glGenSamplers': proc(void, GLsizei, GLuint_star)(glGenSamplers),
        'glDeleteSamplers': proc(void, GLsizei, GLuint_star)(glDeleteSamplers),
        'glBindSampler': proc(void, GLuint, GLuint)(glBindSampler),
        'glSamplerParameteri': proc(void, GLuint, GLenum, GLint)(glSamplerParameteri),
        'glSamplerParameterf': proc(void, GLuint, GLenum, GLfloat)(glSamplerParameterf),
        'glVertexAttribDivisor': proc(void, GLuint, GLuint)(glVertexAttribDivisor),
    }


class Loader:
    def __init__(self, items):
        self.items = items

    def load_opengl_function(self, name):
        return ctypes.cast(self.items.get(name, 0), ctypes.c_void_p).value or 0
