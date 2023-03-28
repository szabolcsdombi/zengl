#include <Python.h>

typedef signed long int GLintptr;
typedef signed long int GLsizeiptr;
typedef unsigned int GLenum;
typedef float GLfloat;
typedef int GLint;
typedef int GLsizei;
typedef unsigned long long GLuint64;
typedef unsigned int GLbitfield;
typedef unsigned int GLuint;
typedef unsigned char GLboolean;
typedef unsigned char GLubyte;
typedef char GLchar;
typedef void * GLsync;

extern void zengl_glCullFace(GLenum mode);
extern void zengl_glClear(GLbitfield mask);
extern void zengl_glTexParameteri(GLenum target, GLenum pname, GLint param);
extern void zengl_glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
extern void zengl_glDepthMask(GLboolean flag);
extern void zengl_glDisable(GLenum cap);
extern void zengl_glEnable(GLenum cap);
extern void zengl_glFlush(void);
extern void zengl_glDepthFunc(GLenum func);
extern void zengl_glReadBuffer(GLenum src);
extern void zengl_glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels);
extern GLenum zengl_glGetError(void);
extern void zengl_glGetIntegerv(GLenum pname, GLint *data);
extern const GLubyte *zengl_glGetString(GLenum name);
extern void zengl_glViewport(GLint x, GLint y, GLsizei width, GLsizei height);
extern void zengl_glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
extern void zengl_glBindTexture(GLenum target, GLuint texture);
extern void zengl_glDeleteTextures(GLsizei n, const GLuint *textures);
extern void zengl_glGenTextures(GLsizei n, GLuint *textures);
extern void zengl_glTexImage3D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
extern void zengl_glTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels);
extern void zengl_glActiveTexture(GLenum texture);
extern void zengl_glBlendFuncSeparate(GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
extern void zengl_glGenQueries(GLsizei n, GLuint *ids);
extern void zengl_glBeginQuery(GLenum target, GLuint id);
extern void zengl_glEndQuery(GLenum target);
extern void zengl_glGetQueryObjectuiv(GLuint id, GLenum pname, GLuint *params);
extern void zengl_glBindBuffer(GLenum target, GLuint buffer);
extern void zengl_glDeleteBuffers(GLsizei n, const GLuint *buffers);
extern void zengl_glGenBuffers(GLsizei n, GLuint *buffers);
extern void zengl_glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
extern void zengl_glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data);
extern GLboolean zengl_glUnmapBuffer(GLenum target);
extern void zengl_glBlendEquationSeparate(GLenum modeRGB, GLenum modeAlpha);
extern void zengl_glDrawBuffers(GLsizei n, const GLenum *bufs);
extern void zengl_glStencilOpSeparate(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
extern void zengl_glStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask);
extern void zengl_glStencilMaskSeparate(GLenum face, GLuint mask);
extern void zengl_glAttachShader(GLuint program, GLuint shader);
extern void zengl_glCompileShader(GLuint shader);
extern GLuint zengl_glCreateProgram(void);
extern GLuint zengl_glCreateShader(GLenum type);
extern void zengl_glDeleteProgram(GLuint program);
extern void zengl_glDeleteShader(GLuint shader);
extern void zengl_glEnableVertexAttribArray(GLuint index);
extern void zengl_glGetActiveAttrib(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
extern void zengl_glGetActiveUniform(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
extern GLint zengl_glGetAttribLocation(GLuint program, const GLchar *name);
extern void zengl_glGetProgramiv(GLuint program, GLenum pname, GLint *params);
extern void zengl_glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
extern void zengl_glGetShaderiv(GLuint shader, GLenum pname, GLint *params);
extern void zengl_glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
extern GLint zengl_glGetUniformLocation(GLuint program, const GLchar *name);
extern void zengl_glLinkProgram(GLuint program);
extern void zengl_glShaderSource(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
extern void zengl_glUseProgram(GLuint program);
extern void zengl_glUniform1i(GLint location, GLint v0);
extern void zengl_glUniform1fv(GLint location, GLsizei count, const GLfloat *value);
extern void zengl_glUniform2fv(GLint location, GLsizei count, const GLfloat *value);
extern void zengl_glUniform3fv(GLint location, GLsizei count, const GLfloat *value);
extern void zengl_glUniform4fv(GLint location, GLsizei count, const GLfloat *value);
extern void zengl_glUniform1iv(GLint location, GLsizei count, const GLint *value);
extern void zengl_glUniform2iv(GLint location, GLsizei count, const GLint *value);
extern void zengl_glUniform3iv(GLint location, GLsizei count, const GLint *value);
extern void zengl_glUniform4iv(GLint location, GLsizei count, const GLint *value);
extern void zengl_glUniformMatrix2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
extern void zengl_glUniformMatrix3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
extern void zengl_glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
extern void zengl_glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
extern void zengl_glUniformMatrix2x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
extern void zengl_glUniformMatrix3x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
extern void zengl_glUniformMatrix2x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
extern void zengl_glUniformMatrix4x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
extern void zengl_glUniformMatrix3x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
extern void zengl_glUniformMatrix4x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
extern void zengl_glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);
extern void zengl_glVertexAttribIPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer);
extern void zengl_glUniform1uiv(GLint location, GLsizei count, const GLuint *value);
extern void zengl_glUniform2uiv(GLint location, GLsizei count, const GLuint *value);
extern void zengl_glUniform3uiv(GLint location, GLsizei count, const GLuint *value);
extern void zengl_glUniform4uiv(GLint location, GLsizei count, const GLuint *value);
extern void zengl_glClearBufferiv(GLenum buffer, GLint drawbuffer, const GLint *value);
extern void zengl_glClearBufferuiv(GLenum buffer, GLint drawbuffer, const GLuint *value);
extern void zengl_glClearBufferfv(GLenum buffer, GLint drawbuffer, const GLfloat *value);
extern void zengl_glClearBufferfi(GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil);
extern void zengl_glBindRenderbuffer(GLenum target, GLuint renderbuffer);
extern void zengl_glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers);
extern void zengl_glGenRenderbuffers(GLsizei n, GLuint *renderbuffers);
extern void zengl_glBindFramebuffer(GLenum target, GLuint framebuffer);
extern void zengl_glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers);
extern void zengl_glGenFramebuffers(GLsizei n, GLuint *framebuffers);
extern void zengl_glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
extern void zengl_glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
extern void zengl_glGenerateMipmap(GLenum target);
extern void zengl_glBlitFramebuffer(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
extern void zengl_glRenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
extern void zengl_glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer);
extern void *zengl_glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
extern void zengl_glBindVertexArray(GLuint array);
extern void zengl_glDeleteVertexArrays(GLsizei n, const GLuint *arrays);
extern void zengl_glGenVertexArrays(GLsizei n, GLuint *arrays);
extern void zengl_glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
extern void zengl_glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount);
extern GLuint zengl_glGetUniformBlockIndex(GLuint program, const GLchar *uniformBlockName);
extern void zengl_glGetActiveUniformBlockiv(GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params);
extern void zengl_glGetActiveUniformBlockName(GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformBlockName);
extern void zengl_glUniformBlockBinding(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
extern GLsync zengl_glFenceSync(GLenum condition, GLbitfield flags);
extern void zengl_glDeleteSync(GLsync sync);
extern GLenum zengl_glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout);
extern void zengl_glGenSamplers(GLsizei count, GLuint *samplers);
extern void zengl_glDeleteSamplers(GLsizei count, const GLuint *samplers);
extern void zengl_glBindSampler(GLuint unit, GLuint sampler);
extern void zengl_glSamplerParameteri(GLuint sampler, GLenum pname, GLint param);
extern void zengl_glSamplerParameterf(GLuint sampler, GLenum pname, GLfloat param);
extern void zengl_glVertexAttribDivisor(GLuint index, GLuint divisor);

void impl_glCullFace(GLenum mode) {
    zengl_glCullFace(mode);
}

void impl_glClear(GLbitfield mask) {
    zengl_glClear(mask);
}

void impl_glTexParameteri(GLenum target, GLenum pname, GLint param) {
    zengl_glTexParameteri(target, pname, param);
}

void impl_glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels) {
    zengl_glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
}

void impl_glDepthMask(GLboolean flag) {
    zengl_glDepthMask(flag);
}

void impl_glDisable(GLenum cap) {
    zengl_glDisable(cap);
}

void impl_glEnable(GLenum cap) {
    zengl_glEnable(cap);
}

void impl_glFlush(void) {
    zengl_glFlush();
}

void impl_glDepthFunc(GLenum func) {
    zengl_glDepthFunc(func);
}

void impl_glReadBuffer(GLenum src) {
    zengl_glReadBuffer(src);
}

void impl_glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels) {
    zengl_glReadPixels(x, y, width, height, format, type, pixels);
}

GLenum impl_glGetError(void) {
    return zengl_glGetError();
}

void impl_glGetIntegerv(GLenum pname, GLint *data) {
    zengl_glGetIntegerv(pname, data);
}

const GLubyte *impl_glGetString(GLenum name) {
    return zengl_glGetString(name);
}

void impl_glViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    zengl_glViewport(x, y, width, height);
}

void impl_glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) {
    zengl_glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels);
}

void impl_glBindTexture(GLenum target, GLuint texture) {
    zengl_glBindTexture(target, texture);
}

void impl_glDeleteTextures(GLsizei n, const GLuint *textures) {
    zengl_glDeleteTextures(n, textures);
}

void impl_glGenTextures(GLsizei n, GLuint *textures) {
    zengl_glGenTextures(n, textures);
}

void impl_glTexImage3D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels) {
    zengl_glTexImage3D(target, level, internalformat, width, height, depth, border, format, type, pixels);
}

void impl_glTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels) {
    zengl_glTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
}

void impl_glActiveTexture(GLenum texture) {
    zengl_glActiveTexture(texture);
}

void impl_glBlendFuncSeparate(GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha) {
    zengl_glBlendFuncSeparate(sfactorRGB, dfactorRGB, sfactorAlpha, dfactorAlpha);
}

void impl_glGenQueries(GLsizei n, GLuint *ids) {
    zengl_glGenQueries(n, ids);
}

void impl_glBeginQuery(GLenum target, GLuint id) {
    zengl_glBeginQuery(target, id);
}

void impl_glEndQuery(GLenum target) {
    zengl_glEndQuery(target);
}

void impl_glGetQueryObjectuiv(GLuint id, GLenum pname, GLuint *params) {
    zengl_glGetQueryObjectuiv(id, pname, params);
}

void impl_glBindBuffer(GLenum target, GLuint buffer) {
    zengl_glBindBuffer(target, buffer);
}

void impl_glDeleteBuffers(GLsizei n, const GLuint *buffers) {
    zengl_glDeleteBuffers(n, buffers);
}

void impl_glGenBuffers(GLsizei n, GLuint *buffers) {
    zengl_glGenBuffers(n, buffers);
}

void impl_glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
    zengl_glBufferData(target, size, data, usage);
}

void impl_glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
    zengl_glBufferSubData(target, offset, size, data);
}

GLboolean impl_glUnmapBuffer(GLenum target) {
    return zengl_glUnmapBuffer(target);
}

void impl_glBlendEquationSeparate(GLenum modeRGB, GLenum modeAlpha) {
    zengl_glBlendEquationSeparate(modeRGB, modeAlpha);
}

void impl_glDrawBuffers(GLsizei n, const GLenum *bufs) {
    zengl_glDrawBuffers(n, bufs);
}

void impl_glStencilOpSeparate(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass) {
    zengl_glStencilOpSeparate(face, sfail, dpfail, dppass);
}

void impl_glStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask) {
    zengl_glStencilFuncSeparate(face, func, ref, mask);
}

void impl_glStencilMaskSeparate(GLenum face, GLuint mask) {
    zengl_glStencilMaskSeparate(face, mask);
}

void impl_glAttachShader(GLuint program, GLuint shader) {
    zengl_glAttachShader(program, shader);
}

void impl_glCompileShader(GLuint shader) {
    zengl_glCompileShader(shader);
}

GLuint impl_glCreateProgram(void) {
    return zengl_glCreateProgram();
}

GLuint impl_glCreateShader(GLenum type) {
    return zengl_glCreateShader(type);
}

void impl_glDeleteProgram(GLuint program) {
    zengl_glDeleteProgram(program);
}

void impl_glDeleteShader(GLuint shader) {
    zengl_glDeleteShader(shader);
}

void impl_glEnableVertexAttribArray(GLuint index) {
    zengl_glEnableVertexAttribArray(index);
}

void impl_glGetActiveAttrib(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name) {
    zengl_glGetActiveAttrib(program, index, bufSize, length, size, type, name);
}

void impl_glGetActiveUniform(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name) {
    zengl_glGetActiveUniform(program, index, bufSize, length, size, type, name);
}

GLint impl_glGetAttribLocation(GLuint program, const GLchar *name) {
    return zengl_glGetAttribLocation(program, name);
}

void impl_glGetProgramiv(GLuint program, GLenum pname, GLint *params) {
    zengl_glGetProgramiv(program, pname, params);
}

void impl_glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
    zengl_glGetProgramInfoLog(program, bufSize, length, infoLog);
}

void impl_glGetShaderiv(GLuint shader, GLenum pname, GLint *params) {
    zengl_glGetShaderiv(shader, pname, params);
}

void impl_glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
    zengl_glGetShaderInfoLog(shader, bufSize, length, infoLog);
}

GLint impl_glGetUniformLocation(GLuint program, const GLchar *name) {
    return zengl_glGetUniformLocation(program, name);
}

void impl_glLinkProgram(GLuint program) {
    zengl_glLinkProgram(program);
}

void impl_glShaderSource(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length) {
    zengl_glShaderSource(shader, count, string, length);
}

void impl_glUseProgram(GLuint program) {
    zengl_glUseProgram(program);
}

void impl_glUniform1i(GLint location, GLint v0) {
    zengl_glUniform1i(location, v0);
}

void impl_glUniform1fv(GLint location, GLsizei count, const GLfloat *value) {
    zengl_glUniform1fv(location, count, value);
}

void impl_glUniform2fv(GLint location, GLsizei count, const GLfloat *value) {
    zengl_glUniform2fv(location, count, value);
}

void impl_glUniform3fv(GLint location, GLsizei count, const GLfloat *value) {
    zengl_glUniform3fv(location, count, value);
}

void impl_glUniform4fv(GLint location, GLsizei count, const GLfloat *value) {
    zengl_glUniform4fv(location, count, value);
}

void impl_glUniform1iv(GLint location, GLsizei count, const GLint *value) {
    zengl_glUniform1iv(location, count, value);
}

void impl_glUniform2iv(GLint location, GLsizei count, const GLint *value) {
    zengl_glUniform2iv(location, count, value);
}

void impl_glUniform3iv(GLint location, GLsizei count, const GLint *value) {
    zengl_glUniform3iv(location, count, value);
}

void impl_glUniform4iv(GLint location, GLsizei count, const GLint *value) {
    zengl_glUniform4iv(location, count, value);
}

void impl_glUniformMatrix2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    zengl_glUniformMatrix2fv(location, count, transpose, value);
}

void impl_glUniformMatrix3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    zengl_glUniformMatrix3fv(location, count, transpose, value);
}

void impl_glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    zengl_glUniformMatrix4fv(location, count, transpose, value);
}

void impl_glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
    zengl_glVertexAttribPointer(index, size, type, normalized, stride, pointer);
}

void impl_glUniformMatrix2x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    zengl_glUniformMatrix2x3fv(location, count, transpose, value);
}

void impl_glUniformMatrix3x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    zengl_glUniformMatrix3x2fv(location, count, transpose, value);
}

void impl_glUniformMatrix2x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    zengl_glUniformMatrix2x4fv(location, count, transpose, value);
}

void impl_glUniformMatrix4x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    zengl_glUniformMatrix4x2fv(location, count, transpose, value);
}

void impl_glUniformMatrix3x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    zengl_glUniformMatrix3x4fv(location, count, transpose, value);
}

void impl_glUniformMatrix4x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    zengl_glUniformMatrix4x3fv(location, count, transpose, value);
}

void impl_glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) {
    zengl_glBindBufferRange(target, index, buffer, offset, size);
}

void impl_glVertexAttribIPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer) {
    zengl_glVertexAttribIPointer(index, size, type, stride, pointer);
}

void impl_glUniform1uiv(GLint location, GLsizei count, const GLuint *value) {
    zengl_glUniform1uiv(location, count, value);
}

void impl_glUniform2uiv(GLint location, GLsizei count, const GLuint *value) {
    zengl_glUniform2uiv(location, count, value);
}

void impl_glUniform3uiv(GLint location, GLsizei count, const GLuint *value) {
    zengl_glUniform3uiv(location, count, value);
}

void impl_glUniform4uiv(GLint location, GLsizei count, const GLuint *value) {
    zengl_glUniform4uiv(location, count, value);
}

void impl_glClearBufferiv(GLenum buffer, GLint drawbuffer, const GLint *value) {
    zengl_glClearBufferiv(buffer, drawbuffer, value);
}

void impl_glClearBufferuiv(GLenum buffer, GLint drawbuffer, const GLuint *value) {
    zengl_glClearBufferuiv(buffer, drawbuffer, value);
}

void impl_glClearBufferfv(GLenum buffer, GLint drawbuffer, const GLfloat *value) {
    zengl_glClearBufferfv(buffer, drawbuffer, value);
}

void impl_glClearBufferfi(GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil) {
    zengl_glClearBufferfi(buffer, drawbuffer, depth, stencil);
}

void impl_glBindRenderbuffer(GLenum target, GLuint renderbuffer) {
    zengl_glBindRenderbuffer(target, renderbuffer);
}

void impl_glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers) {
    zengl_glDeleteRenderbuffers(n, renderbuffers);
}

void impl_glGenRenderbuffers(GLsizei n, GLuint *renderbuffers) {
    zengl_glGenRenderbuffers(n, renderbuffers);
}

void impl_glBindFramebuffer(GLenum target, GLuint framebuffer) {
    zengl_glBindFramebuffer(target, framebuffer);
}

void impl_glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers) {
    zengl_glDeleteFramebuffers(n, framebuffers);
}

void impl_glGenFramebuffers(GLsizei n, GLuint *framebuffers) {
    zengl_glGenFramebuffers(n, framebuffers);
}

void impl_glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    zengl_glFramebufferTexture2D(target, attachment, textarget, texture, level);
}

void impl_glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
    zengl_glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer);
}

void impl_glGenerateMipmap(GLenum target) {
    zengl_glGenerateMipmap(target);
}

void impl_glBlitFramebuffer(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) {
    zengl_glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
}

void impl_glRenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height) {
    zengl_glRenderbufferStorageMultisample(target, samples, internalformat, width, height);
}

void impl_glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer) {
    zengl_glFramebufferTextureLayer(target, attachment, texture, level, layer);
}

void *impl_glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    return zengl_glMapBufferRange(target, offset, length, access);
}

void impl_glBindVertexArray(GLuint array) {
    zengl_glBindVertexArray(array);
}

void impl_glDeleteVertexArrays(GLsizei n, const GLuint *arrays) {
    zengl_glDeleteVertexArrays(n, arrays);
}

void impl_glGenVertexArrays(GLsizei n, GLuint *arrays) {
    zengl_glGenVertexArrays(n, arrays);
}

void impl_glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
    zengl_glDrawArraysInstanced(mode, first, count, instancecount);
}

void impl_glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount) {
    zengl_glDrawElementsInstanced(mode, count, type, indices, instancecount);
}

GLuint impl_glGetUniformBlockIndex(GLuint program, const GLchar *uniformBlockName) {
    return zengl_glGetUniformBlockIndex(program, uniformBlockName);
}

void impl_glGetActiveUniformBlockiv(GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params) {
    zengl_glGetActiveUniformBlockiv(program, uniformBlockIndex, pname, params);
}

void impl_glGetActiveUniformBlockName(GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformBlockName) {
    zengl_glGetActiveUniformBlockName(program, uniformBlockIndex, bufSize, length, uniformBlockName);
}

void impl_glUniformBlockBinding(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding) {
    zengl_glUniformBlockBinding(program, uniformBlockIndex, uniformBlockBinding);
}

GLsync impl_glFenceSync(GLenum condition, GLbitfield flags) {
    return zengl_glFenceSync(condition, flags);
}

void impl_glDeleteSync(GLsync sync) {
    zengl_glDeleteSync(sync);
}

GLenum impl_glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
    return zengl_glClientWaitSync(sync, flags, timeout);
}

void impl_glGenSamplers(GLsizei count, GLuint *samplers) {
    zengl_glGenSamplers(count, samplers);
}

void impl_glDeleteSamplers(GLsizei count, const GLuint *samplers) {
    zengl_glDeleteSamplers(count, samplers);
}

void impl_glBindSampler(GLuint unit, GLuint sampler) {
    zengl_glBindSampler(unit, sampler);
}

void impl_glSamplerParameteri(GLuint sampler, GLenum pname, GLint param) {
    zengl_glSamplerParameteri(sampler, pname, param);
}

void impl_glSamplerParameterf(GLuint sampler, GLenum pname, GLfloat param) {
    zengl_glSamplerParameterf(sampler, pname, param);
}

void impl_glVertexAttribDivisor(GLuint index, GLuint divisor) {
    zengl_glVertexAttribDivisor(index, divisor);
}

PyObject * lookup;

static void fn(const char * name, void * ptr) {
    PyDict_SetItemString(lookup, name, PyLong_FromVoidPtr(ptr));
}

static PyObject * load_opengl_function(PyObject * self, PyObject * arg) {
    PyObject * res = PyDict_GetItem(lookup, arg);
    if (!res) {
        return NULL;
    }
    Py_INCREF(res);
    return res;
    zengl_glCullFace(1);
}

PyMethodDef module_methods[] = {
    {"load_opengl_function", (PyCFunction)load_opengl_function, METH_O, NULL},
    {},
};

PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "webgl", NULL, -1, module_methods};

extern PyObject * PyInit_webgl() {
    PyObject * module = PyModule_Create(&module_def);
    lookup = PyDict_New();
    fn("glCullFace", (void *)impl_glCullFace);
    fn("glClear", (void *)impl_glClear);
    fn("glTexParameteri", (void *)impl_glTexParameteri);
    fn("glTexImage2D", (void *)impl_glTexImage2D);
    fn("glDepthMask", (void *)impl_glDepthMask);
    fn("glDisable", (void *)impl_glDisable);
    fn("glEnable", (void *)impl_glEnable);
    fn("glFlush", (void *)impl_glFlush);
    fn("glDepthFunc", (void *)impl_glDepthFunc);
    fn("glReadBuffer", (void *)impl_glReadBuffer);
    fn("glReadPixels", (void *)impl_glReadPixels);
    fn("glGetError", (void *)impl_glGetError);
    fn("glGetIntegerv", (void *)impl_glGetIntegerv);
    fn("glGetString", (void *)impl_glGetString);
    fn("glViewport", (void *)impl_glViewport);
    fn("glTexSubImage2D", (void *)impl_glTexSubImage2D);
    fn("glBindTexture", (void *)impl_glBindTexture);
    fn("glDeleteTextures", (void *)impl_glDeleteTextures);
    fn("glGenTextures", (void *)impl_glGenTextures);
    fn("glTexImage3D", (void *)impl_glTexImage3D);
    fn("glTexSubImage3D", (void *)impl_glTexSubImage3D);
    fn("glActiveTexture", (void *)impl_glActiveTexture);
    fn("glBlendFuncSeparate", (void *)impl_glBlendFuncSeparate);
    fn("glGenQueries", (void *)impl_glGenQueries);
    fn("glBeginQuery", (void *)impl_glBeginQuery);
    fn("glEndQuery", (void *)impl_glEndQuery);
    fn("glGetQueryObjectuiv", (void *)impl_glGetQueryObjectuiv);
    fn("glBindBuffer", (void *)impl_glBindBuffer);
    fn("glDeleteBuffers", (void *)impl_glDeleteBuffers);
    fn("glGenBuffers", (void *)impl_glGenBuffers);
    fn("glBufferData", (void *)impl_glBufferData);
    fn("glBufferSubData", (void *)impl_glBufferSubData);
    fn("glUnmapBuffer", (void *)impl_glUnmapBuffer);
    fn("glBlendEquationSeparate", (void *)impl_glBlendEquationSeparate);
    fn("glDrawBuffers", (void *)impl_glDrawBuffers);
    fn("glStencilOpSeparate", (void *)impl_glStencilOpSeparate);
    fn("glStencilFuncSeparate", (void *)impl_glStencilFuncSeparate);
    fn("glStencilMaskSeparate", (void *)impl_glStencilMaskSeparate);
    fn("glAttachShader", (void *)impl_glAttachShader);
    fn("glCompileShader", (void *)impl_glCompileShader);
    fn("glCreateProgram", (void *)impl_glCreateProgram);
    fn("glCreateShader", (void *)impl_glCreateShader);
    fn("glDeleteProgram", (void *)impl_glDeleteProgram);
    fn("glDeleteShader", (void *)impl_glDeleteShader);
    fn("glEnableVertexAttribArray", (void *)impl_glEnableVertexAttribArray);
    fn("glGetActiveAttrib", (void *)impl_glGetActiveAttrib);
    fn("glGetActiveUniform", (void *)impl_glGetActiveUniform);
    fn("glGetAttribLocation", (void *)impl_glGetAttribLocation);
    fn("glGetProgramiv", (void *)impl_glGetProgramiv);
    fn("glGetProgramInfoLog", (void *)impl_glGetProgramInfoLog);
    fn("glGetShaderiv", (void *)impl_glGetShaderiv);
    fn("glGetShaderInfoLog", (void *)impl_glGetShaderInfoLog);
    fn("glGetUniformLocation", (void *)impl_glGetUniformLocation);
    fn("glLinkProgram", (void *)impl_glLinkProgram);
    fn("glShaderSource", (void *)impl_glShaderSource);
    fn("glUseProgram", (void *)impl_glUseProgram);
    fn("glUniform1i", (void *)impl_glUniform1i);
    fn("glUniform1fv", (void *)impl_glUniform1fv);
    fn("glUniform2fv", (void *)impl_glUniform2fv);
    fn("glUniform3fv", (void *)impl_glUniform3fv);
    fn("glUniform4fv", (void *)impl_glUniform4fv);
    fn("glUniform1iv", (void *)impl_glUniform1iv);
    fn("glUniform2iv", (void *)impl_glUniform2iv);
    fn("glUniform3iv", (void *)impl_glUniform3iv);
    fn("glUniform4iv", (void *)impl_glUniform4iv);
    fn("glUniformMatrix2fv", (void *)impl_glUniformMatrix2fv);
    fn("glUniformMatrix3fv", (void *)impl_glUniformMatrix3fv);
    fn("glUniformMatrix4fv", (void *)impl_glUniformMatrix4fv);
    fn("glVertexAttribPointer", (void *)impl_glVertexAttribPointer);
    fn("glUniformMatrix2x3fv", (void *)impl_glUniformMatrix2x3fv);
    fn("glUniformMatrix3x2fv", (void *)impl_glUniformMatrix3x2fv);
    fn("glUniformMatrix2x4fv", (void *)impl_glUniformMatrix2x4fv);
    fn("glUniformMatrix4x2fv", (void *)impl_glUniformMatrix4x2fv);
    fn("glUniformMatrix3x4fv", (void *)impl_glUniformMatrix3x4fv);
    fn("glUniformMatrix4x3fv", (void *)impl_glUniformMatrix4x3fv);
    fn("glBindBufferRange", (void *)impl_glBindBufferRange);
    fn("glVertexAttribIPointer", (void *)impl_glVertexAttribIPointer);
    fn("glUniform1uiv", (void *)impl_glUniform1uiv);
    fn("glUniform2uiv", (void *)impl_glUniform2uiv);
    fn("glUniform3uiv", (void *)impl_glUniform3uiv);
    fn("glUniform4uiv", (void *)impl_glUniform4uiv);
    fn("glClearBufferiv", (void *)impl_glClearBufferiv);
    fn("glClearBufferuiv", (void *)impl_glClearBufferuiv);
    fn("glClearBufferfv", (void *)impl_glClearBufferfv);
    fn("glClearBufferfi", (void *)impl_glClearBufferfi);
    fn("glBindRenderbuffer", (void *)impl_glBindRenderbuffer);
    fn("glDeleteRenderbuffers", (void *)impl_glDeleteRenderbuffers);
    fn("glGenRenderbuffers", (void *)impl_glGenRenderbuffers);
    fn("glBindFramebuffer", (void *)impl_glBindFramebuffer);
    fn("glDeleteFramebuffers", (void *)impl_glDeleteFramebuffers);
    fn("glGenFramebuffers", (void *)impl_glGenFramebuffers);
    fn("glFramebufferTexture2D", (void *)impl_glFramebufferTexture2D);
    fn("glFramebufferRenderbuffer", (void *)impl_glFramebufferRenderbuffer);
    fn("glGenerateMipmap", (void *)impl_glGenerateMipmap);
    fn("glBlitFramebuffer", (void *)impl_glBlitFramebuffer);
    fn("glRenderbufferStorageMultisample", (void *)impl_glRenderbufferStorageMultisample);
    fn("glFramebufferTextureLayer", (void *)impl_glFramebufferTextureLayer);
    fn("glMapBufferRange", (void *)impl_glMapBufferRange);
    fn("glBindVertexArray", (void *)impl_glBindVertexArray);
    fn("glDeleteVertexArrays", (void *)impl_glDeleteVertexArrays);
    fn("glGenVertexArrays", (void *)impl_glGenVertexArrays);
    fn("glDrawArraysInstanced", (void *)impl_glDrawArraysInstanced);
    fn("glDrawElementsInstanced", (void *)impl_glDrawElementsInstanced);
    fn("glGetUniformBlockIndex", (void *)impl_glGetUniformBlockIndex);
    fn("glGetActiveUniformBlockiv", (void *)impl_glGetActiveUniformBlockiv);
    fn("glGetActiveUniformBlockName", (void *)impl_glGetActiveUniformBlockName);
    fn("glUniformBlockBinding", (void *)impl_glUniformBlockBinding);
    fn("glFenceSync", (void *)impl_glFenceSync);
    fn("glDeleteSync", (void *)impl_glDeleteSync);
    fn("glClientWaitSync", (void *)impl_glClientWaitSync);
    fn("glGenSamplers", (void *)impl_glGenSamplers);
    fn("glDeleteSamplers", (void *)impl_glDeleteSamplers);
    fn("glBindSampler", (void *)impl_glBindSampler);
    fn("glSamplerParameteri", (void *)impl_glSamplerParameteri);
    fn("glSamplerParameterf", (void *)impl_glSamplerParameterf);
    fn("glVertexAttribDivisor", (void *)impl_glVertexAttribDivisor);
    return module;
}
