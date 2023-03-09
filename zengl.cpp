#include <Python.h>
#include <structmember.h>

#define MAX_ATTACHMENTS 16
#define MAX_UNIFORM_BUFFER_BINDINGS 16
#define MAX_SAMPLER_BINDINGS 64

#ifdef _WIN32
#define GLAPI __stdcall
#else
#define GLAPI
#endif

#ifdef _WIN64
typedef signed long long int GLintptr;
typedef signed long long int GLsizeiptr;
#else
typedef signed long int GLintptr;
typedef signed long int GLsizeiptr;
#endif

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

#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_POINTS 0x0000
#define GL_LINES 0x0001
#define GL_LINE_LOOP 0x0002
#define GL_LINE_STRIP 0x0003
#define GL_TRIANGLES 0x0004
#define GL_TRIANGLE_STRIP 0x0005
#define GL_TRIANGLE_FAN 0x0006
#define GL_FRONT 0x0404
#define GL_BACK 0x0405
#define GL_CULL_FACE 0x0B44
#define GL_DEPTH_TEST 0x0B71
#define GL_STENCIL_TEST 0x0B90
#define GL_BLEND 0x0BE2
#define GL_TEXTURE_2D 0x0DE1
#define GL_TEXTURE_BORDER_COLOR 0x1004
#define GL_BYTE 0x1400
#define GL_UNSIGNED_BYTE 0x1401
#define GL_SHORT 0x1402
#define GL_UNSIGNED_SHORT 0x1403
#define GL_INT 0x1404
#define GL_UNSIGNED_INT 0x1405
#define GL_FLOAT 0x1406
#define GL_COLOR 0x1800
#define GL_DEPTH 0x1801
#define GL_STENCIL 0x1802
#define GL_STENCIL_INDEX 0x1901
#define GL_DEPTH_COMPONENT 0x1902
#define GL_RED 0x1903
#define GL_RGBA 0x1908
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_NEAREST 0x2600
#define GL_LINEAR 0x2601
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_WRAP_S 0x2802
#define GL_TEXTURE_WRAP_T 0x2803
#define GL_POLYGON_OFFSET_POINT 0x2A01
#define GL_POLYGON_OFFSET_LINE 0x2A02
#define GL_POLYGON_OFFSET_FILL 0x8037
#define GL_RGBA8 0x8058
#define GL_TEXTURE_WRAP_R 0x8072
#define GL_BGRA 0x80E1
#define GL_TEXTURE_MIN_LOD 0x813A
#define GL_TEXTURE_MAX_LOD 0x813B
#define GL_TEXTURE_BASE_LEVEL 0x813C
#define GL_TEXTURE_MAX_LEVEL 0x813D
#define GL_TEXTURE0 0x84C0
#define GL_TEXTURE_CUBE_MAP 0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X 0x8515
#define GL_DEPTH_COMPONENT16 0x81A5
#define GL_DEPTH_COMPONENT24 0x81A6
#define GL_TEXTURE_LOD_BIAS 0x8501
#define GL_TEXTURE_COMPARE_MODE 0x884C
#define GL_TEXTURE_COMPARE_FUNC 0x884D
#define GL_QUERY_RESULT 0x8866
#define GL_ARRAY_BUFFER 0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_MAX_DRAW_BUFFERS 0x8824
#define GL_MAX_VERTEX_ATTRIBS 0x8869
#define GL_MAX_TEXTURE_IMAGE_UNITS 0x8872
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS 0x8B4D
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_ACTIVE_UNIFORMS 0x8B86
#define GL_ACTIVE_ATTRIBUTES 0x8B89
#define GL_SHADING_LANGUAGE_VERSION 0x8B8C
#define GL_SRGB8_ALPHA8 0x8C43
#define GL_RGBA32F 0x8814
#define GL_RGBA16F 0x881A
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#define GL_RGBA32UI 0x8D70
#define GL_RGBA16UI 0x8D76
#define GL_RGBA8UI 0x8D7C
#define GL_RGBA32I 0x8D82
#define GL_RGBA16I 0x8D88
#define GL_RGBA8I 0x8D8E
#define GL_RED_INTEGER 0x8D94
#define GL_RGBA_INTEGER 0x8D99
#define GL_DEPTH_COMPONENT32F 0x8CAC
#define GL_DEPTH_STENCIL_ATTACHMENT 0x821A
#define GL_DEPTH_STENCIL 0x84F9
#define GL_UNSIGNED_INT_24_8 0x84FA
#define GL_DEPTH24_STENCIL8 0x88F0
#define GL_READ_FRAMEBUFFER 0x8CA8
#define GL_DRAW_FRAMEBUFFER 0x8CA9
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_DEPTH_ATTACHMENT 0x8D00
#define GL_STENCIL_ATTACHMENT 0x8D20
#define GL_FRAMEBUFFER 0x8D40
#define GL_RENDERBUFFER 0x8D41
#define GL_STENCIL_INDEX8 0x8D48
#define GL_MAX_SAMPLES 0x8D57
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#define GL_HALF_FLOAT 0x140B
#define GL_MAP_READ_BIT 0x0001
#define GL_MAP_WRITE_BIT 0x0002
#define GL_MAP_INVALIDATE_RANGE_BIT 0x0004
#define GL_RG 0x8227
#define GL_RG_INTEGER 0x8228
#define GL_R8 0x8229
#define GL_RG8 0x822B
#define GL_R16F 0x822D
#define GL_R32F 0x822E
#define GL_RG16F 0x822F
#define GL_RG32F 0x8230
#define GL_R8I 0x8231
#define GL_R8UI 0x8232
#define GL_R16I 0x8233
#define GL_R16UI 0x8234
#define GL_R32I 0x8235
#define GL_R32UI 0x8236
#define GL_RG8I 0x8237
#define GL_RG8UI 0x8238
#define GL_RG16I 0x8239
#define GL_RG16UI 0x823A
#define GL_RG32I 0x823B
#define GL_RG32UI 0x823C
#define GL_R8_SNORM 0x8F94
#define GL_RG8_SNORM 0x8F95
#define GL_RGBA8_SNORM 0x8F97
#define GL_PRIMITIVE_RESTART 0x8F9D
#define GL_UNIFORM_BUFFER 0x8A11
#define GL_MAX_COMBINED_UNIFORM_BLOCKS 0x8A2E
#define GL_MAX_UNIFORM_BUFFER_BINDINGS 0x8A2F
#define GL_MAX_UNIFORM_BLOCK_SIZE 0x8A30
#define GL_ACTIVE_UNIFORM_BLOCKS 0x8A36
#define GL_UNIFORM_BLOCK_DATA_SIZE 0x8A40
#define GL_PROGRAM_POINT_SIZE 0x8642
#define GL_TEXTURE_CUBE_MAP_SEAMLESS 0x884F
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#define GL_TIMEOUT_IGNORED 0xFFFFFFFFFFFFFFFFull
#define GL_SYNC_FLUSH_COMMANDS_BIT 0x00000001
#define GL_TIME_ELAPSED 0x88BF
#define GL_PRIMITIVE_RESTART_FIXED_INDEX 0x8D69
#define GL_TEXTURE_MAX_ANISOTROPY 0x84FE

struct GLMethods {
    void (GLAPI * CullFace)(GLenum mode);
    void (GLAPI * FrontFace)(GLenum mode);
    void (GLAPI * LineWidth)(GLfloat width);
    void (GLAPI * TexParameteri)(GLenum target, GLenum pname, GLint param);
    void (GLAPI * TexImage2D)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
    void (GLAPI * DepthMask)(GLboolean flag);
    void (GLAPI * Disable)(GLenum cap);
    void (GLAPI * Enable)(GLenum cap);
    void (GLAPI * Flush)(void);
    void (GLAPI * DepthFunc)(GLenum func);
    void (GLAPI * ReadBuffer)(GLenum src);
    void (GLAPI * ReadPixels)(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels);
    GLenum (GLAPI * GetError)(void);
    void (GLAPI * GetIntegerv)(GLenum pname, GLint *data);
    const GLubyte *(GLAPI * GetString)(GLenum name);
    void (GLAPI * Viewport)(GLint x, GLint y, GLsizei width, GLsizei height);
    void (GLAPI * PolygonOffset)(GLfloat factor, GLfloat units);
    void (GLAPI * TexSubImage2D)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
    void (GLAPI * BindTexture)(GLenum target, GLuint texture);
    void (GLAPI * DeleteTextures)(GLsizei n, const GLuint *textures);
    void (GLAPI * GenTextures)(GLsizei n, GLuint *textures);
    void (GLAPI * TexImage3D)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
    void (GLAPI * TexSubImage3D)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels);
    void (GLAPI * ActiveTexture)(GLenum texture);
    void (GLAPI * BlendFuncSeparate)(GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
    void (GLAPI * GenQueries)(GLsizei n, GLuint *ids);
    void (GLAPI * BeginQuery)(GLenum target, GLuint id);
    void (GLAPI * EndQuery)(GLenum target);
    void (GLAPI * GetQueryObjectuiv)(GLuint id, GLenum pname, GLuint *params);
    void (GLAPI * BindBuffer)(GLenum target, GLuint buffer);
    void (GLAPI * DeleteBuffers)(GLsizei n, const GLuint *buffers);
    void (GLAPI * GenBuffers)(GLsizei n, GLuint *buffers);
    void (GLAPI * BufferData)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
    void (GLAPI * BufferSubData)(GLenum target, GLintptr offset, GLsizeiptr size, const void *data);
    GLboolean (GLAPI * UnmapBuffer)(GLenum target);
    void (GLAPI * BlendEquationSeparate)(GLenum modeRGB, GLenum modeAlpha);
    void (GLAPI * DrawBuffers)(GLsizei n, const GLenum *bufs);
    void (GLAPI * StencilOpSeparate)(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
    void (GLAPI * StencilFuncSeparate)(GLenum face, GLenum func, GLint ref, GLuint mask);
    void (GLAPI * StencilMaskSeparate)(GLenum face, GLuint mask);
    void (GLAPI * AttachShader)(GLuint program, GLuint shader);
    void (GLAPI * CompileShader)(GLuint shader);
    GLuint (GLAPI * CreateProgram)(void);
    GLuint (GLAPI * CreateShader)(GLenum type);
    void (GLAPI * DeleteProgram)(GLuint program);
    void (GLAPI * DeleteShader)(GLuint shader);
    void (GLAPI * EnableVertexAttribArray)(GLuint index);
    void (GLAPI * GetActiveAttrib)(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
    void (GLAPI * GetActiveUniform)(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
    GLint (GLAPI * GetAttribLocation)(GLuint program, const GLchar *name);
    void (GLAPI * GetProgramiv)(GLuint program, GLenum pname, GLint *params);
    void (GLAPI * GetProgramInfoLog)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
    void (GLAPI * GetShaderiv)(GLuint shader, GLenum pname, GLint *params);
    void (GLAPI * GetShaderInfoLog)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
    GLint (GLAPI * GetUniformLocation)(GLuint program, const GLchar *name);
    void (GLAPI * LinkProgram)(GLuint program);
    void (GLAPI * ShaderSource)(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
    void (GLAPI * UseProgram)(GLuint program);
    void (GLAPI * Uniform1i)(GLint location, GLint v0);
    void (GLAPI * Uniform1fv)(GLint location, GLsizei count, const GLfloat *value);
    void (GLAPI * Uniform2fv)(GLint location, GLsizei count, const GLfloat *value);
    void (GLAPI * Uniform3fv)(GLint location, GLsizei count, const GLfloat *value);
    void (GLAPI * Uniform4fv)(GLint location, GLsizei count, const GLfloat *value);
    void (GLAPI * Uniform1iv)(GLint location, GLsizei count, const GLint *value);
    void (GLAPI * Uniform2iv)(GLint location, GLsizei count, const GLint *value);
    void (GLAPI * Uniform3iv)(GLint location, GLsizei count, const GLint *value);
    void (GLAPI * Uniform4iv)(GLint location, GLsizei count, const GLint *value);
    void (GLAPI * UniformMatrix2fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void (GLAPI * UniformMatrix3fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void (GLAPI * UniformMatrix4fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void (GLAPI * VertexAttribPointer)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
    void (GLAPI * UniformMatrix2x3fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void (GLAPI * UniformMatrix3x2fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void (GLAPI * UniformMatrix2x4fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void (GLAPI * UniformMatrix4x2fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void (GLAPI * UniformMatrix3x4fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void (GLAPI * UniformMatrix4x3fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void (GLAPI * Enablei)(GLenum target, GLuint index);
    void (GLAPI * Disablei)(GLenum target, GLuint index);
    void (GLAPI * BindBufferRange)(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);
    void (GLAPI * VertexAttribIPointer)(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer);
    void (GLAPI * Uniform1uiv)(GLint location, GLsizei count, const GLuint *value);
    void (GLAPI * Uniform2uiv)(GLint location, GLsizei count, const GLuint *value);
    void (GLAPI * Uniform3uiv)(GLint location, GLsizei count, const GLuint *value);
    void (GLAPI * Uniform4uiv)(GLint location, GLsizei count, const GLuint *value);
    void (GLAPI * ClearBufferiv)(GLenum buffer, GLint drawbuffer, const GLint *value);
    void (GLAPI * ClearBufferuiv)(GLenum buffer, GLint drawbuffer, const GLuint *value);
    void (GLAPI * ClearBufferfv)(GLenum buffer, GLint drawbuffer, const GLfloat *value);
    void (GLAPI * ClearBufferfi)(GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil);
    void (GLAPI * BindRenderbuffer)(GLenum target, GLuint renderbuffer);
    void (GLAPI * DeleteRenderbuffers)(GLsizei n, const GLuint *renderbuffers);
    void (GLAPI * GenRenderbuffers)(GLsizei n, GLuint *renderbuffers);
    void (GLAPI * BindFramebuffer)(GLenum target, GLuint framebuffer);
    void (GLAPI * DeleteFramebuffers)(GLsizei n, const GLuint *framebuffers);
    void (GLAPI * GenFramebuffers)(GLsizei n, GLuint *framebuffers);
    void (GLAPI * FramebufferTexture2D)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
    void (GLAPI * FramebufferRenderbuffer)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
    void (GLAPI * GenerateMipmap)(GLenum target);
    void (GLAPI * BlitFramebuffer)(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
    void (GLAPI * RenderbufferStorageMultisample)(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
    void (GLAPI * FramebufferTextureLayer)(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer);
    void *(GLAPI * MapBufferRange)(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
    void (GLAPI * BindVertexArray)(GLuint array);
    void (GLAPI * DeleteVertexArrays)(GLsizei n, const GLuint *arrays);
    void (GLAPI * GenVertexArrays)(GLsizei n, GLuint *arrays);
    void (GLAPI * DrawArraysInstanced)(GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
    void (GLAPI * DrawElementsInstanced)(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount);
    void (GLAPI * PrimitiveRestartIndex)(GLuint index);
    GLuint (GLAPI * GetUniformBlockIndex)(GLuint program, const GLchar *uniformBlockName);
    void (GLAPI * GetActiveUniformBlockiv)(GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params);
    void (GLAPI * GetActiveUniformBlockName)(GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformBlockName);
    void (GLAPI * UniformBlockBinding)(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
    GLsync (GLAPI * FenceSync)(GLenum condition, GLbitfield flags);
    void (GLAPI * DeleteSync)(GLsync sync);
    GLenum (GLAPI * ClientWaitSync)(GLsync sync, GLbitfield flags, GLuint64 timeout);
    void (GLAPI * GenSamplers)(GLsizei count, GLuint *samplers);
    void (GLAPI * DeleteSamplers)(GLsizei count, const GLuint *samplers);
    void (GLAPI * BindSampler)(GLuint unit, GLuint sampler);
    void (GLAPI * SamplerParameteri)(GLuint sampler, GLenum pname, GLint param);
    void (GLAPI * SamplerParameterf)(GLuint sampler, GLenum pname, GLfloat param);
    void (GLAPI * SamplerParameterfv)(GLuint sampler, GLenum pname, const GLfloat *param);
    void (GLAPI * VertexAttribDivisor)(GLuint index, GLuint divisor);
    void (GLAPI * BlendEquationSeparatei)(GLuint buf, GLenum modeRGB, GLenum modeAlpha);
    void (GLAPI * BlendFuncSeparatei)(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha);
    void (GLAPI * TexStorage2D)(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
    void (GLAPI * TexStorage3D)(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
};

struct VertexFormat {
    int type;
    int size;
    int normalize;
    int integer;
};

struct ImageFormat {
    int internal_format;
    int format;
    int type;
    int components;
    int pixel_size;
    int buffer;
    int color;
    int clear_type;
    int flags;
};

struct UniformBufferBinding {
    int buffer;
    int offset;
    int size;
};

struct SamplerBinding {
    int sampler;
    int target;
    int image;
};

struct UniformBinding {
    int values;
    int location;
    int count;
    int type;
    union {
        int int_values[1];
        unsigned uint_values[1];
        float float_values[1];
    };
};

struct StencilSettings {
    int fail_op;
    int pass_op;
    int depth_fail_op;
    int compare_op;
    int compare_mask;
    int write_mask;
    int reference;
};

struct Viewport {
    int x;
    int y;
    int width;
    int height;
};

union ClearValue {
    float clear_floats[4];
    int clear_ints[4];
    unsigned int clear_uints[4];
};

struct IntPair {
    int x;
    int y;
};

static int min(int a, int b) {
    return a < b ? a : b;
}

static int max(int a, int b) {
    return a > b ? a : b;
}

// static bool is_uniform_variable(int t) {
//     return (0x1404 <= t && t <= 0x8B5C) || (0x8B65 <= t && t <= 0x8B6A) || (0x8DC6 <= t && t <= 0x8DC8) || (0x8F46 <= t && t <= 0x8FFE);
// }

static bool is_uniform_sampler(int t) {
    return (0x8B5D <= t && t <= 0x8B64) || (0x8DC0 <= t && t <= 0x8DC5) || (0x8DC9 <= t && t <= 0x8DD8) || (0x900C <= t && t <= 0x900F) || (0x9108 <= t && t <= 0x910D);
}

static bool is_uniform_image(int t) {
    return 0x904C <= t && t <= 0x906C;
}

static VertexFormat get_vertex_format(const char * format) {
    if (!strcmp(format, "uint8x2")) return {GL_UNSIGNED_BYTE, 2, false, true};
    if (!strcmp(format, "uint8x4")) return {GL_UNSIGNED_BYTE, 4, false, true};
    if (!strcmp(format, "sint8x2")) return {GL_BYTE, 2, false, true};
    if (!strcmp(format, "sint8x4")) return {GL_BYTE, 4, false, true};
    if (!strcmp(format, "unorm8x2")) return {GL_UNSIGNED_BYTE, 2, true, false};
    if (!strcmp(format, "unorm8x4")) return {GL_UNSIGNED_BYTE, 4, true, false};
    if (!strcmp(format, "snorm8x2")) return {GL_BYTE, 2, true, false};
    if (!strcmp(format, "snorm8x4")) return {GL_BYTE, 4, true, false};
    if (!strcmp(format, "uint16x2")) return {GL_UNSIGNED_SHORT, 2, false, true};
    if (!strcmp(format, "uint16x4")) return {GL_UNSIGNED_SHORT, 4, false, true};
    if (!strcmp(format, "sint16x2")) return {GL_SHORT, 2, false, true};
    if (!strcmp(format, "sint16x4")) return {GL_SHORT, 4, false, true};
    if (!strcmp(format, "unorm16x2")) return {GL_UNSIGNED_SHORT, 2, true, false};
    if (!strcmp(format, "unorm16x4")) return {GL_UNSIGNED_SHORT, 4, true, false};
    if (!strcmp(format, "snorm16x2")) return {GL_SHORT, 2, true, false};
    if (!strcmp(format, "snorm16x4")) return {GL_SHORT, 4, true, false};
    if (!strcmp(format, "float16x2")) return {GL_HALF_FLOAT, 2, false, false};
    if (!strcmp(format, "float16x4")) return {GL_HALF_FLOAT, 4, false, false};
    if (!strcmp(format, "float32")) return {GL_FLOAT, 1, false, false};
    if (!strcmp(format, "float32x2")) return {GL_FLOAT, 2, false, false};
    if (!strcmp(format, "float32x3")) return {GL_FLOAT, 3, false, false};
    if (!strcmp(format, "float32x4")) return {GL_FLOAT, 4, false, false};
    if (!strcmp(format, "uint32")) return {GL_UNSIGNED_INT, 1, false, true};
    if (!strcmp(format, "uint32x2")) return {GL_UNSIGNED_INT, 2, false, true};
    if (!strcmp(format, "uint32x3")) return {GL_UNSIGNED_INT, 3, false, true};
    if (!strcmp(format, "uint32x4")) return {GL_UNSIGNED_INT, 4, false, true};
    if (!strcmp(format, "sint32")) return {GL_INT, 1, false, true};
    if (!strcmp(format, "sint32x2")) return {GL_INT, 2, false, true};
    if (!strcmp(format, "sint32x3")) return {GL_INT, 3, false, true};
    if (!strcmp(format, "sint32x4")) return {GL_INT, 4, false, true};
    return {};
}

static ImageFormat get_image_format(const char * format) {
    if (!strcmp(format, "r8unorm")) return {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1, 1, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "rg8unorm")) return {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2, 2, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "rgba8unorm")) return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "r8snorm")) return {GL_R8_SNORM, GL_RED, GL_UNSIGNED_BYTE, 1, 1, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "rg8snorm")) return {GL_RG8_SNORM, GL_RG, GL_UNSIGNED_BYTE, 2, 2, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "rgba8snorm")) return {GL_RGBA8_SNORM, GL_RGBA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "r8uint")) return {GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, 1, 1, GL_COLOR, true, 'u', 1};
    if (!strcmp(format, "rg8uint")) return {GL_RG8UI, GL_RG_INTEGER, GL_UNSIGNED_BYTE, 2, 2, GL_COLOR, true, 'u', 1};
    if (!strcmp(format, "rgba8uint")) return {GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, true, 'u', 1};
    if (!strcmp(format, "r16uint")) return {GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT, 1, 2, GL_COLOR, true, 'u', 1};
    if (!strcmp(format, "rg16uint")) return {GL_RG16UI, GL_RG_INTEGER, GL_UNSIGNED_SHORT, 2, 4, GL_COLOR, true, 'u', 1};
    if (!strcmp(format, "rgba16uint")) return {GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, 4, 8, GL_COLOR, true, 'u', 1};
    if (!strcmp(format, "r32uint")) return {GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 1, 4, GL_COLOR, true, 'u', 1};
    if (!strcmp(format, "rg32uint")) return {GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT, 2, 8, GL_COLOR, true, 'u', 1};
    if (!strcmp(format, "rgba32uint")) return {GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 4, 16, GL_COLOR, true, 'u', 1};
    if (!strcmp(format, "r8sint")) return {GL_R8I, GL_RED_INTEGER, GL_BYTE, 1, 1, GL_COLOR, true, 'i', 1};
    if (!strcmp(format, "rg8sint")) return {GL_RG8I, GL_RG_INTEGER, GL_BYTE, 2, 2, GL_COLOR, true, 'i', 1};
    if (!strcmp(format, "rgba8sint")) return {GL_RGBA8I, GL_RGBA_INTEGER, GL_BYTE, 4, 4, GL_COLOR, true, 'i', 1};
    if (!strcmp(format, "r16sint")) return {GL_R16I, GL_RED_INTEGER, GL_SHORT, 1, 2, GL_COLOR, true, 'i', 1};
    if (!strcmp(format, "rg16sint")) return {GL_RG16I, GL_RG_INTEGER, GL_SHORT, 2, 4, GL_COLOR, true, 'i', 1};
    if (!strcmp(format, "rgba16sint")) return {GL_RGBA16I, GL_RGBA_INTEGER, GL_SHORT, 4, 8, GL_COLOR, true, 'i', 1};
    if (!strcmp(format, "r32sint")) return {GL_R32I, GL_RED_INTEGER, GL_INT, 1, 4, GL_COLOR, true, 'i', 1};
    if (!strcmp(format, "rg32sint")) return {GL_RG32I, GL_RG_INTEGER, GL_INT, 2, 8, GL_COLOR, true, 'i', 1};
    if (!strcmp(format, "rgba32sint")) return {GL_RGBA32I, GL_RGBA_INTEGER, GL_INT, 4, 16, GL_COLOR, true, 'i', 1};
    if (!strcmp(format, "r16float")) return {GL_R16F, GL_RED, GL_FLOAT, 1, 2, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "rg16float")) return {GL_RG16F, GL_RG, GL_FLOAT, 2, 4, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "rgba16float")) return {GL_RGBA16F, GL_RGBA, GL_FLOAT, 4, 8, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "r32float")) return {GL_R32F, GL_RED, GL_FLOAT, 1, 4, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "rg32float")) return {GL_RG32F, GL_RG, GL_FLOAT, 2, 8, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "rgba32float")) return {GL_RGBA32F, GL_RGBA, GL_FLOAT, 4, 16, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "rgba8unorm-srgb")) return {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, true, 'f', 1};
    if (!strcmp(format, "stencil8")) return {GL_STENCIL_INDEX8, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, 1, 1, GL_STENCIL, false, 'i', 4};
    if (!strcmp(format, "depth16unorm")) return {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, 1, 2, GL_DEPTH, false, 'f', 2};
    if (!strcmp(format, "depth24plus")) return {GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 1, 4, GL_DEPTH, false, 'f', 2};
    if (!strcmp(format, "depth24plus-stencil8")) return {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 2, 4, GL_DEPTH_STENCIL, false, 'x', 6};
    if (!strcmp(format, "depth32float")) return {GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, 4, GL_DEPTH, false, 'f', 2};
    return {};
}

static int get_topology(const char * topology) {
    if (!strcmp(topology, "points")) return GL_POINTS;
    if (!strcmp(topology, "lines")) return GL_LINES;
    if (!strcmp(topology, "line_loop")) return GL_LINE_LOOP;
    if (!strcmp(topology, "line_strip")) return GL_LINE_STRIP;
    if (!strcmp(topology, "triangles")) return GL_TRIANGLES;
    if (!strcmp(topology, "triangle_strip")) return GL_TRIANGLE_STRIP;
    if (!strcmp(topology, "triangle_fan")) return GL_TRIANGLE_FAN;
    return -1;
}

static int topology_converter(PyObject * arg, int * value) {
    if (!PyUnicode_CheckExact(arg)) {
        PyErr_Format(PyExc_TypeError, "topology must be a string");
        return 0;
    }
    int topology = get_topology(PyUnicode_AsUTF8(arg));
    if (topology == -1) {
        PyErr_Format(PyExc_ValueError, "invalid topology");
        return 0;
    }
    *value = topology;
    return 1;
}

static int count_mipmaps(int width, int height) {
    int size = width > height ? width : height;
    for (int i = 1; i < 32; ++i) {
        if (size < (1 << i)) {
            return i;
        }
    }
    return 32;
}

static void remove_dict_value(PyObject * dict, PyObject * obj) {
    PyObject * key = NULL;
    PyObject * value = NULL;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict, &pos, &key, &value)) {
        if (value == obj) {
            PyDict_DelItem(dict, key);
            break;
        }
    }
}

static void * new_ref(void * obj) {
    Py_INCREF(obj);
    return obj;
}

static bool is_int_pair(PyObject * obj) {
    return (
        PySequence_Check(obj) && PySequence_Size(obj) == 2 &&
        PyLong_CheckExact(PySequence_GetItem(obj, 0)) &&
        PyLong_CheckExact(PySequence_GetItem(obj, 1))
    );
}

static bool is_viewport(PyObject * obj) {
    return (
        PySequence_Check(obj) && PySequence_Size(obj) == 4 &&
        PyLong_CheckExact(PySequence_GetItem(obj, 0)) &&
        PyLong_CheckExact(PySequence_GetItem(obj, 1)) &&
        PyLong_CheckExact(PySequence_GetItem(obj, 2)) &&
        PyLong_CheckExact(PySequence_GetItem(obj, 3))
    );
}

static IntPair to_int_pair(PyObject * obj) {
    IntPair res = {};
    res.x = PyLong_AsLong(PySequence_GetItem(obj, 0));
    res.y = PyLong_AsLong(PySequence_GetItem(obj, 1));
    return res;
}

static Viewport to_viewport(PyObject * obj) {
    Viewport res = {};
    res.x = PyLong_AsLong(PySequence_GetItem(obj, 0));
    res.y = PyLong_AsLong(PySequence_GetItem(obj, 1));
    res.width = PyLong_AsLong(PySequence_GetItem(obj, 2));
    res.height = PyLong_AsLong(PySequence_GetItem(obj, 3));
    return res;
}

static void * load_opengl_function(PyObject * loader, const char * method) {
    if (PyObject_HasAttrString(loader, "load_opengl_function")) {
        PyObject * res = PyObject_CallMethod(loader, "load_opengl_function", "s", method);
        if (!res) {
            return NULL;
        }
        return PyLong_AsVoidPtr(res);
    }

    // deprecated path for backward compatibility
    PyObject * res = PyObject_CallMethod(loader, "load", "s", method);
    if (!res) {
        return NULL;
    }
    return PyLong_AsVoidPtr(res);
}

static GLMethods load_gl(PyObject * loader) {
    GLMethods res = {};
    PyObject * missing = PyList_New(0);

    #define check(name) if (!res.name) { if (PyErr_Occurred()) return {}; PyList_Append(missing, PyUnicode_FromString("gl" # name)); }
    #define load(name) *(void **)&res.name = load_opengl_function(loader, "gl" # name); check(name)

    load(CullFace);
    load(FrontFace);
    load(LineWidth);
    load(TexParameteri);
    load(TexImage2D);
    load(DepthMask);
    load(Disable);
    load(Enable);
    load(Flush);
    load(DepthFunc);
    load(ReadBuffer);
    load(ReadPixels);
    load(GetError);
    load(GetIntegerv);
    load(GetString);
    load(Viewport);
    load(PolygonOffset);
    load(TexSubImage2D);
    load(BindTexture);
    load(DeleteTextures);
    load(GenTextures);
    load(TexImage3D);
    load(TexSubImage3D);
    load(ActiveTexture);
    load(BlendFuncSeparate);
    load(GenQueries);
    load(BeginQuery);
    load(EndQuery);
    load(GetQueryObjectuiv);
    load(BindBuffer);
    load(DeleteBuffers);
    load(GenBuffers);
    load(BufferData);
    load(BufferSubData);
    load(UnmapBuffer);
    load(BlendEquationSeparate);
    load(DrawBuffers);
    load(StencilOpSeparate);
    load(StencilFuncSeparate);
    load(StencilMaskSeparate);
    load(AttachShader);
    load(CompileShader);
    load(CreateProgram);
    load(CreateShader);
    load(DeleteProgram);
    load(DeleteShader);
    load(EnableVertexAttribArray);
    load(GetActiveAttrib);
    load(GetActiveUniform);
    load(GetAttribLocation);
    load(GetProgramiv);
    load(GetProgramInfoLog);
    load(GetShaderiv);
    load(GetShaderInfoLog);
    load(GetUniformLocation);
    load(LinkProgram);
    load(ShaderSource);
    load(UseProgram);
    load(Uniform1i);
    load(Uniform1fv);
    load(Uniform2fv);
    load(Uniform3fv);
    load(Uniform4fv);
    load(Uniform1iv);
    load(Uniform2iv);
    load(Uniform3iv);
    load(Uniform4iv);
    load(UniformMatrix2fv);
    load(UniformMatrix3fv);
    load(UniformMatrix4fv);
    load(VertexAttribPointer);
    load(UniformMatrix2x3fv);
    load(UniformMatrix3x2fv);
    load(UniformMatrix2x4fv);
    load(UniformMatrix4x2fv);
    load(UniformMatrix3x4fv);
    load(UniformMatrix4x3fv);
    load(Enablei);
    load(Disablei);
    load(BindBufferRange);
    load(VertexAttribIPointer);
    load(Uniform1uiv);
    load(Uniform2uiv);
    load(Uniform3uiv);
    load(Uniform4uiv);
    load(ClearBufferiv);
    load(ClearBufferuiv);
    load(ClearBufferfv);
    load(ClearBufferfi);
    load(BindRenderbuffer);
    load(DeleteRenderbuffers);
    load(GenRenderbuffers);
    load(BindFramebuffer);
    load(DeleteFramebuffers);
    load(GenFramebuffers);
    load(FramebufferTexture2D);
    load(FramebufferRenderbuffer);
    load(GenerateMipmap);
    load(BlitFramebuffer);
    load(RenderbufferStorageMultisample);
    load(FramebufferTextureLayer);
    load(MapBufferRange);
    load(BindVertexArray);
    load(DeleteVertexArrays);
    load(GenVertexArrays);
    load(DrawArraysInstanced);
    load(DrawElementsInstanced);
    load(PrimitiveRestartIndex);
    load(GetUniformBlockIndex);
    load(GetActiveUniformBlockiv);
    load(GetActiveUniformBlockName);
    load(UniformBlockBinding);
    load(FenceSync);
    load(DeleteSync);
    load(ClientWaitSync);
    load(GenSamplers);
    load(DeleteSamplers);
    load(BindSampler);
    load(SamplerParameteri);
    load(SamplerParameterf);
    load(SamplerParameterfv);
    load(VertexAttribDivisor);
    load(BlendEquationSeparatei);
    load(BlendFuncSeparatei);
    load(TexStorage2D);
    load(TexStorage3D);

    #undef load
    #undef check

    if (PyList_Size(missing)) {
        PyErr_Format(PyExc_RuntimeError, "cannot load opengl %R", missing);
        return {};
    }

    Py_DECREF(missing);
    return res;
}

struct Limits {
    int max_uniform_buffer_bindings;
    int max_uniform_block_size;
    int max_combined_uniform_blocks;
    int max_combined_texture_image_units;
    int max_vertex_attribs;
    int max_draw_buffers;
    int max_samples;
};

struct ModuleState {
    PyObject * helper;
    PyObject * empty_tuple;
    PyObject * str_none;
    PyObject * float_one;
    PyTypeObject * Context_type;
    PyTypeObject * Buffer_type;
    PyTypeObject * Image_type;
    PyTypeObject * Pipeline_type;
    PyTypeObject * ImageFace_type;
    PyTypeObject * DescriptorSet_type;
    PyTypeObject * GlobalSettings_type;
    PyTypeObject * GLObject_type;
};

struct GCHeader {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
};

struct GLObject {
    PyObject_HEAD
    int uses;
    int obj;
    PyObject * extra;
};

struct DescriptorSetBuffers {
    int buffer_count;
    unsigned buffers[MAX_UNIFORM_BUFFER_BINDINGS];
    GLsizeiptr buffer_offsets[MAX_UNIFORM_BUFFER_BINDINGS];
    GLsizeiptr buffer_sizes[MAX_UNIFORM_BUFFER_BINDINGS];
    PyObject * buffer_refs[MAX_UNIFORM_BUFFER_BINDINGS];
};

struct DescriptorSetSamplers {
    int sampler_count;
    unsigned samplers[MAX_SAMPLER_BINDINGS];
    unsigned textures[MAX_SAMPLER_BINDINGS];
    unsigned targets[MAX_SAMPLER_BINDINGS];
    PyObject * sampler_refs[MAX_SAMPLER_BINDINGS];
    PyObject * texture_refs[MAX_SAMPLER_BINDINGS];
};

struct DescriptorSet {
    PyObject_HEAD
    int uses;
    DescriptorSetBuffers uniform_buffers;
    DescriptorSetSamplers samplers;
};

struct BlendState {
    int enable;
    int op_color;
    int op_alpha;
    int src_color;
    int dst_color;
    int src_alpha;
    int dst_alpha;
};

struct GlobalSettings {
    PyObject_HEAD
    int uses;
    int attachments;
    int cull_face;
    int depth_enabled;
    int depth_write;
    int depth_func;
    int stencil_enabled;
    StencilSettings stencil_front;
    StencilSettings stencil_back;
    int blend_enabled;
    BlendState blend[MAX_ATTACHMENTS];
};

struct Context {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    ModuleState * module_state;
    PyObject * descriptor_set_cache;
    PyObject * global_settings_cache;
    PyObject * sampler_cache;
    PyObject * vertex_array_cache;
    PyObject * framebuffer_cache;
    PyObject * program_cache;
    PyObject * shader_cache;
    PyObject * includes;
    PyObject * before_frame_callback;
    PyObject * after_frame_callback;
    PyObject * limits_dict;
    PyObject * info_dict;
    DescriptorSet * current_descriptor_set;
    GlobalSettings * current_global_settings;
    int is_mask_default;
    int is_stencil_default;
    int is_blend_default;
    Viewport current_viewport;
    int current_attachments;
    int current_framebuffer;
    int current_program;
    int current_vertex_array;
    int current_depth_mask;
    int current_stencil_mask;
    int frame_time_query;
    int frame_time_query_running;
    int frame_time;
    int default_texture_unit;
    int mapped_buffers;
    int screen;
    Limits limits;
    GLMethods gl;
};

struct Buffer {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    Context * ctx;
    int buffer;
    int size;
    int dynamic;
    int mapped;
};

struct Image {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    Context * ctx;
    PyObject * size;
    PyObject * format;
    GLObject * framebuffer;
    PyObject * faces;
    ClearValue clear_value;
    ImageFormat fmt;
    int image;
    int width;
    int height;
    int samples;
    int array;
    int cubemap;
    int target;
    int renderbuffer;
    int max_level;
};

struct Pipeline {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    Context * ctx;
    DescriptorSet * descriptor_set;
    GlobalSettings * global_settings;
    GLObject * framebuffer;
    GLObject * vertex_array;
    GLObject * program;
    PyObject * uniform_map;
    char * uniform_data;
    int uniform_count;
    int topology;
    int vertex_count;
    int instance_count;
    int first_vertex;
    int index_type;
    int index_size;
    Viewport viewport;
};

struct ImageFace {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    Context * ctx;
    Image * image;
    GLObject * framebuffer;
    PyObject * size;
    int width;
    int height;
    int layer;
    int level;
    int samples;
    int flags;
};

static void bind_global_settings(Context * self, GlobalSettings * settings) {
    const GLMethods & gl = self->gl;
    if (self->current_global_settings == settings) {
        return;
    }
    if (settings->cull_face) {
        gl.Enable(GL_CULL_FACE);
        gl.CullFace(settings->cull_face);
    } else {
        gl.Disable(GL_CULL_FACE);
    }
    if (settings->depth_enabled) {
        gl.Enable(GL_DEPTH_TEST);
        gl.DepthFunc(settings->depth_func);
        gl.DepthMask(settings->depth_write);
        self->current_depth_mask = settings->depth_write;
    } else {
        gl.Disable(GL_DEPTH_TEST);
    }
    if (settings->stencil_enabled) {
        gl.Enable(GL_STENCIL_TEST);
        gl.StencilMaskSeparate(GL_FRONT, settings->stencil_front.write_mask);
        gl.StencilMaskSeparate(GL_BACK, settings->stencil_back.write_mask);
        gl.StencilFuncSeparate(GL_FRONT, settings->stencil_front.compare_op, settings->stencil_front.reference, settings->stencil_front.compare_mask);
        gl.StencilFuncSeparate(GL_BACK, settings->stencil_back.compare_op, settings->stencil_back.reference, settings->stencil_back.compare_mask);
        gl.StencilOpSeparate(GL_FRONT, settings->stencil_front.fail_op, settings->stencil_front.pass_op, settings->stencil_front.depth_fail_op);
        gl.StencilOpSeparate(GL_BACK, settings->stencil_back.fail_op, settings->stencil_back.pass_op, settings->stencil_back.depth_fail_op);
        self->current_stencil_mask = settings->stencil_front.write_mask;
    } else {
        gl.Disable(GL_STENCIL_TEST);
    }
    if (settings->blend_enabled) {
        for (int i = 0; i < settings->attachments; ++i) {
            if (settings->blend[i].enable) {
                gl.Enablei(GL_BLEND, i);
            } else {
                gl.Disablei(GL_BLEND, i);
            }
            gl.BlendEquationSeparatei(i, settings->blend[i].op_color, settings->blend[i].op_alpha);
            gl.BlendFuncSeparatei(i, settings->blend[i].src_color, settings->blend[i].dst_color, settings->blend[i].src_alpha, settings->blend[i].dst_alpha);
        }
    } else {
        gl.Disable(GL_BLEND);
    }
    self->current_global_settings = settings;
}

static void bind_framebuffer(Context * self, int framebuffer) {
    if (self->current_framebuffer != framebuffer) {
        self->current_framebuffer = framebuffer;
        self->gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    }
}

static void bind_program(Context * self, int program) {
    if (self->current_program != program) {
        self->current_program = program;
        self->gl.UseProgram(program);
    }
}

static void bind_vertex_array(Context * self, int vertex_array) {
    if (self->current_vertex_array != vertex_array) {
        self->current_vertex_array = vertex_array;
        self->gl.BindVertexArray(vertex_array);
    }
}

static void bind_descriptor_set(Context * self, DescriptorSet * set) {
    const GLMethods & gl = self->gl;
    if (self->current_descriptor_set != set) {
        self->current_descriptor_set = set;
        if (set->uniform_buffers.buffer_count) {
            for (int i = 0; i < set->uniform_buffers.buffer_count; ++i) {
                gl.BindBufferRange(
                    GL_UNIFORM_BUFFER,
                    i,
                    set->uniform_buffers.buffers[i],
                    set->uniform_buffers.buffer_offsets[i],
                    set->uniform_buffers.buffer_sizes[i]
                );
            }
        }
        if (set->samplers.sampler_count) {
            for (int i = 0; i < set->samplers.sampler_count; ++i) {
                gl.ActiveTexture(GL_TEXTURE0 + i);
                gl.BindTexture(set->samplers.targets[i], set->samplers.textures[i]);
                gl.BindSampler(i, set->samplers.samplers[i]);
            }
        }
    }
}

static GLObject * build_framebuffer(Context * self, PyObject * attachments) {
    if (GLObject * cache = (GLObject *)PyDict_GetItem(self->framebuffer_cache, attachments)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    PyObject * color_attachments = PyTuple_GetItem(attachments, 1);
    PyObject * depth_stencil_attachment = PyTuple_GetItem(attachments, 2);

    const GLMethods & gl = self->gl;

    int framebuffer = 0;
    gl.GenFramebuffers(1, (unsigned *)&framebuffer);
    bind_framebuffer(self, framebuffer);
    int color_attachment_count = (int)PyTuple_Size(color_attachments);
    for (int i = 0; i < color_attachment_count; ++i) {
        ImageFace * face = (ImageFace *)PyTuple_GetItem(color_attachments, i);
        if (face->image->renderbuffer) {
            gl.FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_RENDERBUFFER, face->image->image);
        } else if (face->image->cubemap) {
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face->layer, face->image->image, face->level);
        } else if (face->image->array) {
            gl.FramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, face->image->image, face->level, face->layer);
        } else {
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, face->image->image, face->level);
        }
    }

    if (depth_stencil_attachment != Py_None) {
        ImageFace * face = (ImageFace *)depth_stencil_attachment;
        int buffer = face->image->fmt.buffer;
        int attachment = buffer == GL_DEPTH ? GL_DEPTH_ATTACHMENT : buffer == GL_STENCIL ? GL_STENCIL_ATTACHMENT : GL_DEPTH_STENCIL_ATTACHMENT;
        if (face->image->renderbuffer) {
            gl.FramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, face->image->image);
        } else if (face->image->cubemap) {
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face->layer, face->image->image, face->level);
        } else if (face->image->array) {
            gl.FramebufferTextureLayer(GL_FRAMEBUFFER, attachment, face->image->image, face->level, face->layer);
        } else {
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, face->image->image, face->level);
        }
    }

    unsigned int draw_buffers[MAX_ATTACHMENTS];
    for (int i = 0; i < color_attachment_count; ++i) {
        draw_buffers[i] = GL_COLOR_ATTACHMENT0 + i;
    }

    gl.DrawBuffers(color_attachment_count, draw_buffers);
    gl.ReadBuffer(color_attachment_count ? GL_COLOR_ATTACHMENT0 : 0);

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = framebuffer;
    res->uses = 1;

    PyDict_SetItem(self->framebuffer_cache, attachments, (PyObject *)res);
    return res;
}

static void bind_uniforms(Context * self, char * data) {
    const GLMethods & gl = self->gl;
    int offset = 0;
    while (true) {
        UniformBinding * header = (UniformBinding *)(data + offset);
        if (header->type == 0) {
            break;
        }
        switch (header->type) {
            case 0x1404: gl.Uniform1iv(header->location, header->count, header->int_values); break;
            case 0x8B53: gl.Uniform2iv(header->location, header->count, header->int_values); break;
            case 0x8B54: gl.Uniform3iv(header->location, header->count, header->int_values); break;
            case 0x8B55: gl.Uniform4iv(header->location, header->count, header->int_values); break;
            case 0x8B56: gl.Uniform1iv(header->location, header->count, header->int_values); break;
            case 0x8B57: gl.Uniform2iv(header->location, header->count, header->int_values); break;
            case 0x8B58: gl.Uniform3iv(header->location, header->count, header->int_values); break;
            case 0x8B59: gl.Uniform4iv(header->location, header->count, header->int_values); break;
            case 0x1405: gl.Uniform1uiv(header->location, header->count, header->uint_values); break;
            case 0x8DC6: gl.Uniform2uiv(header->location, header->count, header->uint_values); break;
            case 0x8DC7: gl.Uniform3uiv(header->location, header->count, header->uint_values); break;
            case 0x8DC8: gl.Uniform4uiv(header->location, header->count, header->uint_values); break;
            case 0x1406: gl.Uniform1fv(header->location, header->count, header->float_values); break;
            case 0x8B50: gl.Uniform2fv(header->location, header->count, header->float_values); break;
            case 0x8B51: gl.Uniform3fv(header->location, header->count, header->float_values); break;
            case 0x8B52: gl.Uniform4fv(header->location, header->count, header->float_values); break;
            case 0x8B5A: gl.UniformMatrix2fv(header->location, header->count, false, header->float_values); break;
            case 0x8B65: gl.UniformMatrix2x3fv(header->location, header->count, false, header->float_values); break;
            case 0x8B66: gl.UniformMatrix2x4fv(header->location, header->count, false, header->float_values); break;
            case 0x8B67: gl.UniformMatrix3x2fv(header->location, header->count, false, header->float_values); break;
            case 0x8B5B: gl.UniformMatrix3fv(header->location, header->count, false, header->float_values); break;
            case 0x8B68: gl.UniformMatrix3x4fv(header->location, header->count, false, header->float_values); break;
            case 0x8B69: gl.UniformMatrix4x2fv(header->location, header->count, false, header->float_values); break;
            case 0x8B6A: gl.UniformMatrix4x3fv(header->location, header->count, false, header->float_values); break;
            case 0x8B5C: gl.UniformMatrix4fv(header->location, header->count, false, header->float_values); break;
        }
        offset += header->values * 4 + 16;
    }
}

static GLObject * build_vertex_array(Context * self, PyObject * bindings) {
    if (GLObject * cache = (GLObject *)PyDict_GetItem(self->vertex_array_cache, bindings)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    const GLMethods & gl = self->gl;

    int length = (int)PyTuple_Size(bindings);
    PyObject ** seq = PySequence_Fast_ITEMS(bindings);
    PyObject * index_buffer = seq[0];

    int vertex_array = 0;
    gl.GenVertexArrays(1, (unsigned *)&vertex_array);
    bind_vertex_array(self, vertex_array);

    for (int i = 1; i < length; i += 6) {
        Buffer * buffer = (Buffer *)seq[i + 0];
        int location = PyLong_AsLong(seq[i + 1]);
        void * offset = PyLong_AsVoidPtr(seq[i + 2]);
        int stride = PyLong_AsLong(seq[i + 3]);
        int divisor = PyLong_AsLong(seq[i + 4]);
        VertexFormat format = get_vertex_format(PyUnicode_AsUTF8(seq[i + 5]));
        gl.BindBuffer(GL_ARRAY_BUFFER, buffer->buffer);
        if (format.integer) {
            gl.VertexAttribIPointer(location, format.size, format.type, stride, offset);
        } else {
            gl.VertexAttribPointer(location, format.size, format.type, format.normalize, stride, offset);
        }
        gl.VertexAttribDivisor(location, divisor);
        gl.EnableVertexAttribArray(location);
    }

    if (index_buffer != Py_None) {
        Buffer * buffer = (Buffer *)index_buffer;
        gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer->buffer);
    }

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = vertex_array;
    res->uses = 1;

    PyDict_SetItem(self->vertex_array_cache, bindings, (PyObject *)res);
    return res;
}

static GLObject * build_sampler(Context * self, PyObject * params) {
    if (GLObject * cache = (GLObject *)PyDict_GetItem(self->sampler_cache, params)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    const GLMethods & gl = self->gl;

    PyObject ** seq = PySequence_Fast_ITEMS(params);

    int sampler = 0;
    gl.GenSamplers(1, (unsigned *)&sampler);
    gl.SamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, PyLong_AsLong(seq[0]));
    gl.SamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, PyLong_AsLong(seq[1]));
    gl.SamplerParameterf(sampler, GL_TEXTURE_MIN_LOD, (float)PyFloat_AsDouble(seq[2]));
    gl.SamplerParameterf(sampler, GL_TEXTURE_MAX_LOD, (float)PyFloat_AsDouble(seq[3]));
    gl.SamplerParameterf(sampler, GL_TEXTURE_LOD_BIAS, (float)PyFloat_AsDouble(seq[4]));
    gl.SamplerParameteri(sampler, GL_TEXTURE_WRAP_S, PyLong_AsLong(seq[5]));
    gl.SamplerParameteri(sampler, GL_TEXTURE_WRAP_T, PyLong_AsLong(seq[6]));
    gl.SamplerParameteri(sampler, GL_TEXTURE_WRAP_R, PyLong_AsLong(seq[7]));
    gl.SamplerParameteri(sampler, GL_TEXTURE_COMPARE_MODE, PyLong_AsLong(seq[8]));
    gl.SamplerParameteri(sampler, GL_TEXTURE_COMPARE_FUNC, PyLong_AsLong(seq[9]));

    float max_anisotropy = (float)PyFloat_AsDouble(seq[10]);
    if (max_anisotropy != 1.0f) {
        gl.SamplerParameterf(sampler, GL_TEXTURE_MAX_ANISOTROPY, max_anisotropy);
    }

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = sampler;
    res->uses = 1;
    res->extra = NULL;

    PyDict_SetItem(self->sampler_cache, params, (PyObject *)res);
    return res;
}

static DescriptorSetBuffers build_descriptor_set_buffers(Context * self, PyObject * bindings) {
    DescriptorSetBuffers res = {};

    int length = (int)PyTuple_Size(bindings);
    PyObject ** seq = PySequence_Fast_ITEMS(bindings);

    for (int i = 0; i < length; i += 4) {
        int binding = PyLong_AsLong(seq[i + 0]);
        Buffer * buffer = (Buffer *)seq[i + 1];
        int offset = PyLong_AsLong(seq[i + 2]);
        int size = PyLong_AsLong(seq[i + 3]);
        res.buffers[binding] = buffer->buffer;
        res.buffer_offsets[binding] = offset;
        res.buffer_sizes[binding] = size;
        res.buffer_refs[binding] = (PyObject *)new_ref(buffer);
        res.buffer_count = res.buffer_count > (binding + 1) ? res.buffer_count : (binding + 1);
    }

    return res;
}

static DescriptorSetSamplers build_descriptor_set_samplers(Context * self, PyObject * bindings) {
    DescriptorSetSamplers res = {};

    int length = (int)PyTuple_Size(bindings);
    PyObject ** seq = PySequence_Fast_ITEMS(bindings);

    for (int i = 0; i < length; i += 3) {
        int binding = PyLong_AsLong(seq[i + 0]);
        Image * image = (Image *)seq[i + 1];
        GLObject * sampler = build_sampler(self, seq[i + 2]);
        res.samplers[binding] = sampler->obj;
        res.textures[binding] = image->image;
        res.targets[binding] = image->target;
        res.sampler_refs[binding] = (PyObject *)sampler;
        res.texture_refs[binding] = (PyObject *)new_ref(image);
        res.sampler_count = res.sampler_count > (binding + 1) ? res.sampler_count : (binding + 1);
    }

    return res;
}

static DescriptorSet * build_descriptor_set(Context * self, PyObject * bindings) {
    if (DescriptorSet * cache = (DescriptorSet *)PyDict_GetItem(self->descriptor_set_cache, bindings)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    DescriptorSet * res = PyObject_New(DescriptorSet, self->module_state->DescriptorSet_type);
    res->uniform_buffers = build_descriptor_set_buffers(self, PyTuple_GetItem(bindings, 0));
    res->samplers = build_descriptor_set_samplers(self, PyTuple_GetItem(bindings, 1));
    res->uses = 1;
    return res;
}

static GlobalSettings * build_global_settings(Context * self, PyObject * settings) {
    if (GlobalSettings * cache = (GlobalSettings *)PyDict_GetItem(self->global_settings_cache, settings)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    PyObject ** seq = PySequence_Fast_ITEMS(settings);

    GlobalSettings * res = PyObject_New(GlobalSettings, self->module_state->GlobalSettings_type);
    res->uses = 1;

    int it = 0;
    res->attachments = PyLong_AsLong(seq[it++]);
    res->cull_face = PyLong_AsLong(seq[it++]);
    res->depth_enabled = PyObject_IsTrue(seq[it++]);
    if (res->depth_enabled) {
        res->depth_func = PyLong_AsLong(seq[it++]);
        res->depth_write = PyObject_IsTrue(seq[it++]);
    }
    res->stencil_enabled = PyObject_IsTrue(seq[it++]);
    if (res->stencil_enabled) {
        res->stencil_front = {
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
        };
        res->stencil_back = {
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
            PyLong_AsLong(seq[it++]),
        };
    }
    res->blend_enabled = PyLong_AsLong(seq[it++]);
    if (res->blend_enabled) {
        for (int i = 0; i < res->attachments; ++i) {
            res->blend[i].enable = PyLong_AsLong(seq[it++]);
            res->blend[i].op_color = PyLong_AsLong(seq[it++]);
            res->blend[i].op_alpha = PyLong_AsLong(seq[it++]);
            res->blend[i].src_color = PyLong_AsLong(seq[it++]);
            res->blend[i].dst_color = PyLong_AsLong(seq[it++]);
            res->blend[i].src_alpha = PyLong_AsLong(seq[it++]);
            res->blend[i].dst_alpha = PyLong_AsLong(seq[it++]);
        }
    }

    PyDict_SetItem(self->global_settings_cache, settings, (PyObject *)res);
    return res;
}

static GLObject * compile_shader(Context * self, PyObject * pair) {
    if (GLObject * cache = (GLObject *)PyDict_GetItem(self->shader_cache, pair)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    const GLMethods & gl = self->gl;

    PyObject * code = PyTuple_GetItem(pair, 0);
    const char * src = PyBytes_AsString(code);
    int type = PyLong_AsLong(PyTuple_GetItem(pair, 1));
    int shader = gl.CreateShader(type);
    gl.ShaderSource(shader, 1, &src, NULL);
    gl.CompileShader(shader);

    int shader_compiled = false;
    gl.GetShaderiv(shader, GL_COMPILE_STATUS, &shader_compiled);

    if (!shader_compiled) {
        int log_size = 0;
        gl.GetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_size);
        PyObject * log_text = PyBytes_FromStringAndSize(NULL, log_size);
        gl.GetShaderInfoLog(shader, log_size, &log_size, PyBytes_AsString(log_text));
        Py_XDECREF(PyObject_CallMethod(self->module_state->helper, "compile_error", "(OiN)", code, type, log_text));
        return NULL;
    }

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = shader;
    res->uses = 1;
    res->extra = NULL;

    PyDict_SetItem(self->shader_cache, pair, (PyObject *)res);
    return res;
}

static PyObject * program_interface(Context * self, int program) {
    Py_RETURN_NONE;
}

static GLObject * compile_program(Context * self, PyObject * includes, PyObject * vert, PyObject * frag, PyObject * layout) {
    const GLMethods & gl = self->gl;

    PyObject * tup = PyObject_CallMethod(self->module_state->helper, "program", "(OOOO)", vert, frag, layout, includes);
    if (!tup) {
        return NULL;
    }

    if (GLObject * cache = (GLObject *)PyDict_GetItem(self->program_cache, tup)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    PyObject * vert_pair = PyTuple_GetItem(tup, 0);
    PyObject * frag_pair = PyTuple_GetItem(tup, 1);

    GLObject * vertex_shader = compile_shader(self, vert_pair);
    if (!vertex_shader) {
        Py_DECREF(tup);
        return NULL;
    }
    int vertex_shader_obj = vertex_shader->obj;
    Py_DECREF(vertex_shader);

    GLObject * fragment_shader = compile_shader(self, frag_pair);
    if (!fragment_shader) {
        Py_DECREF(tup);
        return NULL;
    }
    int fragment_shader_obj = fragment_shader->obj;
    Py_DECREF(fragment_shader);

    int program = gl.CreateProgram();
    gl.AttachShader(program, vertex_shader_obj);
    gl.AttachShader(program, fragment_shader_obj);
    gl.LinkProgram(program);

    int linked = false;
    gl.GetProgramiv(program, GL_LINK_STATUS, &linked);

    if (!linked) {
        int log_size = 0;
        gl.GetProgramiv(program, GL_INFO_LOG_LENGTH, &log_size);
        PyObject * log_text = PyBytes_FromStringAndSize(NULL, log_size);
        gl.GetProgramInfoLog(program, log_size, &log_size, PyBytes_AsString(log_text));
        PyObject * vert_code = PyTuple_GetItem(vert_pair, 0);
        PyObject * frag_code = PyTuple_GetItem(frag_pair, 1);
        Py_XDECREF(PyObject_CallMethod(self->module_state->helper, "linker_error", "(OON)", vert_code, frag_code, log_text));
        return NULL;
    }

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = program;
    res->uses = 1;
    res->extra = program_interface(self, program);

    PyDict_SetItem(self->program_cache, tup, (PyObject *)res);
    Py_DECREF(tup);
    return res;
}

static Context * meth_context(PyObject * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"loader", NULL};

    PyObject * loader = Py_None;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|O", keywords, &loader)) {
        return NULL;
    }

    ModuleState * module_state = (ModuleState *)PyModule_GetState(self);

    if (loader == Py_None) {
        loader = PyObject_CallMethod(module_state->helper, "loader", NULL);
        if (!loader) {
            return NULL;
        }
    } else {
        Py_INCREF(loader);
    }

    GLMethods gl = load_gl(loader);
    Py_DECREF(loader);

    if (PyErr_Occurred()) {
        return NULL;
    }

    Limits limits = {};

    gl.GetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &limits.max_uniform_buffer_bindings);
    gl.GetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &limits.max_uniform_block_size);
    gl.GetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS, &limits.max_combined_uniform_blocks);
    gl.GetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &limits.max_combined_texture_image_units);
    gl.GetIntegerv(GL_MAX_VERTEX_ATTRIBS, &limits.max_vertex_attribs);
    gl.GetIntegerv(GL_MAX_DRAW_BUFFERS, &limits.max_draw_buffers);
    gl.GetIntegerv(GL_MAX_SAMPLES, &limits.max_samples);

    limits.max_uniform_buffer_bindings = min(limits.max_uniform_buffer_bindings, MAX_UNIFORM_BUFFER_BINDINGS);
    limits.max_combined_texture_image_units = min(limits.max_combined_texture_image_units, MAX_SAMPLER_BINDINGS);

    PyObject * limits_dict = Py_BuildValue(
        "{sisisisisisisi}",
        "max_uniform_buffer_bindings", limits.max_uniform_buffer_bindings,
        "max_uniform_block_size", limits.max_uniform_block_size,
        "max_combined_uniform_blocks", limits.max_combined_uniform_blocks,
        "max_combined_texture_image_units", limits.max_combined_texture_image_units,
        "max_vertex_attribs", limits.max_vertex_attribs,
        "max_draw_buffers", limits.max_draw_buffers,
        "max_samples", limits.max_samples
    );

    PyObject * info_dict = Py_BuildValue(
        "{szszszsz}",
        "vendor", gl.GetString(GL_VENDOR),
        "renderer", gl.GetString(GL_RENDERER),
        "version", gl.GetString(GL_VERSION),
        "glsl", gl.GetString(GL_SHADING_LANGUAGE_VERSION)
    );

    int max_texture_image_units = 0;
    gl.GetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &max_texture_image_units);
    int default_texture_unit = GL_TEXTURE0 + max_texture_image_units - 1;

    gl.Enable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    gl.Enable(GL_PROGRAM_POINT_SIZE);
    gl.Enable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    gl.Enable(GL_FRAMEBUFFER_SRGB);

    Context * res = PyObject_New(Context, module_state->Context_type);
    res->gc_prev = (GCHeader *)res;
    res->gc_next = (GCHeader *)res;
    res->module_state = module_state;
    res->descriptor_set_cache = PyDict_New();
    res->global_settings_cache = PyDict_New();
    res->sampler_cache = PyDict_New();
    res->vertex_array_cache = PyDict_New();
    res->framebuffer_cache = PyDict_New();
    res->program_cache = PyDict_New();
    res->shader_cache = PyDict_New();
    res->includes = PyDict_New();
    res->before_frame_callback = (PyObject *)new_ref(Py_None);
    res->after_frame_callback = (PyObject *)new_ref(Py_None);
    res->limits_dict = limits_dict;
    res->info_dict = info_dict;
    res->current_descriptor_set = NULL;
    res->current_global_settings = NULL;
    res->is_mask_default = false;
    res->is_stencil_default = false;
    res->is_blend_default = false;
    res->current_viewport = {};
    res->current_framebuffer = -1;
    res->current_program = -1;
    res->current_vertex_array = -1;
    res->current_depth_mask = 0;
    res->current_stencil_mask = 0;
    res->frame_time_query = 0;
    res->frame_time_query_running = false;
    res->frame_time = 0;
    res->default_texture_unit = default_texture_unit;
    res->mapped_buffers = 0;
    res->screen = 0;
    res->limits = limits;
    res->gl = gl;
    return res;
}

static Buffer * Context_meth_buffer(Context * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "size", "dynamic", "external", NULL};

    PyObject * data = Py_None;
    PyObject * size_arg = Py_None;
    int dynamic = true;
    int external = 0;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|O$Opi", keywords, &data, &size_arg, &dynamic, &external)) {
        return NULL;
    }

    const GLMethods & gl = self->gl;

    Py_buffer view = {};

    if (data != Py_None) {
        if (PyObject_GetBuffer(data, &view, PyBUF_SIMPLE)) {
            return NULL;
        }
    }

    const bool invalid_size_type = size_arg != Py_None && !PyLong_CheckExact(size_arg);

    int size = (int)view.len;
    if (size_arg != Py_None && !invalid_size_type) {
        size = PyLong_AsLong(size_arg);
    }

    const bool data_but_size = data != Py_None && size_arg != Py_None;
    const bool invalid_size = size <= 0;

    if (invalid_size_type || invalid_size || data_but_size) {
        if (invalid_size_type) {
            PyErr_Format(PyExc_TypeError, "the size must be an int");
        } else if (invalid_size) {
            PyErr_Format(PyExc_ValueError, "invalid size");
        } else if (data_but_size) {
            PyErr_Format(PyExc_ValueError, "data and size are exclusive");
        }
        if (data != Py_None) {
            PyBuffer_Release(&view);
        }
        return NULL;
    }

    int buffer = 0;
    if (external) {
        buffer = external;
    } else {
        gl.GenBuffers(1, (unsigned *)&buffer);
        gl.BindBuffer(GL_ARRAY_BUFFER, buffer);
        gl.BufferData(GL_ARRAY_BUFFER, size, view.buf, dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
    }

    Buffer * res = PyObject_New(Buffer, self->module_state->Buffer_type);
    res->gc_prev = self->gc_prev;
    res->gc_next = (GCHeader *)self;
    res->gc_prev->gc_next = (GCHeader *)res;
    res->gc_next->gc_prev = (GCHeader *)res;
    res->ctx = (Context *)new_ref(self);
    res->buffer = buffer;
    res->size = size;
    res->dynamic = dynamic;
    res->mapped = false;

    if (data != Py_None) {
        PyBuffer_Release(&view);
    }

    Py_INCREF(res);
    return res;
}

static Image * Context_meth_image(Context * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"size", "format", "data", "samples", "array", "levels", "texture", "cubemap", "external", NULL};

    int width;
    int height;
    PyObject * format;
    PyObject * data = Py_None;
    int samples = 1;
    int array = 0;
    PyObject * texture = Py_None;
    int cubemap = false;
    int levels = -1;
    int external = 0;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "(ii)O!|OiiiOpi",
        keywords,
        &width,
        &height,
        &PyUnicode_Type,
        &format,
        &data,
        &samples,
        &array,
        &levels,
        &texture,
        &cubemap,
        &external
    );

    if (!args_ok) {
        return NULL;
    }

    const GLMethods & gl = self->gl;

    if (levels < 0) {
        levels = count_mipmaps(width, height);
    }

    const bool invalid_texture_parameter = texture != Py_True && texture != Py_False && texture != Py_None;
    const bool samples_but_texture = samples > 1 && texture == Py_True;
    const bool invalid_samples = samples < 1 || (samples & (samples - 1)) || samples > 16;
    const bool invalid_array = array < 0;
    const bool cubemap_array = cubemap && array;
    const bool cubemap_or_array_renderbuffer = (array || cubemap) && (samples > 1 || texture == Py_False);
    const bool data_and_renderbuffer = data != Py_None && (samples > 1 || texture == Py_False);

    if (invalid_texture_parameter || samples_but_texture || invalid_samples || invalid_array || cubemap_array || cubemap_or_array_renderbuffer || data_and_renderbuffer) {
        if (invalid_texture_parameter) {
            PyErr_Format(PyExc_TypeError, "invalid texture parameter");
        } else if (samples_but_texture) {
            PyErr_Format(PyExc_TypeError, "for multisampled images texture must be False");
        } else if (invalid_samples) {
            PyErr_Format(PyExc_ValueError, "samples must be 1, 2, 4, 8 or 16");
        } else if (invalid_array) {
            PyErr_Format(PyExc_ValueError, "array must not be negative");
        } else if (cubemap_array) {
            PyErr_Format(PyExc_TypeError, "cubemap arrays are not supported");
        } else if (array && samples > 1) {
            PyErr_Format(PyExc_TypeError, "multisampled array images are not supported");
        } else if (cubemap && samples > 1) {
            PyErr_Format(PyExc_TypeError, "multisampled cubemap images are not supported");
        } else if (array && texture == Py_False) {
            PyErr_Format(PyExc_TypeError, "for array images texture must be True");
        } else if (cubemap && texture == Py_False) {
            PyErr_Format(PyExc_TypeError, "for cubemap images texture must be True");
        } else if (data_and_renderbuffer && samples > 1) {
            PyErr_Format(PyExc_ValueError, "cannot write to multisampled images");
        } else if (data_and_renderbuffer && texture == Py_False) {
            PyErr_Format(PyExc_ValueError, "cannot write to renderbuffers");
        }
        return NULL;
    }

    int renderbuffer = samples > 1 || texture == Py_False;
    int target = cubemap ? GL_TEXTURE_CUBE_MAP : array ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;

    if (samples > self->limits.max_samples) {
        samples = self->limits.max_samples;
    }

    ImageFormat fmt = get_image_format(PyUnicode_AsUTF8(format));

    if (!fmt.type) {
        PyErr_Format(PyExc_ValueError, "invalid image format");
        return NULL;
    }

    Py_buffer view = {};

    if (data != Py_None) {
        if (PyObject_GetBuffer(data, &view, PyBUF_SIMPLE)) {
            return NULL;
        }
        int padded_row = (width * fmt.pixel_size + 3) & ~3;
        int expected_size = padded_row * height * (array ? array : 1) * (cubemap ? 6 : 1);
        if ((int)view.len != expected_size) {
            PyBuffer_Release(&view);
            PyErr_Format(PyExc_ValueError, "invalid data size, expected %d, got %d", expected_size, (int)view.len);
            return NULL;
        }
    }

    int image = 0;
    if (external) {
        image = external;
    } else if (renderbuffer) {
        gl.GenRenderbuffers(1, (unsigned *)&image);
        gl.BindRenderbuffer(GL_RENDERBUFFER, image);
        gl.RenderbufferStorageMultisample(GL_RENDERBUFFER, samples > 1 ? samples : 0, fmt.internal_format, width, height);
    } else {
        gl.GenTextures(1, (unsigned *)&image);
        gl.ActiveTexture(self->default_texture_unit);
        gl.BindTexture(target, image);
        gl.TexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl.TexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        if (cubemap) {
            gl.TexStorage2D(target, levels, fmt.internal_format, width, height);
            if (view.buf) {
                gl.TexSubImage3D(target, 0, 0, 0, 0, width, height, 6, fmt.format, fmt.type, view.buf);
            }
        } else if (array) {
            gl.TexStorage3D(target, levels, fmt.internal_format, width, height, array);
            if (view.buf) {
                gl.TexSubImage3D(target, 0, 0, 0, 0, width, height, array, fmt.format, fmt.type, view.buf);
            }
        } else {
            gl.TexStorage2D(target, levels, fmt.internal_format, width, height);
            if (view.buf) {
                gl.TexSubImage2D(target, 0, 0, 0, width, height, fmt.format, fmt.type, view.buf);
            }
        }
    }

    ClearValue clear_value = {};
    if (fmt.buffer == GL_DEPTH || fmt.buffer == GL_DEPTH_STENCIL) {
        clear_value.clear_floats[0] = 1.0f;
    }

    Image * res = PyObject_New(Image, self->module_state->Image_type);
    res->gc_prev = self->gc_prev;
    res->gc_next = (GCHeader *)self;
    res->gc_prev->gc_next = (GCHeader *)res;
    res->gc_next->gc_prev = (GCHeader *)res;
    res->ctx = (Context *)new_ref(self);
    res->size = Py_BuildValue("(ii)", width, height);
    res->format = (PyObject *)new_ref(format);
    res->faces = PyDict_New();
    res->clear_value = clear_value;
    res->fmt = fmt;
    res->image = image;
    res->width = width;
    res->height = height;
    res->samples = samples;
    res->array = array;
    res->cubemap = cubemap;
    res->target = target;
    res->renderbuffer = renderbuffer;
    res->max_level = 0;

    res->framebuffer = NULL;
    if (!cubemap && !array) {
        if (fmt.color) {
            PyObject * face = PyObject_CallMethod((PyObject *)res, "face", NULL);
            PyObject * attachments = Py_BuildValue("((ii)(N)O)", width, height, face, Py_None);
            res->framebuffer = build_framebuffer(self, attachments);
            Py_DECREF(attachments);
        } else {
            PyObject * face = PyObject_CallMethod((PyObject *)res, "face", NULL);
            PyObject * attachments = Py_BuildValue("((ii)()N)", width, height, face);
            res->framebuffer = build_framebuffer(self, attachments);
            Py_DECREF(attachments);
        }
    }

    if (data != Py_None) {
        PyBuffer_Release(&view);
    }

    Py_INCREF(res);
    return res;
}

static Pipeline * Context_meth_pipeline(Context * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {
        "vertex_shader",
        "fragment_shader",
        "layout",
        "resources",
        "uniforms",
        "depth",
        "stencil",
        "blend",
        "framebuffer",
        "vertex_buffers",
        "index_buffer",
        "short_index",
        "cull_face",
        "topology",
        "vertex_count",
        "instance_count",
        "first_vertex",
        "viewport",
        "includes",
        NULL,
    };

    PyObject * vertex_shader = NULL;
    PyObject * fragment_shader = NULL;
    PyObject * layout = self->module_state->empty_tuple;
    PyObject * resources = self->module_state->empty_tuple;
    PyObject * uniforms = Py_None;
    PyObject * depth = Py_None;
    PyObject * stencil = Py_None;
    PyObject * blend = Py_None;
    PyObject * framebuffer_images = self->module_state->empty_tuple;
    PyObject * vertex_buffers = self->module_state->empty_tuple;
    PyObject * index_buffer = Py_None;
    int short_index = false;
    PyObject * cull_face = self->module_state->str_none;
    int topology = GL_TRIANGLES;
    int vertex_count = 0;
    int instance_count = 1;
    int first_vertex = 0;
    PyObject * viewport = Py_None;
    PyObject * includes = Py_None;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "|O!O!OOOOOOOOOpOO&iiiOO",
        keywords,
        &PyUnicode_Type,
        &vertex_shader,
        &PyUnicode_Type,
        &fragment_shader,
        &layout,
        &resources,
        &uniforms,
        &depth,
        &stencil,
        &blend,
        &framebuffer_images,
        &vertex_buffers,
        &index_buffer,
        &short_index,
        &cull_face,
        topology_converter,
        &topology,
        &vertex_count,
        &instance_count,
        &first_vertex,
        &viewport,
        &includes
    );

    if (!args_ok) {
        return NULL;
    }

    if (!vertex_shader || !fragment_shader || !framebuffer_images) {
        if (!vertex_shader) {
            PyErr_Format(PyExc_TypeError, "no vertex_shader was specified");
        } else if (!fragment_shader) {
            PyErr_Format(PyExc_TypeError, "no fragment_shader was specified");
        } else if (!framebuffer_images) {
            PyErr_Format(PyExc_TypeError, "no framebuffer was specified");
        }
        return NULL;
    }

    if (viewport != Py_None && !is_viewport(viewport)) {
        PyErr_Format(PyExc_TypeError, "the viewport must be a tuple of 4 ints");
        return NULL;
    }

    int index_size = short_index ? 2 : 4;
    int index_type = index_buffer != Py_None ? (short_index ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT) : 0;

    GLObject * program = compile_program(self, includes != Py_None ? includes : self->includes, vertex_shader, fragment_shader, layout);
    if (!program) {
        return NULL;
    }

    PyObject * uniform_map = NULL;
    char * uniform_data = NULL;

    if (uniforms != Py_None) {
        PyObject * tuple = PyObject_CallMethod(self->module_state->helper, "uniforms", "OO", program->extra, uniforms);
        if (!tuple) {
            return NULL;
        }

        PyObject * names = PyTuple_GetItem(tuple, 0);
        PyObject * data = PyTuple_GetItem(tuple, 1);
        PyObject * mapping = PyDict_New();

        uniform_data = (char *)PyMem_Malloc(PyByteArray_Size(data));
        memcpy(uniform_data, PyByteArray_AsString(data), PyByteArray_Size(data));
        int offset = 0;
        int idx = 0;

        while (true) {
            UniformBinding * header = (UniformBinding *)(uniform_data + offset);
            if (header->type == 0) {
                break;
            }
            PyObject * name = PyList_GetItem(names, idx++);
            PyObject * mem = PyMemoryView_FromMemory(uniform_data + offset + 16, header->values * 4, PyBUF_WRITE);
            PyDict_SetItem(mapping, name, mem);
            Py_DECREF(mem);
            offset += header->values * 4 + 16;
        }

        uniform_map = PyDictProxy_New(mapping);
        Py_DECREF(mapping);
        Py_DECREF(tuple);
    }

    PyObject * attachments = PyObject_CallMethod(self->module_state->helper, "framebuffer_attachments", "(O)", framebuffer_images);
    if (!attachments) {
        return NULL;
    }

    PyObject * validate = PyObject_CallMethod(
        self->module_state->helper,
        "validate",
        "OOOOOO",
        program->extra,
        layout,
        resources,
        vertex_buffers,
        PyTuple_GetItem(attachments, 1),
        self->limits_dict
    );

    if (!validate) {
        return NULL;
    }

    const GLMethods & gl = self->gl;
    int layout_count = layout != Py_None ? (int)PyList_Size(layout) : 0; // TODO: check
    for (int i = 0; i < layout_count; ++i) {
        PyObject * obj = PyList_GetItem(layout, i);
        PyObject * name = PyDict_GetItemString(obj, "name");
        int binding = PyLong_AsLong(PyDict_GetItemString(obj, "binding"));
        int location = gl.GetUniformLocation(program->obj, PyUnicode_AsUTF8(name));
        if (location >= 0) {
            gl.Uniform1i(location, binding);
        } else {
            int index = gl.GetUniformBlockIndex(program->obj, PyUnicode_AsUTF8(name));
            gl.UniformBlockBinding(program->obj, index, binding);
        }
    }

    Viewport viewport_value = {};
    if (viewport != Py_None) {
        viewport_value = to_viewport(viewport);
    } else {
        PyObject * size = PyTuple_GetItem(attachments, 0);
        viewport_value.width = PyLong_AsLong(PyTuple_GetItem(size, 0));
        viewport_value.height = PyLong_AsLong(PyTuple_GetItem(size, 1));
    }

    GLObject * framebuffer = build_framebuffer(self, attachments);

    PyObject * bindings = PyObject_CallMethod(self->module_state->helper, "vertex_array_bindings", "OO", vertex_buffers, index_buffer);
    if (!bindings) {
        return NULL;
    }

    GLObject * vertex_array = build_vertex_array(self, bindings);
    Py_DECREF(bindings);

    PyObject * resource_bindings = PyObject_CallMethod(self->module_state->helper, "resource_bindings", "(O)", resources);
    if (!resource_bindings) {
        return NULL;
    }

    DescriptorSet * descriptor_set = build_descriptor_set(self, resource_bindings);
    Py_DECREF(resource_bindings);

    PyObject * settings = PyObject_CallMethod(
        self->module_state->helper,
        "settings",
        "OOOON",
        cull_face,
        depth,
        stencil,
        blend,
        attachments
    );

    if (!settings) {
        return NULL;
    }

    GlobalSettings * global_settings = build_global_settings(self, settings);
    Py_DECREF(settings);

    Pipeline * res = PyObject_New(Pipeline, self->module_state->Pipeline_type);
    res->gc_prev = self->gc_prev;
    res->gc_next = (GCHeader *)self;
    res->gc_prev->gc_next = (GCHeader *)res;
    res->gc_next->gc_prev = (GCHeader *)res;
    res->ctx = (Context *)new_ref(self);
    res->framebuffer = framebuffer;
    res->vertex_array = vertex_array;
    res->program = program;
    res->uniform_map = uniform_map;
    res->uniform_data = uniform_data;
    res->topology = topology;
    res->vertex_count = vertex_count;
    res->instance_count = instance_count;
    res->first_vertex = first_vertex;
    res->index_type = index_type;
    res->index_size = index_size;
    res->viewport = viewport_value;
    res->descriptor_set = descriptor_set;
    res->global_settings = global_settings;
    Py_INCREF(res);
    return res;
}

static PyObject * Context_meth_new_frame(Context * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"reset", "clear", "frame_time", NULL};

    int reset = true;
    int clear = true;
    int frame_time = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ppp", keywords, &reset, &clear, &frame_time)) {
        return NULL;
    }

    const GLMethods & gl = self->gl;

    if (self->before_frame_callback != Py_None) {
        PyObject * temp = PyObject_CallObject(self->before_frame_callback, NULL);
        Py_XDECREF(temp);
        if (!temp) {
            return NULL;
        }
    }

    if (reset) {
        self->current_descriptor_set = NULL;
        self->current_global_settings = NULL;
        self->is_stencil_default = false;
        self->is_mask_default = false;
        self->is_blend_default = false;
        self->current_viewport = {-1, -1, -1, -1};
        self->current_framebuffer = -1;
        self->current_program = -1;
        self->current_vertex_array = -1;
        self->current_depth_mask = 0;
        self->current_stencil_mask = 0;
    }

    if (clear) {
        bind_framebuffer(self, 0);
        gl.Clear(GL_COLOR_BUFFER_BIT);
    }

    if (frame_time) {
        if (!self->frame_time_query) {
            gl.GenQueries(1, (unsigned *)&self->frame_time_query);
        }
        gl.BeginQuery(GL_TIME_ELAPSED, self->frame_time_query);
        self->frame_time_query_running = true;
        self->frame_time = 0;
    }

    gl.Enable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    gl.Enable(GL_PROGRAM_POINT_SIZE);
    gl.Enable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    gl.Enable(GL_FRAMEBUFFER_SRGB);
    Py_RETURN_NONE;
}

static PyObject * Context_meth_end_frame(Context * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"clean", "flush", "sync", NULL};

    int clean = true;
    int flush = true;
    int sync = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ppp", keywords, &clean, &flush, &sync)) {
        return NULL;
    }

    const GLMethods & gl = self->gl;

    if (clean) {
        bind_framebuffer(self, 0);
        bind_program(self, 0);
        bind_vertex_array(self, 0);

        self->current_descriptor_set = NULL;
        self->current_global_settings = NULL;

        // gl.BindBuffersRange(GL_UNIFORM_BUFFER, 0, self->limits.max_uniform_buffer_bindings, NULL, NULL, NULL);
        // gl.BindTextures(0, self->limits.max_combined_texture_image_units, NULL);
        // gl.BindSamplers(0, self->limits.max_combined_texture_image_units, NULL);

        gl.Disable(GL_CULL_FACE);
        gl.Disable(GL_DEPTH_TEST);
        gl.Disable(GL_STENCIL_TEST);
        gl.Disable(GL_BLEND);

        gl.Disable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        gl.Disable(GL_PROGRAM_POINT_SIZE);
        gl.Disable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
        gl.Disable(GL_FRAMEBUFFER_SRGB);
    }

    if (self->frame_time_query_running) {
        gl.EndQuery(GL_TIME_ELAPSED);
        gl.GetQueryObjectuiv(self->frame_time_query, GL_QUERY_RESULT, (unsigned *)&self->frame_time);
        self->frame_time_query_running = false;
    } else {
        self->frame_time = 0;
    }

    if (flush) {
        gl.Flush();
    }

    if (sync) {
        void * sync = gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl.ClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
        gl.DeleteSync(sync);
    }

    if (self->after_frame_callback != Py_None) {
        PyObject * temp = PyObject_CallObject(self->after_frame_callback, NULL);
        Py_XDECREF(temp);
        if (!temp) {
            return NULL;
        }
    }

    Py_RETURN_NONE;
}

static void release_descriptor_set(Context * self, DescriptorSet * set) {
    set->uses -= 1;
    if (!set->uses) {
        for (int i = 0; i < set->samplers.sampler_count; ++i) {
            GLObject * sampler = (GLObject *)set->samplers.sampler_refs[i];
            sampler->uses -= 1;
            if (!sampler->uses) {
                remove_dict_value(self->sampler_cache, (PyObject *)sampler);
                self->gl.DeleteSamplers(1, (unsigned int *)&sampler->obj);
            }
        }
        for (int i = 0; i < set->uniform_buffers.buffer_count; ++i) {
            Py_XDECREF(set->uniform_buffers.buffer_refs[i]);
        }
        for (int i = 0; i < set->samplers.sampler_count; ++i) {
            Py_XDECREF(set->samplers.sampler_refs[i]);
            Py_XDECREF(set->samplers.texture_refs[i]);
        }
        remove_dict_value(self->descriptor_set_cache, (PyObject *)set);
        if (self->current_descriptor_set == set) {
            self->current_descriptor_set = NULL;
        }
    }
}

static void release_global_settings(Context * self, GlobalSettings * settings) {
    settings->uses -= 1;
    if (!settings->uses) {
        remove_dict_value(self->global_settings_cache, (PyObject *)settings);
        if (self->current_global_settings == settings) {
            self->current_global_settings = NULL;
        }
    }
}

static void release_framebuffer(Context * self, GLObject * framebuffer) {
    framebuffer->uses -= 1;
    if (!framebuffer->uses) {
        remove_dict_value(self->framebuffer_cache, (PyObject *)framebuffer);
        if (self->current_framebuffer == framebuffer->obj) {
            self->current_framebuffer = 0;
        }
        self->gl.DeleteFramebuffers(1, (unsigned int *)&framebuffer->obj);
    }
}

static void release_program(Context * self, GLObject * program) {
    program->uses -= 1;
    if (!program->uses) {
        remove_dict_value(self->program_cache, (PyObject *)program);
        if (self->current_program == program->obj) {
            self->current_program = 0;
        }
        self->gl.DeleteProgram(program->obj);
    }
}

static void release_vertex_array(Context * self, GLObject * vertex_array) {
    vertex_array->uses -= 1;
    if (!vertex_array->uses) {
        remove_dict_value(self->vertex_array_cache, (PyObject *)vertex_array);
        if (self->current_vertex_array == vertex_array->obj) {
            self->current_vertex_array = 0;
        }
        self->gl.DeleteVertexArrays(1, (unsigned int *)&vertex_array->obj);
    }
}

static PyObject * Context_meth_release(Context * self, PyObject * arg) {
    const GLMethods & gl = self->gl;
    if (Py_TYPE(arg) == self->module_state->Buffer_type) {
        Buffer * buffer = (Buffer *)arg;
        buffer->gc_prev->gc_next = buffer->gc_next;
        buffer->gc_next->gc_prev = buffer->gc_prev;
        gl.DeleteBuffers(1, (unsigned int *)&buffer->buffer);
        Py_DECREF(arg);
    } else if (Py_TYPE(arg) == self->module_state->Image_type) {
        Image * image = (Image *)arg;
        image->gc_prev->gc_next = image->gc_next;
        image->gc_next->gc_prev = image->gc_prev;
        if (image->framebuffer) {
            release_framebuffer(self, image->framebuffer);
        }
        if (image->faces) {
            PyObject * key = NULL;
            PyObject * value = NULL;
            Py_ssize_t pos = 0;
            while (PyDict_Next(image->faces, &pos, &key, &value)) {
                release_framebuffer(self, (GLObject *)value);
            }
            PyDict_Clear(self->shader_cache);
        }
        if (image->renderbuffer) {
            gl.DeleteRenderbuffers(1, (unsigned int *)&image->image);
        } else {
            gl.DeleteTextures(1, (unsigned int *)&image->image);
        }
        Py_DECREF(arg);
    } else if (Py_TYPE(arg) == self->module_state->Pipeline_type) {
        Pipeline * pipeline = (Pipeline *)arg;
        pipeline->gc_prev->gc_next = pipeline->gc_next;
        pipeline->gc_next->gc_prev = pipeline->gc_prev;
        release_descriptor_set(self, pipeline->descriptor_set);
        release_global_settings(self, pipeline->global_settings);
        release_framebuffer(self, pipeline->framebuffer);
        release_program(self, pipeline->program);
        release_vertex_array(self, pipeline->vertex_array);
        if (pipeline->uniform_data) {
            PyMem_Free(pipeline->uniform_data);
            Py_DECREF(pipeline->uniform_map);
        }
        Py_DECREF(pipeline);
    } else if (PyUnicode_CheckExact(arg) && !PyUnicode_CompareWithASCIIString(arg, "shader_cache")) {
        PyObject * key = NULL;
        PyObject * value = NULL;
        Py_ssize_t pos = 0;
        while (PyDict_Next(self->shader_cache, &pos, &key, &value)) {
            GLObject * shader = (GLObject *)value;
            gl.DeleteShader(shader->obj);
        }
        PyDict_Clear(self->shader_cache);
    } else if (PyUnicode_CheckExact(arg) && !PyUnicode_CompareWithASCIIString(arg, "all")) {
        GCHeader * it = self->gc_next;
        while (it != (GCHeader *)self) {
            GCHeader * next = it->gc_next;
            if (Py_TYPE(it) == self->module_state->Pipeline_type) {
                Py_DECREF(Context_meth_release(self, (PyObject *)it));
            }
            it = next;
        }
        it = self->gc_next;
        while (it != (GCHeader *)self) {
            GCHeader * next = it->gc_next;
            if (Py_TYPE(it) == self->module_state->Buffer_type) {
                Py_DECREF(Context_meth_release(self, (PyObject *)it));
            } else if (Py_TYPE(it) == self->module_state->Image_type) {
                Py_DECREF(Context_meth_release(self, (PyObject *)it));
            }
            it = next;
        }
    }
    Py_RETURN_NONE;
}

static PyObject * Buffer_meth_write(Buffer * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "offset", NULL};

    Py_buffer view;
    int offset = 0;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "y*|i", keywords, &view, &offset)) {
        return NULL;
    }

    const bool already_mapped = self->mapped;
    const bool invalid_offset = offset < 0 || offset > self->size;
    const bool invalid_size = (int)view.len + offset > self->size;

    if (already_mapped || invalid_offset || invalid_size) {
        PyBuffer_Release(&view);
        if (already_mapped) {
            PyErr_Format(PyExc_RuntimeError, "already mapped");
        } else if (invalid_offset) {
            PyErr_Format(PyExc_ValueError, "invalid offset");
        } else if (invalid_size) {
            PyErr_Format(PyExc_ValueError, "invalid size");
        }
        return NULL;
    }

    const GLMethods & gl = self->ctx->gl;

    if (view.len) {
        gl.BindBuffer(GL_ARRAY_BUFFER, self->buffer);
        gl.BufferSubData(GL_ARRAY_BUFFER, offset, (int)view.len, view.buf);
    }

    PyBuffer_Release(&view);
    Py_RETURN_NONE;
}

static PyObject * Buffer_meth_map(Buffer * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"size", "offset", "discard", NULL};

    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;
    int discard = false;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|OOp", keywords, &size_arg, &offset_arg, &discard)) {
        return NULL;
    }

    int size = self->size;
    int offset = 0;

    const bool invalid_size_type = size_arg != Py_None && !PyLong_CheckExact(size_arg);
    const bool invalid_offset_type = offset_arg != Py_None && !PyLong_CheckExact(offset_arg);

    if (size_arg != Py_None && !invalid_size_type) {
        size = PyLong_AsLong(size_arg);
    }

    if (offset_arg != Py_None && !invalid_offset_type) {
        offset = PyLong_AsLong(offset_arg);
    }

    const bool already_mapped = self->mapped;
    const bool offset_but_no_size = size_arg == Py_None && offset_arg != Py_None;
    const bool invalid_size = invalid_size_type || size <= 0 || size > self->size;
    const bool invalid_offset = invalid_offset_type || offset < 0 || offset + size > self->size;

    if (already_mapped || offset_but_no_size || invalid_size || invalid_offset) {
        if (already_mapped) {
            PyErr_Format(PyExc_RuntimeError, "already mapped");
        } else if (offset_but_no_size) {
            PyErr_Format(PyExc_ValueError, "the size is required when the offset is not None");
        } else if (invalid_size_type) {
            PyErr_Format(PyExc_TypeError, "the size must be an int or None");
        } else if (invalid_offset_type) {
            PyErr_Format(PyExc_TypeError, "the offset must be an int or None");
        } else if (invalid_size) {
            PyErr_Format(PyExc_ValueError, "invalid size");
        } else if (invalid_offset) {
            PyErr_Format(PyExc_ValueError, "invalid offset");
        }
        return NULL;
    }

    const GLMethods & gl = self->ctx->gl;

    self->mapped = true;
    self->ctx->mapped_buffers += 1;
    const int access = discard ? GL_MAP_READ_BIT | GL_MAP_WRITE_BIT : GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_WRITE_BIT;
    gl.BindBuffer(GL_ARRAY_BUFFER, self->buffer);
    void * ptr = gl.MapBufferRange(GL_ARRAY_BUFFER, offset, size, access);
    return PyMemoryView_FromMemory((char *)ptr, size, PyBUF_WRITE);
}

static PyObject * Buffer_meth_unmap(Buffer * self, PyObject * args) {
    const GLMethods & gl = self->ctx->gl;
    if (self->mapped) {
        self->mapped = false;
        self->ctx->mapped_buffers -= 1;
        gl.BindBuffer(GL_ARRAY_BUFFER, self->buffer);
        gl.UnmapBuffer(GL_ARRAY_BUFFER);
    }
    Py_RETURN_NONE;
}

static void clear_bound_image(Image * self) {
    const GLMethods & gl = self->ctx->gl;
    const bool depth_mask = self->ctx->current_depth_mask != 1 && (self->fmt.buffer == GL_DEPTH || self->fmt.buffer == GL_DEPTH_STENCIL);
    const bool stencil_mask = self->ctx->current_stencil_mask != 0xff && (self->fmt.buffer == GL_STENCIL || self->fmt.buffer == GL_DEPTH_STENCIL);
    if (depth_mask) {
        gl.DepthMask(1);
        self->ctx->current_depth_mask = 1;
    }
    if (stencil_mask) {
        gl.StencilMaskSeparate(GL_FRONT, 0xff);
        self->ctx->current_stencil_mask = 0xff;
    }
    if (self->fmt.clear_type == 'f') {
        gl.ClearBufferfv(self->fmt.buffer, 0, self->clear_value.clear_floats);
    } else if (self->fmt.clear_type == 'i') {
        gl.ClearBufferiv(self->fmt.buffer, 0, self->clear_value.clear_ints);
    } else if (self->fmt.clear_type == 'u') {
        gl.ClearBufferuiv(self->fmt.buffer, 0, self->clear_value.clear_uints);
    } else if (self->fmt.clear_type == 'x') {
        gl.ClearBufferfi(self->fmt.buffer, 0, self->clear_value.clear_floats[0], self->clear_value.clear_ints[1]);
    }
}

static PyObject * Image_meth_clear(Image * self, PyObject * args) {
    if (!self->framebuffer) {
        PyErr_Format(PyExc_TypeError, "cannot clear cubemap or array textures");
        return NULL;
    }
    bind_framebuffer(self->ctx, self->framebuffer->obj);
    clear_bound_image(self);
    Py_RETURN_NONE;
}

static PyObject * Image_meth_write(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "size", "offset", "layer", "level", NULL};

    Py_buffer view;
    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;
    PyObject * layer_arg = Py_None;
    int level = 0;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "y*|OOOi", keywords, &view, &size_arg, &offset_arg, &layer_arg, &level)) {
        return NULL;
    }

    IntPair size = {};
    IntPair offset = {};
    int layer = 0;

    const bool invalid_size_type = size_arg != Py_None && !is_int_pair(size_arg);
    const bool invalid_offset_type = offset_arg != Py_None && !is_int_pair(offset_arg);
    const bool invalid_layer_type = layer_arg != Py_None && !PyLong_CheckExact(layer_arg);

    if (size_arg != Py_None && !invalid_size_type) {
        size = to_int_pair(size_arg);
    } else {
        size.x = max(self->width >> level, 1);
        size.y = max(self->height >> level, 1);
    }

    if (offset_arg != Py_None && !invalid_offset_type) {
        offset = to_int_pair(offset_arg);
    }

    if (layer_arg != Py_None && !invalid_layer_type) {
        layer = PyLong_AsLong(layer_arg);
    }

    const bool offset_but_no_size = size_arg == Py_None && offset_arg != Py_None;
    const bool invalid_size = invalid_size_type || size.x <= 0 || size.y <= 0 || size.x > self->width || size.y > self->height;
    const bool invalid_offset = invalid_offset_type || offset.x < 0 || offset.y < 0 || size.x + offset.x > self->width || size.y + offset.y > self->height;
    const bool invalid_layer = invalid_layer_type || layer < 0 || layer >= (self->array ? self->array : 1) * (self->cubemap ? 6 : 1);
    const bool invalid_level = level < 0 || level > self->max_level;
    const bool layer_but_simple = !self->cubemap && !self->array && layer_arg != Py_None;
    const bool invalid_type = !self->fmt.color || self->samples != 1;

    if (offset_but_no_size || invalid_size || invalid_offset || invalid_layer || invalid_level || layer_but_simple || invalid_type) {
        PyBuffer_Release(&view);
        if (offset_but_no_size) {
            PyErr_Format(PyExc_ValueError, "the size is required when the offset is not None");
        } else if (invalid_size_type) {
            PyErr_Format(PyExc_TypeError, "the size must be a tuple of 2 ints");
        } else if (invalid_offset_type) {
            PyErr_Format(PyExc_TypeError, "the offset must be a tuple of 2 ints");
        } else if (invalid_layer_type) {
            PyErr_Format(PyExc_TypeError, "the layer must be an int or None");
        } else if (invalid_size) {
            PyErr_Format(PyExc_ValueError, "invalid size");
        } else if (invalid_offset) {
            PyErr_Format(PyExc_ValueError, "invalid offset");
        } else if (invalid_layer) {
            PyErr_Format(PyExc_ValueError, "invalid layer");
        } else if (invalid_level) {
            PyErr_Format(PyExc_ValueError, "invalid level");
        } else if (layer_but_simple) {
            PyErr_Format(PyExc_TypeError, "the image is not layered");
        } else if (!self->fmt.color) {
            PyErr_Format(PyExc_TypeError, "cannot write to depth or stencil images");
        } else if (self->samples != 1) {
            PyErr_Format(PyExc_TypeError, "cannot write to multisampled images");
        }
        return NULL;
    }

    int padded_row = (size.x * self->fmt.pixel_size + 3) & ~3;
    int expected_size = padded_row * size.y;

    if (layer_arg == Py_None) {
        expected_size *= (self->array ? self->array : 1) * (self->cubemap ? 6 : 1);
    }

    if ((int)view.len != expected_size) {
        PyBuffer_Release(&view);
        PyErr_Format(PyExc_ValueError, "invalid data size, expected %d, got %d", expected_size, (int)view.len);
        return NULL;
    }

    const GLMethods & gl = self->ctx->gl;

    gl.ActiveTexture(self->ctx->default_texture_unit);
    gl.BindTexture(self->target, self->image);
    if (self->cubemap) {
        if (layer_arg != Py_None) {
            gl.TexSubImage3D(self->target, level, offset.x, offset.y, layer, size.x, size.y, 1, self->fmt.format, self->fmt.type, view.buf);
        } else {
            gl.TexSubImage3D(self->target, level, offset.x, offset.y, 0, size.x, size.y, 6, self->fmt.format, self->fmt.type, view.buf);
        }
    } else if (self->array) {
        if (layer_arg != Py_None) {
            gl.TexSubImage3D(self->target, level, offset.x, offset.y, layer, size.x, size.y, 1, self->fmt.format, self->fmt.type, view.buf);
        } else {
            gl.TexSubImage3D(self->target, level, offset.x, offset.y, 0, size.x, size.y, self->array, self->fmt.format, self->fmt.type, view.buf);
        }
    } else {
        gl.TexSubImage2D(self->target, level, offset.x, offset.y, size.x, size.y, self->fmt.format, self->fmt.type, view.buf);
    }

    PyBuffer_Release(&view);
    Py_RETURN_NONE;
}

static PyObject * Image_meth_mipmaps(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"base", "levels", NULL};

    int base = 0;
    PyObject * levels_arg = Py_None;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|iO", keywords, &base, &levels_arg)) {
        return NULL;
    }

    int max_levels = count_mipmaps(self->width, self->height);

    const bool invalid_levels_type = levels_arg != Py_None && !PyLong_CheckExact(levels_arg);

    int levels = max_levels - base;
    if (levels_arg != Py_None && !invalid_levels_type) {
        levels = PyLong_AsLong(levels_arg);
    }

    const bool invalid_base = base < 0 || base >= max_levels;
    const bool invalid_levels = levels <= 0 || base + levels > max_levels;

    if (invalid_levels_type || invalid_base || invalid_levels) {
        if (invalid_levels_type) {
            PyErr_Format(PyExc_TypeError, "levels must be an int");
        } else if (invalid_base) {
            PyErr_Format(PyExc_ValueError, "invalid base");
        } else if (invalid_levels) {
            PyErr_Format(PyExc_ValueError, "invalid levels");
        }
        return NULL;
    }

    if (self->max_level < base + levels) {
        self->max_level = base + levels;
    }

    const GLMethods & gl = self->ctx->gl;
    gl.BindTexture(self->target, self->image);
    gl.TexParameteri(self->target, GL_TEXTURE_BASE_LEVEL, base);
    gl.TexParameteri(self->target, GL_TEXTURE_MAX_LEVEL, base + levels);
    gl.GenerateMipmap(self->target);
    Py_RETURN_NONE;
}

static PyObject * Image_meth_read(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"size", "offset", NULL};

    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|OO", keywords, &size_arg, &offset_arg)) {
        return NULL;
    }

    IntPair size = {};
    IntPair offset = {};

    const bool invalid_size_type = size_arg != Py_None && !is_int_pair(size_arg);
    const bool invalid_offset_type = offset_arg != Py_None && !is_int_pair(offset_arg);

    if (size_arg != Py_None && !invalid_size_type) {
        size = to_int_pair(size_arg);
    } else {
        size.x = self->width;
        size.y = self->height;
    }

    if (offset_arg != Py_None && !invalid_offset_type) {
        offset = to_int_pair(offset_arg);
    }

    const bool offset_but_no_size = size_arg == Py_None && offset_arg != Py_None;
    const bool invalid_size = invalid_size_type || size.x <= 0 || size.y <= 0 || size.x > self->width || size.y > self->height;
    const bool invalid_offset = invalid_offset_type || offset.x < 0 || offset.y < 0 || size.x + offset.x > self->width || size.y + offset.y > self->height;
    const bool invalid_type = self->cubemap || self->array || self->samples != 1;

    if (offset_but_no_size || invalid_size || invalid_offset || invalid_type) {
        if (offset_but_no_size) {
            PyErr_Format(PyExc_ValueError, "the size is required when the offset is not None");
        } else if (invalid_size_type) {
            PyErr_Format(PyExc_TypeError, "the size must be a tuple of 2 ints");
        } else if (invalid_offset_type) {
            PyErr_Format(PyExc_TypeError, "the offset must be a tuple of 2 ints");
        } else if (invalid_size) {
            PyErr_Format(PyExc_ValueError, "invalid size");
        } else if (invalid_offset) {
            PyErr_Format(PyExc_ValueError, "invalid offset");
        } else if (self->cubemap) {
            PyErr_Format(PyExc_TypeError, "cannot read cubemap images");
        } else if (self->array) {
            PyErr_Format(PyExc_TypeError, "cannot read array images");
        } else if (self->samples != 1) {
            PyErr_Format(PyExc_TypeError, "multisampled images must be blit to a non multisampled image before read");
        }
        return NULL;
    }

    const GLMethods & gl = self->ctx->gl;

    PyObject * res = PyBytes_FromStringAndSize(NULL, (long long)size.x * size.y * self->fmt.pixel_size);
    bind_framebuffer(self->ctx, self->framebuffer->obj);
    gl.ReadPixels(offset.x, offset.y, size.x, size.y, self->fmt.format, self->fmt.type, PyBytes_AS_STRING(res));
    return res;
}

static PyObject * Image_meth_blit(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"target", "target_viewport", "source_viewport", "filter", "srgb", NULL};

    PyObject * target_arg = Py_None;
    PyObject * target_viewport_arg = Py_None;
    PyObject * source_viewport_arg = Py_None;
    int filter = true;
    PyObject * srgb_arg = Py_None;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "|OOOpO",
        keywords,
        &target_arg,
        &target_viewport_arg,
        &source_viewport_arg,
        &filter,
        &srgb_arg
    );

    if (!args_ok) {
        return NULL;
    }

    const bool invalid_target_type = target_arg != Py_None && Py_TYPE(target_arg) != self->ctx->module_state->Image_type;
    const bool invalid_srgb_parameter = srgb_arg != Py_True && srgb_arg != Py_False && srgb_arg != Py_None;

    Image * target = target_arg != Py_None && !invalid_target_type ? (Image *)target_arg : NULL;

    Viewport target_viewport = {};
    Viewport source_viewport = {};

    const bool invalid_target_viewport_type = target_viewport_arg != Py_None && !is_viewport(target_viewport_arg);
    const bool invalid_source_viewport_type = source_viewport_arg != Py_None && !is_viewport(source_viewport_arg);

    if (target_viewport_arg != Py_None && !invalid_target_viewport_type) {
        target_viewport = to_viewport(target_viewport_arg);
    } else {
        target_viewport.width = target ? target->width : self->width;
        target_viewport.height = target ? target->height : self->height;
    }

    if (source_viewport_arg != Py_None && !invalid_source_viewport_type) {
        source_viewport = to_viewport(source_viewport_arg);
    } else {
        source_viewport.width = self->width;
        source_viewport.height = self->height;
    }

    const bool srgb = (srgb_arg == Py_None && self->fmt.internal_format == GL_SRGB8_ALPHA8) || srgb_arg == Py_True;

    const bool invalid_target_viewport = invalid_target_viewport_type || (
        target_viewport.x < 0 || target_viewport.y < 0 || target_viewport.width <= 0 || target_viewport.height <= 0 ||
        (target && (target_viewport.x + target_viewport.width > target->width || target_viewport.y + target_viewport.height > target->height))
    );

    const bool invalid_source_viewport = invalid_source_viewport_type || (
        source_viewport.x < 0 || source_viewport.y < 0 || source_viewport.width <= 0 || source_viewport.height <= 0 ||
        source_viewport.x + source_viewport.width > self->width || source_viewport.y + source_viewport.height > self->height
    );

    const bool invalid_target = target && (target->cubemap || target->array || !target->fmt.color || target->samples > 1);
    const bool invalid_source = self->cubemap || self->array || !self->fmt.color;

    const bool error = (
        invalid_target_type || invalid_srgb_parameter || invalid_target_viewport_type ||
        invalid_source_viewport_type || invalid_target_viewport || invalid_source_viewport ||
        invalid_target || invalid_source
    );

    if (error) {
        if (invalid_target_type) {
            PyErr_Format(PyExc_TypeError, "target must be an Image or None");
        } else if (invalid_srgb_parameter) {
            PyErr_Format(PyExc_TypeError, "invalid srgb parameter");
        } else if (invalid_target_viewport_type) {
            PyErr_Format(PyExc_TypeError, "the target viewport must be a tuple of 4 ints");
        } else if (invalid_source_viewport_type) {
            PyErr_Format(PyExc_TypeError, "the source viewport must be a tuple of 4 ints");
        } else if (invalid_target_viewport) {
            PyErr_Format(PyExc_ValueError, "the target viewport is out of range");
        } else if (invalid_source_viewport) {
            PyErr_Format(PyExc_ValueError, "the source viewport is out of range");
        } else if (self->cubemap) {
            PyErr_Format(PyExc_TypeError, "cannot blit cubemap images");
        } else if (self->array) {
            PyErr_Format(PyExc_TypeError, "cannot blit array images");
        } else if (!self->fmt.color) {
            PyErr_Format(PyExc_TypeError, "cannot blit depth or stencil images");
        } else if (target && target->cubemap) {
            PyErr_Format(PyExc_TypeError, "cannot blit to cubemap images");
        } else if (target && target->array) {
            PyErr_Format(PyExc_TypeError, "cannot blit to array images");
        } else if (target && !target->fmt.color) {
            PyErr_Format(PyExc_TypeError, "cannot blit to depth or stencil images");
        } else if (target && target->samples > 1) {
            PyErr_Format(PyExc_TypeError, "cannot blit to multisampled images");
        }
        return NULL;
    }

    const GLMethods & gl = self->ctx->gl;

    if (!srgb) {
        gl.Disable(GL_FRAMEBUFFER_SRGB);
    }

    int target_framebuffer = target ? target->framebuffer->obj : self->ctx->screen;
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, self->framebuffer->obj);
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, target_framebuffer);
    gl.BlitFramebuffer(
        source_viewport.x, source_viewport.y, source_viewport.x + source_viewport.width, source_viewport.y + source_viewport.height,
        target_viewport.x, target_viewport.y, target_viewport.x + target_viewport.width, target_viewport.y + target_viewport.height,
        GL_COLOR_BUFFER_BIT, filter ? GL_LINEAR : GL_NEAREST
    );
    self->ctx->current_framebuffer = -1;

    if (!srgb) {
        gl.Enable(GL_FRAMEBUFFER_SRGB);
    }

    Py_RETURN_NONE;
}

static ImageFace * Image_meth_face(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"layer", "level", NULL};

    int layer = 0;
    int level = 0;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|ii", keywords, &layer, &level)) {
        return NULL;
    }

    if (layer < 0 || layer >= (self->array ? self->array : 1) * (self->cubemap ? 6 : 1) || level > self->max_level) {
        return NULL;
    }

    PyObject * key = Py_BuildValue("(ii)", layer, level);
    if (ImageFace * cache = (ImageFace *)PyDict_GetItem(self->faces, key)) {
        Py_DECREF(key);
        Py_INCREF(cache);
        return cache;
    }

    int width = max(self->width >> level, 1);
    int height = max(self->height >> level, 1);

    ImageFace * res = PyObject_New(ImageFace, self->ctx->module_state->ImageFace_type);
    res->gc_prev = self->gc_prev;
    res->gc_next = (GCHeader *)self;
    res->gc_prev->gc_next = (GCHeader *)res;
    res->gc_next->gc_prev = (GCHeader *)res;
    res->ctx = (Context *)new_ref(self->ctx);
    res->image = (Image *)new_ref(self);
    res->size = Py_BuildValue("(ii)", width, height);
    res->width = width;
    res->height = height;
    res->layer = layer;
    res->level = level;
    res->samples = self->samples;
    res->flags = self->fmt.flags;

    res->framebuffer = NULL;
    if (self->fmt.color) {
        PyObject * attachments = Py_BuildValue("((ii)(O)O)", width, height, res, Py_None);
        res->framebuffer = build_framebuffer(self->ctx, attachments);
        Py_DECREF(attachments);
    } else {
        PyObject * attachments = Py_BuildValue("((ii)()O)", width, height, res);
        res->framebuffer = build_framebuffer(self->ctx, attachments);
        Py_DECREF(attachments);
    }

    PyDict_SetItem(self->faces, key, (PyObject *)res);
    Py_DECREF(key);
    return res;
}

static PyObject * Image_get_clear_value(Image * self, void * closure) {
    if (self->fmt.clear_type == 'x') {
        return Py_BuildValue("fi", self->clear_value.clear_floats[0], self->clear_value.clear_ints[1]);
    }
    if (self->fmt.components == 1) {
        if (self->fmt.clear_type == 'f') {
            return PyFloat_FromDouble(self->clear_value.clear_floats[0]);
        } else if (self->fmt.clear_type == 'i') {
            return PyLong_FromLong(self->clear_value.clear_ints[0]);
        } else if (self->fmt.clear_type == 'u') {
            return PyLong_FromUnsignedLong(self->clear_value.clear_uints[0]);
        }
    }
    PyObject * res = PyTuple_New(self->fmt.components);
    for (int i = 0; i < self->fmt.components; ++i) {
        if (self->fmt.clear_type == 'f') {
            PyTuple_SetItem(res, i, PyFloat_FromDouble(self->clear_value.clear_floats[i]));
        } else if (self->fmt.clear_type == 'i') {
            PyTuple_SetItem(res, i, PyLong_FromLong(self->clear_value.clear_ints[i]));
        } else if (self->fmt.clear_type == 'u') {
            PyTuple_SetItem(res, i, PyLong_FromUnsignedLong(self->clear_value.clear_uints[i]));
        }
    }
    return res;
}

static int Image_set_clear_value(Image * self, PyObject * value, void * closure) {
    ClearValue clear_value = {};
    if (self->fmt.components == 1) {
        if (self->fmt.clear_type == 'f' ? !PyFloat_CheckExact(value) : !PyLong_CheckExact(value)) {
            if (self->fmt.clear_type == 'f') {
                PyErr_Format(PyExc_TypeError, "the clear value must be a float");
            } else {
                PyErr_Format(PyExc_TypeError, "the clear value must be an int");
            }
            return -1;
        }
        if (self->fmt.clear_type == 'f') {
            clear_value.clear_floats[0] = (float)PyFloat_AsDouble(value);
        } else if (self->fmt.clear_type == 'i') {
            clear_value.clear_ints[0] = PyLong_AsLong(value);
        } else if (self->fmt.clear_type == 'u') {
            clear_value.clear_uints[0] = PyLong_AsUnsignedLong(value);
        }
        self->clear_value = clear_value;
        return 0;
    }
    PyObject * values = PySequence_Fast(value, "");
    if (!values) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError, "the clear value must be a tuple");
        return -1;
    }

    int size = (int)PySequence_Fast_GET_SIZE(values);
    PyObject ** seq = PySequence_Fast_ITEMS(values);

    if (size != self->fmt.components) {
        Py_DECREF(values);
        PyErr_Format(PyExc_ValueError, "invalid clear value size");
        return -1;
    }

    if (self->fmt.clear_type == 'f') {
        for (int i = 0; i < self->fmt.components; ++i) {
            clear_value.clear_floats[i] = (float)PyFloat_AsDouble(seq[i]);
        }
    } else if (self->fmt.clear_type == 'i') {
        for (int i = 0; i < self->fmt.components; ++i) {
            clear_value.clear_ints[i] = PyLong_AsLong(seq[i]);
        }
    } else if (self->fmt.clear_type == 'u') {
        for (int i = 0; i < self->fmt.components; ++i) {
            clear_value.clear_uints[i] = PyLong_AsUnsignedLong(seq[i]);
        }
    } else if (self->fmt.clear_type == 'x') {
        clear_value.clear_floats[0] = (float)PyFloat_AsDouble(seq[0]);
        clear_value.clear_ints[1] = PyLong_AsLong(seq[1]);
    }
    if (PyErr_Occurred()) {
        Py_DECREF(values);
        return -1;
    }
    self->clear_value = clear_value;
    Py_DECREF(values);
    return 0;
}

static PyObject * Pipeline_meth_render(Pipeline * self, PyObject * args) {
    const GLMethods & gl = self->ctx->gl;
    if (memcmp(&self->viewport, &self->ctx->current_viewport, sizeof(Viewport))) {
        gl.Viewport(self->viewport.x, self->viewport.y, self->viewport.width, self->viewport.height);
        self->ctx->current_viewport = self->viewport;
    }
    bind_global_settings(self->ctx, self->global_settings);
    bind_framebuffer(self->ctx, self->framebuffer->obj);
    bind_program(self->ctx, self->program->obj);
    bind_vertex_array(self->ctx, self->vertex_array->obj);
    bind_descriptor_set(self->ctx, self->descriptor_set);
    if (self->uniform_data) {
        bind_uniforms(self->ctx, self->uniform_data);
    }
    if (self->index_type) {
        long long offset = (long long)self->first_vertex * self->index_size;
        gl.DrawElementsInstanced(self->topology, self->vertex_count, self->index_type, (void *)offset, self->instance_count);
    } else {
        gl.DrawArraysInstanced(self->topology, self->first_vertex, self->vertex_count, self->instance_count);
    }
    Py_RETURN_NONE;
}

static PyObject * Pipeline_get_viewport(Pipeline * self, void * closure) {
    return Py_BuildValue("iiii", self->viewport.x, self->viewport.y, self->viewport.width, self->viewport.height);
}

static int Pipeline_set_viewport(Pipeline * self, PyObject * viewport, void * closure) {
    if (!is_viewport(viewport)) {
        PyErr_Format(PyExc_TypeError, "the viewport must be a tuple of 4 ints");
        return -1;
    }
    self->viewport = to_viewport(viewport);
    return 0;
}

static PyObject * inspect_descriptor_set(DescriptorSet * set) {
    PyObject * res = PyList_New(0);
    for (int i = 0; i < set->uniform_buffers.buffer_count; ++i) {
        if (set->uniform_buffers.buffer_refs[i]) {
            PyObject * obj = Py_BuildValue(
                "{sssisisisi}",
                "type", "uniform_buffer",
                "binding", i,
                "buffer", set->uniform_buffers.buffers[i],
                "offset", set->uniform_buffers.buffer_offsets[i],
                "size", set->uniform_buffers.buffer_sizes[i]
            );
            PyList_Append(res, obj);
            Py_DECREF(obj);
        }
    }
    for (int i = 0; i < set->samplers.sampler_count; ++i) {
        if (set->samplers.sampler_refs[i]) {
            PyObject * obj = Py_BuildValue(
                "{sssisisi}",
                "type", "sampler",
                "binding", i,
                "sampler", set->samplers.samplers[i],
                "texture", set->samplers.textures[i]
            );
            PyList_Append(res, obj);
            Py_DECREF(obj);
        }
    }
    return res;
}

static PyObject * meth_inspect(PyObject * self, PyObject * arg) {
    ModuleState * module_state = (ModuleState *)PyModule_GetState(self);
    if (Py_TYPE(arg) == module_state->Buffer_type) {
        Buffer * buffer = (Buffer *)arg;
        return Py_BuildValue("{sssi}", "type", "buffer", "buffer", buffer->buffer);
    } else if (Py_TYPE(arg) == module_state->Image_type) {
        Image * image = (Image *)arg;
        const char * gltype = image->renderbuffer ? "renderbuffer" : "texture";
        int framebuffer = image->framebuffer ? image->framebuffer->obj : -1;
        return Py_BuildValue("{sssisi}", "type", gltype, gltype, image->image, "framebuffer", framebuffer);
    } else if (Py_TYPE(arg) == module_state->Pipeline_type) {
        Pipeline * pipeline = (Pipeline *)arg;
        return Py_BuildValue(
            "{sssNsNsisisi}",
            "type", "pipeline",
            "interface", pipeline->program->extra,
            "resources", inspect_descriptor_set(pipeline->descriptor_set),
            "framebuffer", pipeline->framebuffer->obj,
            "vertex_array", pipeline->vertex_array->obj,
            "program", pipeline->program->obj
        );
    }
    Py_RETURN_NONE;
}

static PyObject * ImageFace_meth_clear(ImageFace * self, PyObject * args) {
    bind_framebuffer(self->ctx, self->framebuffer->obj);
    clear_bound_image(self->image);
    Py_RETURN_NONE;
}

static PyObject * ImageFace_meth_blit(ImageFace * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"target", "target_viewport", "source_viewport", "filter", "srgb", NULL};

    ImageFace * target = NULL;
    PyObject * target_viewport_arg = Py_None;
    PyObject * source_viewport_arg = Py_None;
    int filter = true;
    PyObject * srgb_arg = Py_None;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "O!|OOpO",
        keywords,
        self->image->ctx->module_state->ImageFace_type,
        &target,
        &target_viewport_arg,
        &source_viewport_arg,
        &filter,
        &srgb_arg
    );

    if (!args_ok) {
        return NULL;
    }

    const bool invalid_srgb_parameter = srgb_arg != Py_True && srgb_arg != Py_False && srgb_arg != Py_None;

    Viewport target_viewport = {};
    Viewport source_viewport = {};

    const bool invalid_target_viewport_type = target_viewport_arg != Py_None && !is_viewport(target_viewport_arg);
    const bool invalid_source_viewport_type = source_viewport_arg != Py_None && !is_viewport(source_viewport_arg);

    if (target_viewport_arg != Py_None && !invalid_target_viewport_type) {
        target_viewport = to_viewport(target_viewport_arg);
    } else {
        target_viewport.width = target ? target->width : self->width;
        target_viewport.height = target ? target->height : self->height;
    }

    if (source_viewport_arg != Py_None && !invalid_source_viewport_type) {
        source_viewport = to_viewport(source_viewport_arg);
    } else {
        source_viewport.width = self->width;
        source_viewport.height = self->height;
    }

    const bool srgb = (srgb_arg == Py_None && self->image->fmt.internal_format == GL_SRGB8_ALPHA8) || srgb_arg == Py_True;

    const bool invalid_target_viewport = invalid_target_viewport_type || (
        target_viewport.x < 0 || target_viewport.y < 0 || target_viewport.width <= 0 || target_viewport.height <= 0 ||
        (target && (target_viewport.x + target_viewport.width > target->width || target_viewport.y + target_viewport.height > target->height))
    );

    const bool invalid_source_viewport = invalid_source_viewport_type || (
        source_viewport.x < 0 || source_viewport.y < 0 || source_viewport.width <= 0 || source_viewport.height <= 0 ||
        source_viewport.x + source_viewport.width > self->width || source_viewport.y + source_viewport.height > self->height
    );

    const bool invalid_target = target->samples > 1 || !(target->flags & 1);
    const bool invalid_source = !(self->flags & 1);

    const bool error = (
        invalid_srgb_parameter || invalid_target_viewport ||
        invalid_source_viewport || invalid_target || invalid_source
    );

    if (error) {
        return NULL;
    }

    const GLMethods & gl = self->image->ctx->gl;

    if (!srgb) {
        gl.Disable(GL_FRAMEBUFFER_SRGB);
    }

    int target_framebuffer = target ? target->framebuffer->obj : self->ctx->screen;
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, self->framebuffer->obj);
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, target_framebuffer);
    gl.BlitFramebuffer(
        source_viewport.x, source_viewport.y, source_viewport.x + source_viewport.width, source_viewport.y + source_viewport.height,
        target_viewport.x, target_viewport.y, target_viewport.x + target_viewport.width, target_viewport.y + target_viewport.height,
        GL_COLOR_BUFFER_BIT, filter ? GL_LINEAR : GL_NEAREST
    );
    self->ctx->current_framebuffer = -1;

    if (!srgb) {
        gl.Enable(GL_FRAMEBUFFER_SRGB);
    }

    Py_RETURN_NONE;
}

struct vec3 {
    double x, y, z;
};

static vec3 operator - (const vec3 & a, const vec3 & b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

static vec3 normalize(const vec3 & a) {
    const double l = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    return {a.x / l, a.y / l, a.z / l};
}

static vec3 cross(const vec3 & a, const vec3 & b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

static double dot(const vec3 & a, const vec3 & b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static PyObject * meth_camera(PyObject * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"eye", "target", "up", "fov", "aspect", "near", "far", "size", "clip", NULL};

    vec3 eye;
    vec3 target;
    vec3 up = {0.0, 0.0, 1.0};
    double fov = 60.0;
    double aspect = 1.0;
    double znear = 0.1;
    double zfar = 1000.0;
    double size = 1.0;
    int clip = false;

    int args_ok = PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "(ddd)(ddd)|(ddd)dddddp",
        keywords,
        &eye.x,
        &eye.y,
        &eye.z,
        &target.x,
        &target.y,
        &target.z,
        &up.x,
        &up.y,
        &up.z,
        &fov,
        &aspect,
        &znear,
        &zfar,
        &size,
        &clip
    );

    if (!args_ok) {
        return NULL;
    }

    const vec3 f = normalize(target - eye);
    const vec3 s = normalize(cross(f, up));
    const vec3 u = cross(s, f);
    const vec3 t = {-dot(s, eye), -dot(u, eye), -dot(f, eye)};

    if (!fov) {
        const double r1 = size;
        const double r2 = r1 * aspect;
        const double r3 = clip ? 1.0 / (zfar - znear) : 2.0 / (zfar - znear);
        const double r4 = clip ? znear / (zfar - znear) : (zfar + znear) / (zfar - znear);

        float res[] = {
            (float)(s.x / r2), (float)(u.x / r1), (float)(r3 * f.x), 0.0f,
            (float)(s.y / r2), (float)(u.y / r1), (float)(r3 * f.y), 0.0f,
            (float)(s.z / r2), (float)(u.z / r1), (float)(r3 * f.z), 0.0f,
            (float)(t.x / r2), (float)(t.y / r1), (float)(r3 * t.z - r4), 1.0f,
        };

        return PyBytes_FromStringAndSize((char *)res, 64);
    }

    const double r1 = tan(fov * 0.008726646259971647884618453842);
    const double r2 = r1 * aspect;
    const double r3 = clip ? zfar / (zfar - znear) : (zfar + znear) / (zfar - znear);
    const double r4 = clip ? (zfar * znear) / (zfar - znear) : (2.0 * zfar * znear) / (zfar - znear);

    float res[] = {
        (float)(s.x / r2), (float)(u.x / r1), (float)(r3 * f.x), (float)f.x,
        (float)(s.y / r2), (float)(u.y / r1), (float)(r3 * f.y), (float)f.y,
        (float)(s.z / r2), (float)(u.z / r1), (float)(r3 * f.z), (float)f.z,
        (float)(t.x / r2), (float)(t.y / r1), (float)(r3 * t.z - r4), (float)t.z,
    };

    return PyBytes_FromStringAndSize((char *)res, 64);
}

static void Context_dealloc(Context * self) {
    Py_DECREF(self->descriptor_set_cache);
    Py_DECREF(self->global_settings_cache);
    Py_DECREF(self->sampler_cache);
    Py_DECREF(self->vertex_array_cache);
    Py_DECREF(self->framebuffer_cache);
    Py_DECREF(self->program_cache);
    Py_DECREF(self->shader_cache);
    Py_DECREF(self->includes);
    Py_DECREF(self->limits_dict);
    Py_DECREF(self->info_dict);
    Py_TYPE(self)->tp_free(self);
}

static void Buffer_dealloc(Buffer * self) {
    Py_DECREF(self->ctx);
    Py_TYPE(self)->tp_free(self);
}

static void Image_dealloc(Image * self) {
    Py_DECREF(self->ctx);
    Py_DECREF(self->framebuffer);
    Py_DECREF(self->faces);
    Py_DECREF(self->size);
    Py_TYPE(self)->tp_free(self);
}

static void Pipeline_dealloc(Pipeline * self) {
    Py_DECREF(self->ctx);
    Py_DECREF(self->descriptor_set);
    Py_DECREF(self->global_settings);
    Py_DECREF(self->framebuffer);
    Py_DECREF(self->program);
    Py_DECREF(self->vertex_array);
    Py_TYPE(self)->tp_free(self);
}

static void ImageFace_dealloc(ImageFace * self) {
    Py_TYPE(self)->tp_free(self);
}

static void DescriptorSet_dealloc(DescriptorSet * self) {
    Py_TYPE(self)->tp_free(self);
}

static void GlobalSettings_dealloc(GlobalSettings * self) {
    Py_TYPE(self)->tp_free(self);
}

static void GLObject_dealloc(GLObject * self) {
    Py_TYPE(self)->tp_free(self);
}

static PyMethodDef Context_methods[] = {
    {"buffer", (PyCFunction)Context_meth_buffer, METH_VARARGS | METH_KEYWORDS},
    {"image", (PyCFunction)Context_meth_image, METH_VARARGS | METH_KEYWORDS},
    {"pipeline", (PyCFunction)Context_meth_pipeline, METH_VARARGS | METH_KEYWORDS},
    {"new_frame", (PyCFunction)Context_meth_new_frame, METH_VARARGS | METH_KEYWORDS},
    {"end_frame", (PyCFunction)Context_meth_end_frame, METH_VARARGS | METH_KEYWORDS},
    {"release", (PyCFunction)Context_meth_release, METH_O},
    {},
};

static PyMemberDef Context_members[] = {
    {"includes", T_OBJECT, offsetof(Context, includes), READONLY},
    {"limits", T_OBJECT, offsetof(Context, limits_dict), READONLY},
    {"info", T_OBJECT, offsetof(Context, info_dict), READONLY},
    {"before_frame", T_OBJECT, offsetof(Context, before_frame_callback), 0},
    {"after_frame", T_OBJECT, offsetof(Context, after_frame_callback), 0},
    {"frame_time", T_INT, offsetof(Context, frame_time), READONLY},
    {"screen", T_INT, offsetof(Context, screen), 0},
    {},
};

static PyMethodDef Buffer_methods[] = {
    {"write", (PyCFunction)Buffer_meth_write, METH_VARARGS | METH_KEYWORDS},
    {"map", (PyCFunction)Buffer_meth_map, METH_VARARGS | METH_KEYWORDS},
    {"unmap", (PyCFunction)Buffer_meth_unmap, METH_NOARGS},
    {},
};

static PyMemberDef Buffer_members[] = {
    {"size", T_INT, offsetof(Buffer, size), READONLY},
    {},
};

static PyMethodDef Image_methods[] = {
    {"clear", (PyCFunction)Image_meth_clear, METH_NOARGS},
    {"write", (PyCFunction)Image_meth_write, METH_VARARGS | METH_KEYWORDS},
    {"read", (PyCFunction)Image_meth_read, METH_VARARGS | METH_KEYWORDS},
    {"mipmaps", (PyCFunction)Image_meth_mipmaps, METH_VARARGS | METH_KEYWORDS},
    {"blit", (PyCFunction)Image_meth_blit, METH_VARARGS | METH_KEYWORDS},
    {"face", (PyCFunction)Image_meth_face, METH_VARARGS | METH_KEYWORDS},
    {},
};

static PyGetSetDef Image_getset[] = {
    {"clear_value", (getter)Image_get_clear_value, (setter)Image_set_clear_value},
    {},
};

static PyMemberDef Image_members[] = {
    {"size", T_OBJECT, offsetof(Image, size), READONLY},
    {"format", T_OBJECT, offsetof(Image, format), READONLY},
    {"samples", T_INT, offsetof(Image, samples), READONLY},
    {"array", T_INT, offsetof(Image, array), READONLY},
    {"flags", T_INT, offsetof(Image, fmt.flags), READONLY},
    {},
};

static PyMethodDef Pipeline_methods[] = {
    {"render", (PyCFunction)Pipeline_meth_render, METH_NOARGS},
    {},
};

static PyGetSetDef Pipeline_getset[] = {
    {"viewport", (getter)Pipeline_get_viewport, (setter)Pipeline_set_viewport},
    {},
};

static PyMemberDef Pipeline_members[] = {
    {"vertex_count", T_INT, offsetof(Pipeline, vertex_count), 0},
    {"instance_count", T_INT, offsetof(Pipeline, instance_count), 0},
    {"first_vertex", T_INT, offsetof(Pipeline, first_vertex), 0},
    {"uniforms", T_OBJECT_EX, offsetof(Pipeline, uniform_map), READONLY},
    {},
};

static PyMethodDef ImageFace_methods[] = {
    {"clear", (PyCFunction)ImageFace_meth_clear, METH_NOARGS},
    {"blit", (PyCFunction)ImageFace_meth_blit, METH_VARARGS | METH_KEYWORDS},
    {},
};

static PyMemberDef ImageFace_members[] = {
    {"image", T_OBJECT, offsetof(ImageFace, image), READONLY},
    {"size", T_OBJECT, offsetof(ImageFace, size), READONLY},
    {"layer", T_INT, offsetof(ImageFace, layer), READONLY},
    {"level", T_INT, offsetof(ImageFace, level), READONLY},
    {"samples", T_INT, offsetof(ImageFace, samples), READONLY},
    {"flags", T_INT, offsetof(ImageFace, flags), READONLY},
    {},
};

static PyType_Slot Context_slots[] = {
    {Py_tp_methods, Context_methods},
    {Py_tp_members, Context_members},
    {Py_tp_dealloc, (void *)Context_dealloc},
    {},
};

static PyType_Slot Buffer_slots[] = {
    {Py_tp_methods, Buffer_methods},
    {Py_tp_members, Buffer_members},
    {Py_tp_dealloc, (void *)Buffer_dealloc},
    {},
};

static PyType_Slot Image_slots[] = {
    {Py_tp_methods, Image_methods},
    {Py_tp_getset, Image_getset},
    {Py_tp_members, Image_members},
    {Py_tp_dealloc, (void *)Image_dealloc},
    {},
};

static PyType_Slot Pipeline_slots[] = {
    {Py_tp_methods, Pipeline_methods},
    {Py_tp_getset, Pipeline_getset},
    {Py_tp_members, Pipeline_members},
    {Py_tp_dealloc, (void *)Pipeline_dealloc},
    {},
};

static PyType_Slot ImageFace_slots[] = {
    {Py_tp_methods, ImageFace_methods},
    {Py_tp_members, ImageFace_members},
    {Py_tp_dealloc, (void *)ImageFace_dealloc},
    {},
};

static PyType_Slot DescriptorSet_slots[] = {
    {Py_tp_dealloc, (void *)DescriptorSet_dealloc},
    {},
};

static PyType_Slot GlobalSettings_slots[] = {
    {Py_tp_dealloc, (void *)GlobalSettings_dealloc},
    {},
};

static PyType_Slot GLObject_slots[] = {
    {Py_tp_dealloc, (void *)GLObject_dealloc},
    {},
};

static PyType_Spec Context_spec = {"zengl.Context", sizeof(Context), 0, Py_TPFLAGS_DEFAULT, Context_slots};
static PyType_Spec Buffer_spec = {"zengl.Buffer", sizeof(Buffer), 0, Py_TPFLAGS_DEFAULT, Buffer_slots};
static PyType_Spec Image_spec = {"zengl.Image", sizeof(Image), 0, Py_TPFLAGS_DEFAULT, Image_slots};
static PyType_Spec Pipeline_spec = {"zengl.Pipeline", sizeof(Pipeline), 0, Py_TPFLAGS_DEFAULT, Pipeline_slots};
static PyType_Spec ImageFace_spec = {"zengl.ImageFace", sizeof(ImageFace), 0, Py_TPFLAGS_DEFAULT, ImageFace_slots};
static PyType_Spec DescriptorSet_spec = {"zengl.DescriptorSet", sizeof(DescriptorSet), 0, Py_TPFLAGS_DEFAULT, DescriptorSet_slots};
static PyType_Spec GlobalSettings_spec = {"zengl.GlobalSettings", sizeof(GlobalSettings), 0, Py_TPFLAGS_DEFAULT, GlobalSettings_slots};
static PyType_Spec GLObject_spec = {"zengl.GLObject", sizeof(GLObject), 0, Py_TPFLAGS_DEFAULT, GLObject_slots};

static int module_exec(PyObject * self) {
    ModuleState * state = (ModuleState *)PyModule_GetState(self);

    state->helper = PyImport_ImportModule("_zengl");
    if (!state->helper) {
        return -1;
    }

    state->empty_tuple = PyTuple_New(0);
    state->str_none = PyUnicode_FromString("none");
    state->float_one = PyFloat_FromDouble(1.0);
    state->Context_type = (PyTypeObject *)PyType_FromSpec(&Context_spec);
    state->Buffer_type = (PyTypeObject *)PyType_FromSpec(&Buffer_spec);
    state->Image_type = (PyTypeObject *)PyType_FromSpec(&Image_spec);
    state->Pipeline_type = (PyTypeObject *)PyType_FromSpec(&Pipeline_spec);
    state->ImageFace_type = (PyTypeObject *)PyType_FromSpec(&ImageFace_spec);
    state->DescriptorSet_type = (PyTypeObject *)PyType_FromSpec(&DescriptorSet_spec);
    state->GlobalSettings_type = (PyTypeObject *)PyType_FromSpec(&GlobalSettings_spec);
    state->GLObject_type = (PyTypeObject *)PyType_FromSpec(&GLObject_spec);

    PyModule_AddObject(self, "Context", (PyObject *)new_ref(state->Context_type));
    PyModule_AddObject(self, "Buffer", (PyObject *)new_ref(state->Buffer_type));
    PyModule_AddObject(self, "Image", (PyObject *)new_ref(state->Image_type));
    PyModule_AddObject(self, "Pipeline", (PyObject *)new_ref(state->Pipeline_type));

    PyModule_AddObject(self, "loader", (PyObject *)new_ref(PyObject_GetAttrString(state->helper, "loader")));
    PyModule_AddObject(self, "calcsize", (PyObject *)new_ref(PyObject_GetAttrString(state->helper, "calcsize")));
    PyModule_AddObject(self, "bind", (PyObject *)new_ref(PyObject_GetAttrString(state->helper, "bind")));

    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, (void *)module_exec},
    {},
};

static PyMethodDef module_methods[] = {
    {"context", (PyCFunction)meth_context, METH_VARARGS | METH_KEYWORDS},
    {"inspect", (PyCFunction)meth_inspect, METH_O},
    {"camera", (PyCFunction)meth_camera, METH_VARARGS | METH_KEYWORDS},
    {},
};

static void module_free(PyObject * self) {
    ModuleState * state = (ModuleState *)PyModule_GetState(self);
    if (!state) {
        return;
    }
    Py_DECREF(state->empty_tuple);
    Py_DECREF(state->str_none);
    Py_DECREF(state->float_one);
    Py_DECREF(state->Context_type);
    Py_DECREF(state->Buffer_type);
    Py_DECREF(state->Image_type);
    Py_DECREF(state->Pipeline_type);
    Py_DECREF(state->ImageFace_type);
    Py_DECREF(state->DescriptorSet_type);
    Py_DECREF(state->GlobalSettings_type);
    Py_DECREF(state->GLObject_type);
}

static PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "zengl", NULL, sizeof(ModuleState), module_methods, module_slots, NULL, NULL, (freefunc)module_free,
};

extern "C" PyObject * PyInit_zengl() {
    return PyModuleDef_Init(&module_def);
}
