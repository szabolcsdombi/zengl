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
#define GL_RGBA8 0x8058
#define GL_TEXTURE_WRAP_R 0x8072
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

typedef struct GLMethods {
    void (GLAPI * CullFace)(GLenum mode);
    void (GLAPI * Clear)(GLbitfield mask);
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
    void (GLAPI * VertexAttribDivisor)(GLuint index, GLuint divisor);
} GLMethods;

typedef struct VertexFormat {
    const char * name;
    int type;
    int size;
    int normalize;
    int integer;
} VertexFormat;

typedef struct ImageFormat {
    const char * name;
    int internal_format;
    int format;
    int type;
    int components;
    int pixel_size;
    int buffer;
    int color;
    int clear_type;
    int flags;
} ImageFormat;

typedef struct UniformBufferBinding {
    int buffer;
    int offset;
    int size;
} UniformBufferBinding;

typedef struct SamplerBinding {
    int sampler;
    int target;
    int image;
} SamplerBinding;

typedef struct UniformBinding {
    int values;
    int location;
    int count;
    int type;
    union {
        int int_values[1];
        unsigned uint_values[1];
        float float_values[1];
    };
} UniformBinding;

typedef struct StencilSettings {
    int fail_op;
    int pass_op;
    int depth_fail_op;
    int compare_op;
    int compare_mask;
    int write_mask;
    int reference;
} StencilSettings;

typedef struct Viewport {
    int x;
    int y;
    int width;
    int height;
} Viewport;

typedef union ClearValue {
    float clear_floats[4];
    int clear_ints[4];
    unsigned int clear_uints[4];
} ClearValue;

typedef struct IntPair {
    int x;
    int y;
} IntPair;

static int least_one(int value) {
    return value > 1 ? value : 1;
}

static const int num_vertex_formats = 30;
static const VertexFormat vertex_formats[] = {
    {"uint8x2", GL_UNSIGNED_BYTE, 2, 0, 1},
    {"uint8x4", GL_UNSIGNED_BYTE, 4, 0, 1},
    {"sint8x2", GL_BYTE, 2, 0, 1},
    {"sint8x4", GL_BYTE, 4, 0, 1},
    {"unorm8x2", GL_UNSIGNED_BYTE, 2, 1, 0},
    {"unorm8x4", GL_UNSIGNED_BYTE, 4, 1, 0},
    {"snorm8x2", GL_BYTE, 2, 1, 0},
    {"snorm8x4", GL_BYTE, 4, 1, 0},
    {"uint16x2", GL_UNSIGNED_SHORT, 2, 0, 1},
    {"uint16x4", GL_UNSIGNED_SHORT, 4, 0, 1},
    {"sint16x2", GL_SHORT, 2, 0, 1},
    {"sint16x4", GL_SHORT, 4, 0, 1},
    {"unorm16x2", GL_UNSIGNED_SHORT, 2, 1, 0},
    {"unorm16x4", GL_UNSIGNED_SHORT, 4, 1, 0},
    {"snorm16x2", GL_SHORT, 2, 1, 0},
    {"snorm16x4", GL_SHORT, 4, 1, 0},
    {"float16x2", GL_HALF_FLOAT, 2, 0, 0},
    {"float16x4", GL_HALF_FLOAT, 4, 0, 0},
    {"float32", GL_FLOAT, 1, 0, 0},
    {"float32x2", GL_FLOAT, 2, 0, 0},
    {"float32x3", GL_FLOAT, 3, 0, 0},
    {"float32x4", GL_FLOAT, 4, 0, 0},
    {"uint32", GL_UNSIGNED_INT, 1, 0, 1},
    {"uint32x2", GL_UNSIGNED_INT, 2, 0, 1},
    {"uint32x3", GL_UNSIGNED_INT, 3, 0, 1},
    {"uint32x4", GL_UNSIGNED_INT, 4, 0, 1},
    {"sint32", GL_INT, 1, 0, 1},
    {"sint32x2", GL_INT, 2, 0, 1},
    {"sint32x3", GL_INT, 3, 0, 1},
    {"sint32x4", GL_INT, 4, 0, 1},
};

static const int num_image_formats = 35;
static const ImageFormat image_formats[] = {
    {"r8unorm", GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1, 1, GL_COLOR, 1, 'f', 1},
    {"rg8unorm", GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2, 2, GL_COLOR, 1, 'f', 1},
    {"rgba8unorm", GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, 1, 'f', 1},
    {"r8snorm", GL_R8_SNORM, GL_RED, GL_UNSIGNED_BYTE, 1, 1, GL_COLOR, 1, 'f', 1},
    {"rg8snorm", GL_RG8_SNORM, GL_RG, GL_UNSIGNED_BYTE, 2, 2, GL_COLOR, 1, 'f', 1},
    {"rgba8snorm", GL_RGBA8_SNORM, GL_RGBA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, 1, 'f', 1},
    {"r8uint", GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, 1, 1, GL_COLOR, 1, 'u', 1},
    {"rg8uint", GL_RG8UI, GL_RG_INTEGER, GL_UNSIGNED_BYTE, 2, 2, GL_COLOR, 1, 'u', 1},
    {"rgba8uint", GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, 1, 'u', 1},
    {"r16uint", GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT, 1, 2, GL_COLOR, 1, 'u', 1},
    {"rg16uint", GL_RG16UI, GL_RG_INTEGER, GL_UNSIGNED_SHORT, 2, 4, GL_COLOR, 1, 'u', 1},
    {"rgba16uint", GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, 4, 8, GL_COLOR, 1, 'u', 1},
    {"r32uint", GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 1, 4, GL_COLOR, 1, 'u', 1},
    {"rg32uint", GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT, 2, 8, GL_COLOR, 1, 'u', 1},
    {"rgba32uint", GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 4, 16, GL_COLOR, 1, 'u', 1},
    {"r8sint", GL_R8I, GL_RED_INTEGER, GL_BYTE, 1, 1, GL_COLOR, 1, 'i', 1},
    {"rg8sint", GL_RG8I, GL_RG_INTEGER, GL_BYTE, 2, 2, GL_COLOR, 1, 'i', 1},
    {"rgba8sint", GL_RGBA8I, GL_RGBA_INTEGER, GL_BYTE, 4, 4, GL_COLOR, 1, 'i', 1},
    {"r16sint", GL_R16I, GL_RED_INTEGER, GL_SHORT, 1, 2, GL_COLOR, 1, 'i', 1},
    {"rg16sint", GL_RG16I, GL_RG_INTEGER, GL_SHORT, 2, 4, GL_COLOR, 1, 'i', 1},
    {"rgba16sint", GL_RGBA16I, GL_RGBA_INTEGER, GL_SHORT, 4, 8, GL_COLOR, 1, 'i', 1},
    {"r32sint", GL_R32I, GL_RED_INTEGER, GL_INT, 1, 4, GL_COLOR, 1, 'i', 1},
    {"rg32sint", GL_RG32I, GL_RG_INTEGER, GL_INT, 2, 8, GL_COLOR, 1, 'i', 1},
    {"rgba32sint", GL_RGBA32I, GL_RGBA_INTEGER, GL_INT, 4, 16, GL_COLOR, 1, 'i', 1},
    {"r16float", GL_R16F, GL_RED, GL_FLOAT, 1, 2, GL_COLOR, 1, 'f', 1},
    {"rg16float", GL_RG16F, GL_RG, GL_FLOAT, 2, 4, GL_COLOR, 1, 'f', 1},
    {"rgba16float", GL_RGBA16F, GL_RGBA, GL_FLOAT, 4, 8, GL_COLOR, 1, 'f', 1},
    {"r32float", GL_R32F, GL_RED, GL_FLOAT, 1, 4, GL_COLOR, 1, 'f', 1},
    {"rg32float", GL_RG32F, GL_RG, GL_FLOAT, 2, 8, GL_COLOR, 1, 'f', 1},
    {"rgba32float", GL_RGBA32F, GL_RGBA, GL_FLOAT, 4, 16, GL_COLOR, 1, 'f', 1},
    {"rgba8unorm-srgb", GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, 1, 'f', 1},
    {"depth16unorm", GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, 1, 2, GL_DEPTH, 0, 'f', 2},
    {"depth24plus", GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 1, 4, GL_DEPTH, 0, 'f', 2},
    {"depth24plus-stencil8", GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 2, 4, GL_DEPTH_STENCIL, 0, 'x', 6},
    {"depth32float", GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, 4, GL_DEPTH, 0, 'f', 2},
};

static const VertexFormat * get_vertex_format(const char * format) {
    for (int i = 0; i < num_vertex_formats; ++i) {
        if (!strcmp(format, vertex_formats[i].name)) {
            return vertex_formats + i;
        }
    }
    return NULL;
}

static const ImageFormat * get_image_format(const char * format) {
    for (int i = 0; i < num_image_formats; ++i) {
        if (!strcmp(format, image_formats[i].name)) {
            return image_formats + i;
        }
    }
    return NULL;
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

PyObject * contiguous(PyObject * data) {
    PyObject * mem = PyMemoryView_FromObject(data);
    if (!mem) {
        return NULL;
    }

    if (PyBuffer_IsContiguous(PyMemoryView_GET_BUFFER(mem), 'C')) {
        return mem;
    }

    PyObject * bytes = PyObject_Bytes(mem);
    Py_XDECREF(mem);
    if (!bytes) {
        return NULL;
    }

    PyObject * res = PyMemoryView_FromObject(bytes);
    Py_XDECREF(bytes);
    return res;
}

static IntPair to_int_pair(PyObject * obj, int x, int y) {
    IntPair res;
    if (obj != Py_None) {
        res.x = PyLong_AsLong(PySequence_GetItem(obj, 0));
        res.y = PyLong_AsLong(PySequence_GetItem(obj, 1));
    } else {
        res.x = x;
        res.y = y;
    }
    return res;
}

static Viewport to_viewport(PyObject * obj, int x, int y, int width, int height) {
    Viewport res;
    if (obj != Py_None) {
        res.x = PyLong_AsLong(PySequence_GetItem(obj, 0));
        res.y = PyLong_AsLong(PySequence_GetItem(obj, 1));
        res.width = PyLong_AsLong(PySequence_GetItem(obj, 2));
        res.height = PyLong_AsLong(PySequence_GetItem(obj, 3));
    } else {
        res.x = x;
        res.y = y;
        res.width = width;
        res.height = height;
    }
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

static void load_gl(GLMethods * gl, PyObject * loader) {
    PyObject * missing = PyList_New(0);

    #define check(name) if (!gl->name) { if (PyErr_Occurred()) return; PyList_Append(missing, PyUnicode_FromString("gl" # name)); }
    #define load(name) *(void **)&gl->name = load_opengl_function(loader, "gl" # name); check(name)

    load(CullFace);
    load(Clear);
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
    load(VertexAttribDivisor);

    #undef load
    #undef check

    if (PyList_Size(missing)) {
        PyErr_Format(PyExc_RuntimeError, "cannot load opengl %R", missing);
        return;
    }

    Py_DECREF(missing);
}

typedef struct Limits {
    int max_uniform_buffer_bindings;
    int max_uniform_block_size;
    int max_combined_uniform_blocks;
    int max_combined_texture_image_units;
    int max_vertex_attribs;
    int max_draw_buffers;
    int max_samples;
} Limits;

typedef struct ModuleState {
    PyObject * helper;
    PyObject * empty_tuple;
    PyObject * str_none;
    PyTypeObject * Context_type;
    PyTypeObject * Buffer_type;
    PyTypeObject * Image_type;
    PyTypeObject * Pipeline_type;
    PyTypeObject * ImageFace_type;
    PyTypeObject * DescriptorSet_type;
    PyTypeObject * GlobalSettings_type;
    PyTypeObject * GLObject_type;
} ModuleState;

typedef struct GCHeader {
    PyObject_HEAD
    struct GCHeader * gc_prev;
    struct GCHeader * gc_next;
} GCHeader;

typedef struct GLObject {
    PyObject_HEAD
    int uses;
    int obj;
    PyObject * extra;
} GLObject;

typedef struct DescriptorSetBuffers {
    int buffer_count;
    unsigned buffers[MAX_UNIFORM_BUFFER_BINDINGS];
    GLsizeiptr buffer_offsets[MAX_UNIFORM_BUFFER_BINDINGS];
    GLsizeiptr buffer_sizes[MAX_UNIFORM_BUFFER_BINDINGS];
    PyObject * buffer_refs[MAX_UNIFORM_BUFFER_BINDINGS];
} DescriptorSetBuffers;

typedef struct DescriptorSetSamplers {
    int sampler_count;
    unsigned samplers[MAX_SAMPLER_BINDINGS];
    unsigned textures[MAX_SAMPLER_BINDINGS];
    unsigned targets[MAX_SAMPLER_BINDINGS];
    PyObject * sampler_refs[MAX_SAMPLER_BINDINGS];
    PyObject * texture_refs[MAX_SAMPLER_BINDINGS];
} DescriptorSetSamplers;

typedef struct DescriptorSet {
    PyObject_HEAD
    int uses;
    DescriptorSetBuffers uniform_buffers;
    DescriptorSetSamplers samplers;
} DescriptorSet;

typedef struct BlendState {
    int enable;
    int op_color;
    int op_alpha;
    int src_color;
    int dst_color;
    int src_alpha;
    int dst_alpha;
} BlendState;

typedef struct GlobalSettings {
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
    BlendState blend;
} GlobalSettings;

typedef struct Context {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    ModuleState * module_state;
    PyObject * loader;
    PyObject * descriptor_set_cache;
    PyObject * global_settings_cache;
    PyObject * sampler_cache;
    PyObject * vertex_array_cache;
    PyObject * framebuffer_cache;
    PyObject * program_cache;
    PyObject * shader_cache;
    PyObject * includes;
    GLObject * default_framebuffer;
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
    int gles;
    Limits limits;
    GLMethods gl;
} Context;

typedef struct Buffer {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    Context * ctx;
    int buffer;
    int size;
    int dynamic;
    int mapped;
} Buffer;

typedef struct Image {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    Context * ctx;
    PyObject * size;
    PyObject * format;
    PyObject * faces;
    PyObject * layers;
    const ImageFormat * fmt;
    ClearValue clear_value;
    int image;
    int width;
    int height;
    int samples;
    int array;
    int cubemap;
    int target;
    int renderbuffer;
    int layer_count;
    int level_count;
} Image;

typedef struct Pipeline {
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
} Pipeline;

typedef struct ImageFace {
    PyObject_HEAD
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
} ImageFace;

static void bind_global_settings(Context * self, GlobalSettings * settings) {
    const GLMethods * const gl = &self->gl;
    if (self->current_global_settings == settings) {
        return;
    }
    if (settings->cull_face) {
        gl->Enable(GL_CULL_FACE);
        gl->CullFace(settings->cull_face);
    } else {
        gl->Disable(GL_CULL_FACE);
    }
    if (settings->depth_enabled) {
        gl->Enable(GL_DEPTH_TEST);
        gl->DepthFunc(settings->depth_func);
        gl->DepthMask(settings->depth_write);
        self->current_depth_mask = settings->depth_write;
    } else {
        gl->Disable(GL_DEPTH_TEST);
    }
    if (settings->stencil_enabled) {
        gl->Enable(GL_STENCIL_TEST);
        gl->StencilMaskSeparate(GL_FRONT, settings->stencil_front.write_mask);
        gl->StencilMaskSeparate(GL_BACK, settings->stencil_back.write_mask);
        gl->StencilFuncSeparate(GL_FRONT, settings->stencil_front.compare_op, settings->stencil_front.reference, settings->stencil_front.compare_mask);
        gl->StencilFuncSeparate(GL_BACK, settings->stencil_back.compare_op, settings->stencil_back.reference, settings->stencil_back.compare_mask);
        gl->StencilOpSeparate(GL_FRONT, settings->stencil_front.fail_op, settings->stencil_front.pass_op, settings->stencil_front.depth_fail_op);
        gl->StencilOpSeparate(GL_BACK, settings->stencil_back.fail_op, settings->stencil_back.pass_op, settings->stencil_back.depth_fail_op);
        self->current_stencil_mask = settings->stencil_front.write_mask;
    } else {
        gl->Disable(GL_STENCIL_TEST);
    }
    if (settings->blend_enabled) {
        gl->Enable(GL_BLEND);
        gl->BlendEquationSeparate(settings->blend.op_color, settings->blend.op_alpha);
        gl->BlendFuncSeparate(settings->blend.src_color, settings->blend.dst_color, settings->blend.src_alpha, settings->blend.dst_alpha);
    } else {
        gl->Disable(GL_BLEND);
    }
    self->current_global_settings = settings;
}

static void bind_framebuffer(Context * self, int framebuffer) {
    const GLMethods * const gl = &self->gl;
    if (self->current_framebuffer != framebuffer) {
        self->current_framebuffer = framebuffer;
        gl->BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    }
}

static void bind_program(Context * self, int program) {
    const GLMethods * const gl = &self->gl;
    if (self->current_program != program) {
        self->current_program = program;
        gl->UseProgram(program);
    }
}

static void bind_vertex_array(Context * self, int vertex_array) {
    const GLMethods * const gl = &self->gl;
    if (self->current_vertex_array != vertex_array) {
        self->current_vertex_array = vertex_array;
        gl->BindVertexArray(vertex_array);
    }
}

static void bind_descriptor_set(Context * self, DescriptorSet * set) {
    const GLMethods * const gl = &self->gl;
    if (self->current_descriptor_set != set) {
        self->current_descriptor_set = set;
        if (set->uniform_buffers.buffer_count) {
            for (int i = 0; i < set->uniform_buffers.buffer_count; ++i) {
                gl->BindBufferRange(
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
                gl->ActiveTexture(GL_TEXTURE0 + i);
                gl->BindTexture(set->samplers.targets[i], set->samplers.textures[i]);
                gl->BindSampler(i, set->samplers.samplers[i]);
            }
        }
    }
}

static GLObject * build_framebuffer(Context * self, PyObject * attachments) {
    GLObject * cache = (GLObject *)PyDict_GetItem(self->framebuffer_cache, attachments);
    if (cache) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    PyObject * color_attachments = PyTuple_GetItem(attachments, 1);
    PyObject * depth_stencil_attachment = PyTuple_GetItem(attachments, 2);

    const GLMethods * const gl = &self->gl;

    int framebuffer = 0;
    gl->GenFramebuffers(1, (unsigned *)&framebuffer);
    bind_framebuffer(self, framebuffer);
    int color_attachment_count = (int)PyTuple_Size(color_attachments);
    for (int i = 0; i < color_attachment_count; ++i) {
        ImageFace * face = (ImageFace *)PyTuple_GetItem(color_attachments, i);
        if (face->image->renderbuffer) {
            gl->FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_RENDERBUFFER, face->image->image);
        } else if (face->image->cubemap) {
            gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face->layer, face->image->image, face->level);
        } else if (face->image->array) {
            gl->FramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, face->image->image, face->level, face->layer);
        } else {
            gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, face->image->image, face->level);
        }
    }

    if (depth_stencil_attachment != Py_None) {
        ImageFace * face = (ImageFace *)depth_stencil_attachment;
        int buffer = face->image->fmt->buffer;
        int attachment = buffer == GL_DEPTH ? GL_DEPTH_ATTACHMENT : buffer == GL_STENCIL ? GL_STENCIL_ATTACHMENT : GL_DEPTH_STENCIL_ATTACHMENT;
        if (face->image->renderbuffer) {
            gl->FramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, face->image->image);
        } else if (face->image->cubemap) {
            gl->FramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face->layer, face->image->image, face->level);
        } else if (face->image->array) {
            gl->FramebufferTextureLayer(GL_FRAMEBUFFER, attachment, face->image->image, face->level, face->layer);
        } else {
            gl->FramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, face->image->image, face->level);
        }
    }

    unsigned int draw_buffers[MAX_ATTACHMENTS];
    for (int i = 0; i < color_attachment_count; ++i) {
        draw_buffers[i] = GL_COLOR_ATTACHMENT0 + i;
    }

    gl->DrawBuffers(color_attachment_count, draw_buffers);
    gl->ReadBuffer(color_attachment_count ? GL_COLOR_ATTACHMENT0 : 0);

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = framebuffer;
    res->uses = 1;
    res->extra = NULL;

    PyDict_SetItem(self->framebuffer_cache, attachments, (PyObject *)res);
    return res;
}

static void bind_uniforms(Context * self, char * data) {
    const GLMethods * const gl = &self->gl;
    int offset = 0;
    while (1) {
        UniformBinding * header = (UniformBinding *)(data + offset);
        if (header->type == 0) {
            break;
        }
        switch (header->type) {
            case 0x1404: gl->Uniform1iv(header->location, header->count, header->int_values); break;
            case 0x8B53: gl->Uniform2iv(header->location, header->count, header->int_values); break;
            case 0x8B54: gl->Uniform3iv(header->location, header->count, header->int_values); break;
            case 0x8B55: gl->Uniform4iv(header->location, header->count, header->int_values); break;
            case 0x8B56: gl->Uniform1iv(header->location, header->count, header->int_values); break;
            case 0x8B57: gl->Uniform2iv(header->location, header->count, header->int_values); break;
            case 0x8B58: gl->Uniform3iv(header->location, header->count, header->int_values); break;
            case 0x8B59: gl->Uniform4iv(header->location, header->count, header->int_values); break;
            case 0x1405: gl->Uniform1uiv(header->location, header->count, header->uint_values); break;
            case 0x8DC6: gl->Uniform2uiv(header->location, header->count, header->uint_values); break;
            case 0x8DC7: gl->Uniform3uiv(header->location, header->count, header->uint_values); break;
            case 0x8DC8: gl->Uniform4uiv(header->location, header->count, header->uint_values); break;
            case 0x1406: gl->Uniform1fv(header->location, header->count, header->float_values); break;
            case 0x8B50: gl->Uniform2fv(header->location, header->count, header->float_values); break;
            case 0x8B51: gl->Uniform3fv(header->location, header->count, header->float_values); break;
            case 0x8B52: gl->Uniform4fv(header->location, header->count, header->float_values); break;
            case 0x8B5A: gl->UniformMatrix2fv(header->location, header->count, 0, header->float_values); break;
            case 0x8B65: gl->UniformMatrix2x3fv(header->location, header->count, 0, header->float_values); break;
            case 0x8B66: gl->UniformMatrix2x4fv(header->location, header->count, 0, header->float_values); break;
            case 0x8B67: gl->UniformMatrix3x2fv(header->location, header->count, 0, header->float_values); break;
            case 0x8B5B: gl->UniformMatrix3fv(header->location, header->count, 0, header->float_values); break;
            case 0x8B68: gl->UniformMatrix3x4fv(header->location, header->count, 0, header->float_values); break;
            case 0x8B69: gl->UniformMatrix4x2fv(header->location, header->count, 0, header->float_values); break;
            case 0x8B6A: gl->UniformMatrix4x3fv(header->location, header->count, 0, header->float_values); break;
            case 0x8B5C: gl->UniformMatrix4fv(header->location, header->count, 0, header->float_values); break;
        }
        offset += header->values * 4 + 16;
    }
}

static GLObject * build_vertex_array(Context * self, PyObject * bindings) {
    GLObject * cache = (GLObject *)PyDict_GetItem(self->vertex_array_cache, bindings);
    if (cache) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    const GLMethods * const gl = &self->gl;

    int length = (int)PyTuple_Size(bindings);
    PyObject ** seq = PySequence_Fast_ITEMS(bindings);
    PyObject * index_buffer = seq[0];

    int vertex_array = 0;
    gl->GenVertexArrays(1, (unsigned *)&vertex_array);
    bind_vertex_array(self, vertex_array);

    for (int i = 1; i < length; i += 6) {
        Buffer * buffer = (Buffer *)seq[i + 0];
        int location = PyLong_AsLong(seq[i + 1]);
        void * offset = PyLong_AsVoidPtr(seq[i + 2]);
        int stride = PyLong_AsLong(seq[i + 3]);
        int divisor = PyLong_AsLong(seq[i + 4]);
        const VertexFormat * format = get_vertex_format(PyUnicode_AsUTF8(seq[i + 5]));
        if (!format) {
            PyErr_Format(PyExc_ValueError, "invalid vertex format");
            return NULL;
        }
        gl->BindBuffer(GL_ARRAY_BUFFER, buffer->buffer);
        if (format->integer) {
            gl->VertexAttribIPointer(location, format->size, format->type, stride, offset);
        } else {
            gl->VertexAttribPointer(location, format->size, format->type, format->normalize, stride, offset);
        }
        gl->VertexAttribDivisor(location, divisor);
        gl->EnableVertexAttribArray(location);
    }

    if (index_buffer != Py_None) {
        Buffer * buffer = (Buffer *)index_buffer;
        gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer->buffer);
    }

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = vertex_array;
    res->uses = 1;
    res->extra = NULL;

    PyDict_SetItem(self->vertex_array_cache, bindings, (PyObject *)res);
    return res;
}

static GLObject * build_sampler(Context * self, PyObject * params) {
    GLObject * cache = (GLObject *)PyDict_GetItem(self->sampler_cache, params);
    if (cache) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    const GLMethods * const gl = &self->gl;

    PyObject ** seq = PySequence_Fast_ITEMS(params);

    int sampler = 0;
    gl->GenSamplers(1, (unsigned *)&sampler);
    gl->SamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, PyLong_AsLong(seq[0]));
    gl->SamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, PyLong_AsLong(seq[1]));
    gl->SamplerParameterf(sampler, GL_TEXTURE_MIN_LOD, (float)PyFloat_AsDouble(seq[2]));
    gl->SamplerParameterf(sampler, GL_TEXTURE_MAX_LOD, (float)PyFloat_AsDouble(seq[3]));

    float lod_bias = (float)PyFloat_AsDouble(seq[4]);
    if (lod_bias != 0.0f) {
        gl->SamplerParameterf(sampler, GL_TEXTURE_LOD_BIAS, lod_bias);
    }

    gl->SamplerParameteri(sampler, GL_TEXTURE_WRAP_S, PyLong_AsLong(seq[5]));
    gl->SamplerParameteri(sampler, GL_TEXTURE_WRAP_T, PyLong_AsLong(seq[6]));
    gl->SamplerParameteri(sampler, GL_TEXTURE_WRAP_R, PyLong_AsLong(seq[7]));
    gl->SamplerParameteri(sampler, GL_TEXTURE_COMPARE_MODE, PyLong_AsLong(seq[8]));
    gl->SamplerParameteri(sampler, GL_TEXTURE_COMPARE_FUNC, PyLong_AsLong(seq[9]));

    float max_anisotropy = (float)PyFloat_AsDouble(seq[10]);
    if (max_anisotropy != 1.0f) {
        gl->SamplerParameterf(sampler, GL_TEXTURE_MAX_ANISOTROPY, max_anisotropy);
    }

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = sampler;
    res->uses = 1;
    res->extra = NULL;

    PyDict_SetItem(self->sampler_cache, params, (PyObject *)res);
    return res;
}

static DescriptorSetBuffers build_descriptor_set_buffers(Context * self, PyObject * bindings) {
    DescriptorSetBuffers res;
    memset(&res, 0, sizeof(res));

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
    DescriptorSetSamplers res;
    memset(&res, 0, sizeof(res));

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
    DescriptorSet * cache = (DescriptorSet *)PyDict_GetItem(self->descriptor_set_cache, bindings);
    if (cache) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    DescriptorSet * res = PyObject_New(DescriptorSet, self->module_state->DescriptorSet_type);
    res->uniform_buffers = build_descriptor_set_buffers(self, PyTuple_GetItem(bindings, 0));
    res->samplers = build_descriptor_set_samplers(self, PyTuple_GetItem(bindings, 1));
    res->uses = 1;

    PyDict_SetItem(self->descriptor_set_cache, bindings, (PyObject *)res);
    return res;
}

static GlobalSettings * build_global_settings(Context * self, PyObject * settings) {
    GlobalSettings * cache = (GlobalSettings *)PyDict_GetItem(self->global_settings_cache, settings);
    if (cache) {
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
        res->stencil_front.fail_op = PyLong_AsLong(seq[it++]);
        res->stencil_front.pass_op = PyLong_AsLong(seq[it++]);
        res->stencil_front.depth_fail_op = PyLong_AsLong(seq[it++]);
        res->stencil_front.compare_op = PyLong_AsLong(seq[it++]);
        res->stencil_front.compare_mask = PyLong_AsLong(seq[it++]);
        res->stencil_front.write_mask = PyLong_AsLong(seq[it++]);
        res->stencil_front.reference = PyLong_AsLong(seq[it++]);
        res->stencil_back.fail_op = PyLong_AsLong(seq[it++]);
        res->stencil_back.pass_op = PyLong_AsLong(seq[it++]);
        res->stencil_back.depth_fail_op = PyLong_AsLong(seq[it++]);
        res->stencil_back.compare_op = PyLong_AsLong(seq[it++]);
        res->stencil_back.compare_mask = PyLong_AsLong(seq[it++]);
        res->stencil_back.write_mask = PyLong_AsLong(seq[it++]);
        res->stencil_back.reference = PyLong_AsLong(seq[it++]);
    }
    res->blend_enabled = PyLong_AsLong(seq[it++]);
    if (res->blend_enabled) {
        res->blend.enable = PyLong_AsLong(seq[it++]);
        res->blend.op_color = PyLong_AsLong(seq[it++]);
        res->blend.op_alpha = PyLong_AsLong(seq[it++]);
        res->blend.src_color = PyLong_AsLong(seq[it++]);
        res->blend.dst_color = PyLong_AsLong(seq[it++]);
        res->blend.src_alpha = PyLong_AsLong(seq[it++]);
        res->blend.dst_alpha = PyLong_AsLong(seq[it++]);
    }

    PyDict_SetItem(self->global_settings_cache, settings, (PyObject *)res);
    return res;
}

static GLObject * compile_shader(Context * self, PyObject * pair) {
    GLObject * cache = (GLObject *)PyDict_GetItem(self->shader_cache, pair);
    if (cache) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    const GLMethods * const gl = &self->gl;

    PyObject * code = PyTuple_GetItem(pair, 0);
    const char * src = PyBytes_AsString(code);
    int type = PyLong_AsLong(PyTuple_GetItem(pair, 1));
    int shader = gl->CreateShader(type);
    gl->ShaderSource(shader, 1, &src, NULL);
    gl->CompileShader(shader);

    int shader_compiled = 0;
    gl->GetShaderiv(shader, GL_COMPILE_STATUS, &shader_compiled);

    if (!shader_compiled) {
        int log_size = 0;
        gl->GetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_size);
        PyObject * log_text = PyBytes_FromStringAndSize(NULL, log_size);
        gl->GetShaderInfoLog(shader, log_size, &log_size, PyBytes_AsString(log_text));
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
    const GLMethods * const gl = &self->gl;

    bind_program(self, program);

    int num_attribs = 0;
    int num_uniforms = 0;
    int num_uniform_buffers = 0;
    gl->GetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &num_attribs);
    gl->GetProgramiv(program, GL_ACTIVE_UNIFORMS, &num_uniforms);
    gl->GetProgramiv(program, GL_ACTIVE_UNIFORM_BLOCKS, &num_uniform_buffers);

    PyObject * attributes = PyList_New(num_attribs);
    PyObject * uniforms = PyList_New(num_uniforms);
    PyObject * uniform_buffers = PyList_New(num_uniform_buffers);

    for (int i = 0; i < num_attribs; ++i) {
        int size = 0;
        int type = 0;
        int length = 0;
        char name[256] = {0};
        gl->GetActiveAttrib(program, i, 256, &length, &size, (unsigned *)&type, name);
        int location = gl->GetAttribLocation(program, name);
        PyList_SET_ITEM(attributes, i, Py_BuildValue("{sssisisi}", "name", name, "location", location, "gltype", type, "size", size));
    }

    for (int i = 0; i < num_uniforms; ++i) {
        int size = 0;
        int type = 0;
        int length = 0;
        char name[256] = {0};
        gl->GetActiveUniform(program, i, 256, &length, &size, (unsigned *)&type, name);
        int location = gl->GetUniformLocation(program, name);
        PyList_SET_ITEM(uniforms, i, Py_BuildValue("{sssisisi}", "name", name, "location", location, "gltype", type, "size", size));
    }

    for (int i = 0; i < num_uniform_buffers; ++i) {
        int size = 0;
        int length = 0;
        char name[256] = {0};
        gl->GetActiveUniformBlockiv(program, i, GL_UNIFORM_BLOCK_DATA_SIZE, &size);
        gl->GetActiveUniformBlockName(program, i, 256, &length, name);
        PyList_SET_ITEM(uniform_buffers, i, Py_BuildValue("{sssi}", "name", name, "size", size));
    }

    return Py_BuildValue("(NNN)", attributes, uniforms, uniform_buffers);
}

static GLObject * compile_program(Context * self, PyObject * includes, PyObject * vert, PyObject * frag, PyObject * layout) {
    const GLMethods * const gl = &self->gl;

    PyObject * tup = PyObject_CallMethod(self->module_state->helper, "program", "(OOOO)", vert, frag, layout, includes);
    if (!tup) {
        return NULL;
    }

    GLObject * cache = (GLObject *)PyDict_GetItem(self->program_cache, tup);
    if (cache) {
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

    int program = gl->CreateProgram();
    gl->AttachShader(program, vertex_shader_obj);
    gl->AttachShader(program, fragment_shader_obj);
    gl->LinkProgram(program);

    int linked = 0;
    gl->GetProgramiv(program, GL_LINK_STATUS, &linked);

    if (!linked) {
        int log_size = 0;
        gl->GetProgramiv(program, GL_INFO_LOG_LENGTH, &log_size);
        PyObject * log_text = PyBytes_FromStringAndSize(NULL, log_size);
        gl->GetProgramInfoLog(program, log_size, &log_size, PyBytes_AsString(log_text));
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

static ImageFace * build_image_face(Image * self, PyObject * key) {
    ImageFace * cache = (ImageFace *)PyDict_GetItem(self->faces, key);
    if (cache) {
        Py_INCREF(cache);
        return cache;
    }

    int layer = PyLong_AsLong(PyTuple_GetItem(key, 0));
    int level = PyLong_AsLong(PyTuple_GetItem(key, 1));

    int width = least_one(self->width >> level);
    int height = least_one(self->height >> level);

    ImageFace * res = PyObject_New(ImageFace, self->ctx->module_state->ImageFace_type);
    res->ctx = self->ctx;
    res->image = self;
    res->size = Py_BuildValue("(ii)", width, height);
    res->width = width;
    res->height = height;
    res->layer = layer;
    res->level = level;
    res->samples = self->samples;
    res->flags = self->fmt->flags;

    if (self->fmt->color) {
        PyObject * attachments = Py_BuildValue("((ii)(O)O)", width, height, res, Py_None);
        res->framebuffer = build_framebuffer(self->ctx, attachments);
        Py_DECREF(attachments);
    } else {
        PyObject * attachments = Py_BuildValue("((ii)()O)", width, height, res);
        res->framebuffer = build_framebuffer(self->ctx, attachments);
        Py_DECREF(attachments);
    }

    PyDict_SetItem(self->faces, key, (PyObject *)res);
    return res;
}

static void clear_bound_image(Image * self) {
    const GLMethods * const gl = &self->ctx->gl;
    const int depth_mask = self->ctx->current_depth_mask != 1 && (self->fmt->buffer == GL_DEPTH || self->fmt->buffer == GL_DEPTH_STENCIL);
    const int stencil_mask = self->ctx->current_stencil_mask != 0xff && (self->fmt->buffer == GL_STENCIL || self->fmt->buffer == GL_DEPTH_STENCIL);
    if (depth_mask) {
        gl->DepthMask(1);
        self->ctx->current_depth_mask = 1;
    }
    if (stencil_mask) {
        gl->StencilMaskSeparate(GL_FRONT, 0xff);
        self->ctx->current_stencil_mask = 0xff;
    }
    if (self->fmt->clear_type == 'f') {
        gl->ClearBufferfv(self->fmt->buffer, 0, self->clear_value.clear_floats);
    } else if (self->fmt->clear_type == 'i') {
        gl->ClearBufferiv(self->fmt->buffer, 0, self->clear_value.clear_ints);
    } else if (self->fmt->clear_type == 'u') {
        gl->ClearBufferuiv(self->fmt->buffer, 0, self->clear_value.clear_uints);
    } else if (self->fmt->clear_type == 'x') {
        gl->ClearBufferfi(self->fmt->buffer, 0, self->clear_value.clear_floats[0], self->clear_value.clear_ints[1]);
    }
}

static PyObject * blit_image_face(ImageFace * src, PyObject * dst, PyObject * src_viewport, PyObject * dst_viewport, int filter, PyObject * srgb) {
    if (Py_TYPE(dst) == src->image->ctx->module_state->Image_type) {
        Image * image = (Image *)dst;
        if (image->array || image->cubemap) {
            PyErr_Format(PyExc_TypeError, "cannot blit to whole cubemap or array images");
            return NULL;
        }
        dst = PyTuple_GetItem(image->layers, 0);
    }

    if (dst != Py_None && Py_TYPE(dst) != src->image->ctx->module_state->ImageFace_type) {
        PyErr_Format(PyExc_TypeError, "target must be an Image or ImageFace or None");
        return NULL;
    }

    ImageFace * target = dst != Py_None ? (ImageFace *)dst : NULL;

    if (target && target->image->samples > 1) {
        PyErr_Format(PyExc_TypeError, "cannot blit to multisampled images");
        return NULL;
    }

    Viewport tv = to_viewport(dst_viewport, 0, 0, target ? target->width : src->width, target ? target->height : src->height);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "the target viewport must be a tuple of 4 ints");
        return NULL;
    }

    Viewport sv = to_viewport(src_viewport, 0, 0, src->width, src->height);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "the source viewport must be a tuple of 4 ints");
        return NULL;
    }

    if (srgb == Py_None) {
        srgb = src->image->fmt->internal_format == GL_SRGB8_ALPHA8 ? Py_True : Py_False;
    }

    const int disable_srgb = !PyObject_IsTrue(srgb);

    if (tv.x < 0 || tv.y < 0 || tv.width <= 0 || tv.height <= 0 || (target && (tv.x + tv.width > target->width || tv.y + tv.height > target->height))) {
        PyErr_Format(PyExc_ValueError, "the target viewport is out of range");
        return NULL;
    }

    if (sv.x < 0 || sv.y < 0 || sv.width <= 0 || sv.height <= 0 || sv.x + sv.width > src->width || sv.y + sv.height > src->height) {
        PyErr_Format(PyExc_ValueError, "the source viewport is out of range");
        return NULL;
    }

    if (!src->image->fmt->color) {
        PyErr_Format(PyExc_TypeError, "cannot blit depth or stencil images");
        return NULL;
    }

    if (target && !target->image->fmt->color) {
        PyErr_Format(PyExc_TypeError, "cannot blit to depth or stencil images");
        return NULL;
    }

    if (target && target->image->samples > 1) {
        PyErr_Format(PyExc_TypeError, "cannot blit to multisampled images");
        return NULL;
    }

    const GLMethods * const gl = &src->ctx->gl;

    if (disable_srgb) {
        gl->Disable(GL_FRAMEBUFFER_SRGB);
    }

    int target_framebuffer = target ? target->framebuffer->obj : src->ctx->default_framebuffer->obj;
    gl->BindFramebuffer(GL_READ_FRAMEBUFFER, src->framebuffer->obj);
    gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, target_framebuffer);
    gl->BlitFramebuffer(
        sv.x, sv.y, sv.x + sv.width, sv.y + sv.height,
        tv.x, tv.y, tv.x + tv.width, tv.y + tv.height,
        GL_COLOR_BUFFER_BIT, filter ? GL_LINEAR : GL_NEAREST
    );
    src->image->ctx->current_framebuffer = -1;

    if (disable_srgb) {
        gl->Enable(GL_FRAMEBUFFER_SRGB);
    }

    Py_RETURN_NONE;
}

static PyObject * read_image_face(ImageFace * src, PyObject * size_arg, PyObject * offset_arg) {
    if (size_arg == Py_None && offset_arg != Py_None) {
        PyErr_Format(PyExc_ValueError, "the size is required when the offset is not None");
        return NULL;
    }

    IntPair size = to_int_pair(size_arg, src->width, src->height);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "the size must be a tuple of 2 ints");
        return NULL;
    }

    IntPair offset = to_int_pair(size_arg, 0, 0);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "the offset must be a tuple of 2 ints");
        return NULL;
    }

    if (size.x <= 0 || size.y <= 0 || size.x > src->width || size.y > src->height) {
        PyErr_Format(PyExc_ValueError, "invalid size");
        return NULL;
    }

    if (offset.x < 0 || offset.y < 0 || size.x + offset.x > src->width || size.y + offset.y > src->height) {
        PyErr_Format(PyExc_ValueError, "invalid offset");
        return NULL;
    }

    if (src->image->samples > 1) {
        PyObject * temp = PyObject_CallMethod((PyObject *)src->image->ctx, "image", "((ii)O)", size.x, size.y, src->image->format);
        if (!temp) {
            return NULL;
        }

        PyObject * blit = PyObject_CallMethod((PyObject *)src, "blit", "(O(iiii)(iiii))", temp, 0, 0, size.x, size.y, offset.x, offset.y, size.x, size.y);
        if (!blit) {
            return NULL;
        }
        Py_DECREF(blit);

        PyObject * res = PyObject_CallMethod(temp, "read", NULL);
        if (!res) {
            return NULL;
        }

        PyObject * release = PyObject_CallMethod((PyObject *)src->image->ctx, "release", "(N)", temp);
        if (!release) {
            return NULL;
        }
        Py_DECREF(release);
        return res;
    }

    const GLMethods * const gl = &src->image->ctx->gl;

    PyObject * res = PyBytes_FromStringAndSize(NULL, (long long)size.x * size.y * src->image->fmt->pixel_size);
    bind_framebuffer(src->ctx, src->framebuffer->obj);
    gl->ReadPixels(offset.x, offset.y, size.x, size.y, src->image->fmt->format, src->image->fmt->type, PyBytes_AS_STRING(res));
    return res;
}

static Context * meth_context(PyObject * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"loader", NULL};

    PyObject * loader = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", keywords, &loader)) {
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

    GLObject * default_framebuffer = PyObject_New(GLObject, module_state->GLObject_type);
    default_framebuffer->obj = 0;
    default_framebuffer->uses = 1;
    default_framebuffer->extra = NULL;

    Context * res = PyObject_New(Context, module_state->Context_type);
    res->gc_prev = (GCHeader *)res;
    res->gc_next = (GCHeader *)res;
    res->module_state = module_state;
    res->loader = loader;
    res->descriptor_set_cache = PyDict_New();
    res->global_settings_cache = PyDict_New();
    res->sampler_cache = PyDict_New();
    res->vertex_array_cache = PyDict_New();
    res->framebuffer_cache = Py_BuildValue("{OO}", Py_None, default_framebuffer);
    res->program_cache = PyDict_New();
    res->shader_cache = PyDict_New();
    res->includes = PyDict_New();
    res->default_framebuffer = default_framebuffer;
    res->before_frame_callback = (PyObject *)new_ref(Py_None);
    res->after_frame_callback = (PyObject *)new_ref(Py_None);
    res->limits_dict = NULL;
    res->info_dict = NULL;
    res->current_descriptor_set = NULL;
    res->current_global_settings = NULL;
    res->is_mask_default = 0;
    res->is_stencil_default = 0;
    res->is_blend_default = 0;
    res->current_viewport.x = 0;
    res->current_viewport.y = 0;
    res->current_viewport.width = 0;
    res->current_viewport.height = 0;
    res->current_framebuffer = -1;
    res->current_program = -1;
    res->current_vertex_array = -1;
    res->current_depth_mask = 0;
    res->current_stencil_mask = 0;
    res->frame_time_query = 0;
    res->frame_time_query_running = 0;
    res->frame_time = 0;
    res->default_texture_unit = 0;
    res->mapped_buffers = 0;
    res->gles = 0;

    load_gl(&res->gl, loader);

    const GLMethods * const gl = &res->gl;

    if (PyErr_Occurred()) {
        return NULL;
    }

    memset(&res->limits, 0, sizeof(res->limits));
    gl->GetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &res->limits.max_uniform_buffer_bindings);
    gl->GetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &res->limits.max_uniform_block_size);
    gl->GetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS, &res->limits.max_combined_uniform_blocks);
    gl->GetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &res->limits.max_combined_texture_image_units);
    gl->GetIntegerv(GL_MAX_VERTEX_ATTRIBS, &res->limits.max_vertex_attribs);
    gl->GetIntegerv(GL_MAX_DRAW_BUFFERS, &res->limits.max_draw_buffers);
    gl->GetIntegerv(GL_MAX_SAMPLES, &res->limits.max_samples);

    if (res->limits.max_uniform_buffer_bindings > MAX_UNIFORM_BUFFER_BINDINGS) {
        res->limits.max_uniform_buffer_bindings = MAX_UNIFORM_BUFFER_BINDINGS;
    }

    if (res->limits.max_combined_texture_image_units > MAX_SAMPLER_BINDINGS) {
        res->limits.max_combined_texture_image_units = MAX_SAMPLER_BINDINGS;
    }

    res->limits_dict = Py_BuildValue(
        "{sisisisisisisi}",
        "max_uniform_buffer_bindings", res->limits.max_uniform_buffer_bindings,
        "max_uniform_block_size", res->limits.max_uniform_block_size,
        "max_combined_uniform_blocks", res->limits.max_combined_uniform_blocks,
        "max_combined_texture_image_units", res->limits.max_combined_texture_image_units,
        "max_vertex_attribs", res->limits.max_vertex_attribs,
        "max_draw_buffers", res->limits.max_draw_buffers,
        "max_samples", res->limits.max_samples
    );

    res->info_dict = Py_BuildValue(
        "{szszszsz}",
        "vendor", gl->GetString(GL_VENDOR),
        "renderer", gl->GetString(GL_RENDERER),
        "version", gl->GetString(GL_VERSION),
        "glsl", gl->GetString(GL_SHADING_LANGUAGE_VERSION)
    );

    PyObject * detect_gles = PyObject_CallMethod(module_state->helper, "detect_gles", "(O)", res->info_dict);
    if (!detect_gles) {
        return NULL;
    }

    res->gles = PyObject_IsTrue(detect_gles);
    Py_DECREF(detect_gles);

    int max_texture_image_units = 0;
    gl->GetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &max_texture_image_units);
    res->default_texture_unit = GL_TEXTURE0 + max_texture_image_units - 1;

    gl->Enable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    if (!res->gles) {
        gl->Enable(GL_PROGRAM_POINT_SIZE);
        gl->Enable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
        gl->Enable(GL_FRAMEBUFFER_SRGB);
    }

    return res;
}

static Buffer * Context_meth_buffer(Context * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"data", "size", "dynamic", "external", NULL};

    PyObject * data = Py_None;
    PyObject * size_arg = Py_None;
    int dynamic = 1;
    int external = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O$Opi", keywords, &data, &size_arg, &dynamic, &external)) {
        return NULL;
    }

    const GLMethods * const gl = &self->gl;

    if (size_arg != Py_None && !PyLong_CheckExact(size_arg)) {
        PyErr_Format(PyExc_TypeError, "the size must be an int");
        return NULL;
    }

    if (data == Py_None && size_arg == Py_None) {
        PyErr_Format(PyExc_ValueError, "data or size is required");
        return NULL;
    }

    if (data != Py_None && size_arg != Py_None) {
        PyErr_Format(PyExc_ValueError, "data and size are exclusive");
        return NULL;
    }

    int size = 0;
    if (size_arg != Py_None) {
        size = PyLong_AsLong(size_arg);
        if (size <= 0) {
            PyErr_Format(PyExc_ValueError, "invalid size");
            return NULL;
        }
    }

    if (data != Py_None) {
        data = PyMemoryView_FromObject(data);
        if (PyErr_Occurred()) {
            return NULL;
        }
        size = (int)PyMemoryView_GET_BUFFER(data)->len;
        if (size == 0) {
            PyErr_Format(PyExc_ValueError, "invalid size");
            return NULL;
        }
    }

    int buffer = 0;
    if (external) {
        buffer = external;
    } else {
        gl->GenBuffers(1, (unsigned *)&buffer);
        gl->BindBuffer(GL_ARRAY_BUFFER, buffer);
        gl->BufferData(GL_ARRAY_BUFFER, size, NULL, dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
    }

    Buffer * res = PyObject_New(Buffer, self->module_state->Buffer_type);
    res->gc_prev = self->gc_prev;
    res->gc_next = (GCHeader *)self;
    res->gc_prev->gc_next = (GCHeader *)res;
    res->gc_next->gc_prev = (GCHeader *)res;
    Py_INCREF(res);

    res->ctx = self;
    res->buffer = buffer;
    res->size = size;
    res->dynamic = dynamic;
    res->mapped = 0;

    if (data != Py_None) {
        Py_XDECREF(PyObject_CallMethod((PyObject *)res, "write", "N", data));
        if (PyErr_Occurred()) {
            return NULL;
        }
    }

    return res;
}

static Image * Context_meth_image(Context * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"size", "format", "data", "samples", "array", "levels", "texture", "cubemap", "external", NULL};

    int width;
    int height;
    PyObject * format;
    PyObject * data = Py_None;
    int samples = 1;
    int array = 0;
    PyObject * texture = Py_None;
    int cubemap = 0;
    int levels = 1;
    int external = 0;

    int args_ok = PyArg_ParseTupleAndKeywords(
        args,
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

    const GLMethods * const gl = &self->gl;

    int max_levels = count_mipmaps(width, height);
    if (levels <= 0) {
        levels = max_levels;
    }

    if (texture != Py_True && texture != Py_False && texture != Py_None) {
        PyErr_Format(PyExc_TypeError, "invalid texture parameter");
        return NULL;
    }
    if (samples > 1 && texture == Py_True) {
        PyErr_Format(PyExc_TypeError, "for multisampled images texture must be False");
        return NULL;
    }
    if (samples < 1 || (samples & (samples - 1)) || samples > 16) {
        PyErr_Format(PyExc_ValueError, "samples must be 1, 2, 4, 8 or 16");
        return NULL;
    }
    if (array < 0) {
        PyErr_Format(PyExc_ValueError, "array must not be negative");
        return NULL;
    }
    if (levels > max_levels) {
        PyErr_Format(PyExc_ValueError, "too many levels");
        return NULL;
    }
    if (cubemap && array) {
        PyErr_Format(PyExc_TypeError, "cubemap arrays are not supported");
        return NULL;
    }
    if (samples > 1 && (array || cubemap)) {
        PyErr_Format(PyExc_TypeError, "multisampled array or cubemap images are not supported");
        return NULL;
    }
    if (texture == Py_False && (array || cubemap)) {
        PyErr_Format(PyExc_TypeError, "for array or cubemap images texture must be True");
        return NULL;
    }
    if (data != Py_None && samples > 1) {
        PyErr_Format(PyExc_ValueError, "cannot write to multisampled images");
        return NULL;
    }
    if (data != Py_None && texture == Py_False) {
        PyErr_Format(PyExc_ValueError, "cannot write to renderbuffers");
        return NULL;
    }

    int renderbuffer = samples > 1 || texture == Py_False;
    int target = cubemap ? GL_TEXTURE_CUBE_MAP : array ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;

    if (samples > self->limits.max_samples) {
        samples = self->limits.max_samples;
    }

    const ImageFormat * fmt = get_image_format(PyUnicode_AsUTF8(format));
    if (!fmt) {
        PyErr_Format(PyExc_ValueError, "invalid image format");
        return NULL;
    }

    int image = 0;
    if (external) {
        image = external;
    } else if (renderbuffer) {
        gl->GenRenderbuffers(1, (unsigned *)&image);
        gl->BindRenderbuffer(GL_RENDERBUFFER, image);
        gl->RenderbufferStorageMultisample(GL_RENDERBUFFER, samples > 1 ? samples : 0, fmt->internal_format, width, height);
    } else {
        gl->GenTextures(1, (unsigned *)&image);
        gl->ActiveTexture(self->default_texture_unit);
        gl->BindTexture(target, image);
        gl->TexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl->TexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        for (int level = 0; level < levels; ++level) {
            int w = least_one(width >> level);
            int h = least_one(height >> level);
            if (cubemap) {
                for (int i = 0; i < 6; ++i) {
                    int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i;
                    gl->TexImage2D(face, level, fmt->internal_format, w, h, 0, fmt->format, fmt->type, NULL);
                }
            } else if (array) {
                gl->TexImage3D(target, level, fmt->internal_format, w, h, array, 0, fmt->format, fmt->type, NULL);
            } else {
                gl->TexImage2D(target, level, fmt->internal_format, w, h, 0, fmt->format, fmt->type, NULL);
            }
        }
    }

    Image * res = PyObject_New(Image, self->module_state->Image_type);
    res->gc_prev = self->gc_prev;
    res->gc_next = (GCHeader *)self;
    res->gc_prev->gc_next = (GCHeader *)res;
    res->gc_next->gc_prev = (GCHeader *)res;
    Py_INCREF(res);

    res->ctx = self;
    res->size = Py_BuildValue("(ii)", width, height);
    res->format = (PyObject *)new_ref(format);
    res->faces = PyDict_New();
    res->fmt = fmt;
    res->clear_value.clear_ints[0] = 0;
    res->clear_value.clear_ints[1] = 0;
    res->clear_value.clear_ints[2] = 0;
    res->clear_value.clear_ints[3] = 0;
    res->image = image;
    res->width = width;
    res->height = height;
    res->samples = samples;
    res->array = array;
    res->cubemap = cubemap;
    res->target = target;
    res->renderbuffer = renderbuffer;
    res->layer_count = (array ? array : 1) * (cubemap ? 6 : 1);
    res->level_count = levels;

    if (fmt->buffer == GL_DEPTH || fmt->buffer == GL_DEPTH_STENCIL) {
        res->clear_value.clear_floats[0] = 1.0f;
    }

    res->layers = PyTuple_New(res->layer_count);
    for (int i = 0; i < res->layer_count; ++i) {
        PyObject * key = Py_BuildValue("(ii)", i, 0);
        PyTuple_SetItem(res->layers, i, (PyObject *)build_image_face(res, key));
        Py_DECREF(key);
    }

    if (data != Py_None) {
        Py_XDECREF(PyObject_CallMethod((PyObject *)res, "write", "O", data));
        if (PyErr_Occurred()) {
            return NULL;
        }
    }

    return res;
}

static Pipeline * Context_meth_pipeline(Context * self, PyObject * args, PyObject * kwargs) {
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
    PyObject * framebuffer_attachments = NULL;
    PyObject * vertex_buffers = self->module_state->empty_tuple;
    PyObject * index_buffer = Py_None;
    int short_index = 0;
    PyObject * cull_face = self->module_state->str_none;
    int topology = GL_TRIANGLES;
    int vertex_count = 0;
    int instance_count = 1;
    int first_vertex = 0;
    PyObject * viewport = Py_None;
    PyObject * includes = Py_None;

    int args_ok = PyArg_ParseTupleAndKeywords(
        args,
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
        &framebuffer_attachments,
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

    if (!vertex_shader) {
        PyErr_Format(PyExc_TypeError, "no vertex_shader was specified");
        return NULL;
    }

    if (!fragment_shader) {
        PyErr_Format(PyExc_TypeError, "no fragment_shader was specified");
        return NULL;
    }

    if (!framebuffer_attachments) {
        PyErr_Format(PyExc_TypeError, "no framebuffer was specified");
        return NULL;
    }

    if (framebuffer_attachments == Py_None && viewport == Py_None) {
        PyErr_Format(PyExc_TypeError, "no viewport was specified");
        return NULL;
    }

    Viewport viewport_value = to_viewport(viewport, 0, 0, 0, 0);
    if (PyErr_Occurred()) {
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

        while (1) {
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

    PyObject * attachments = PyObject_CallMethod(self->module_state->helper, "framebuffer_attachments", "(O)", framebuffer_attachments);
    if (!attachments) {
        return NULL;
    }

    PyObject * validate = PyObject_CallMethod(
        self->module_state->helper,
        "validate",
        "OOOOO",
        program->extra,
        layout,
        resources,
        vertex_buffers,
        self->limits_dict
    );

    if (!validate) {
        return NULL;
    }

    const GLMethods * const gl = &self->gl;
    int layout_count = layout != Py_None ? (int)PyList_Size(layout) : 0; // TODO: check
    for (int i = 0; i < layout_count; ++i) {
        PyObject * obj = PyList_GetItem(layout, i);
        PyObject * name = PyDict_GetItemString(obj, "name");
        int binding = PyLong_AsLong(PyDict_GetItemString(obj, "binding"));
        int location = gl->GetUniformLocation(program->obj, PyUnicode_AsUTF8(name));
        if (location >= 0) {
            gl->Uniform1i(location, binding);
        } else {
            int index = gl->GetUniformBlockIndex(program->obj, PyUnicode_AsUTF8(name));
            gl->UniformBlockBinding(program->obj, index, binding);
        }
    }

    if (attachments != Py_None && viewport == Py_None) {
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
    Py_INCREF(res);

    res->ctx = self;
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
    return res;
}

static PyObject * Context_meth_new_frame(Context * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"reset", "clear", "frame_time", NULL};

    int reset = 1;
    int clear = 1;
    int frame_time = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ppp", keywords, &reset, &clear, &frame_time)) {
        return NULL;
    }

    const GLMethods * const gl = &self->gl;

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
        self->is_stencil_default = 0;
        self->is_mask_default = 0;
        self->is_blend_default = 0;
        self->current_viewport.x = -1;
        self->current_viewport.y = -1;
        self->current_viewport.width = -1;
        self->current_viewport.height = -1;
        self->current_framebuffer = -1;
        self->current_program = -1;
        self->current_vertex_array = -1;
        self->current_depth_mask = 0;
        self->current_stencil_mask = 0;
    }

    if (clear) {
        bind_framebuffer(self, 0);
        gl->Clear(GL_COLOR_BUFFER_BIT);
    }

    if (frame_time) {
        if (!self->frame_time_query) {
            gl->GenQueries(1, (unsigned *)&self->frame_time_query);
        }
        gl->BeginQuery(GL_TIME_ELAPSED, self->frame_time_query);
        self->frame_time_query_running = 1;
        self->frame_time = 0;
    }

    gl->Enable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    if (!self->gles) {
        gl->Enable(GL_PROGRAM_POINT_SIZE);
        gl->Enable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
        gl->Enable(GL_FRAMEBUFFER_SRGB);
    }
    Py_RETURN_NONE;
}

static PyObject * Context_meth_end_frame(Context * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"clean", "flush", "sync", NULL};

    int clean = 1;
    int flush = 1;
    int sync = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ppp", keywords, &clean, &flush, &sync)) {
        return NULL;
    }

    const GLMethods * const gl = &self->gl;

    if (clean) {
        bind_framebuffer(self, 0);
        bind_program(self, 0);
        bind_vertex_array(self, 0);

        self->current_descriptor_set = NULL;
        self->current_global_settings = NULL;

        gl->ActiveTexture(GL_TEXTURE0);

        gl->Disable(GL_CULL_FACE);
        gl->Disable(GL_DEPTH_TEST);
        gl->Disable(GL_STENCIL_TEST);
        gl->Disable(GL_BLEND);

        gl->Disable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        if (!self->gles) {
            gl->Disable(GL_PROGRAM_POINT_SIZE);
            gl->Disable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
            gl->Disable(GL_FRAMEBUFFER_SRGB);
        }
    }

    if (self->frame_time_query_running) {
        gl->EndQuery(GL_TIME_ELAPSED);
        gl->GetQueryObjectuiv(self->frame_time_query, GL_QUERY_RESULT, (unsigned *)&self->frame_time);
        self->frame_time_query_running = 0;
    } else {
        self->frame_time = 0;
    }

    if (flush) {
        gl->Flush();
    }

    if (sync) {
        void * sync = gl->FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl->ClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
        gl->DeleteSync(sync);
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
    const GLMethods * const gl = &self->gl;
    set->uses -= 1;
    if (!set->uses) {
        for (int i = 0; i < set->samplers.sampler_count; ++i) {
            GLObject * sampler = (GLObject *)set->samplers.sampler_refs[i];
            if (sampler) {
                sampler->uses -= 1;
                if (!sampler->uses) {
                    remove_dict_value(self->sampler_cache, (PyObject *)sampler);
                    gl->DeleteSamplers(1, (unsigned int *)&sampler->obj);
                }
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
    const GLMethods * const gl = &self->gl;
    framebuffer->uses -= 1;
    if (!framebuffer->uses) {
        remove_dict_value(self->framebuffer_cache, (PyObject *)framebuffer);
        if (self->current_framebuffer == framebuffer->obj) {
            self->current_framebuffer = 0;
        }
        if (framebuffer->obj) {
            gl->DeleteFramebuffers(1, (unsigned int *)&framebuffer->obj);
        }
    }
}

static void release_program(Context * self, GLObject * program) {
    const GLMethods * const gl = &self->gl;
    program->uses -= 1;
    if (!program->uses) {
        remove_dict_value(self->program_cache, (PyObject *)program);
        if (self->current_program == program->obj) {
            self->current_program = 0;
        }
        gl->DeleteProgram(program->obj);
    }
}

static void release_vertex_array(Context * self, GLObject * vertex_array) {
    const GLMethods * const gl = &self->gl;
    vertex_array->uses -= 1;
    if (!vertex_array->uses) {
        remove_dict_value(self->vertex_array_cache, (PyObject *)vertex_array);
        if (self->current_vertex_array == vertex_array->obj) {
            self->current_vertex_array = 0;
        }
        gl->DeleteVertexArrays(1, (unsigned int *)&vertex_array->obj);
    }
}

static PyObject * Context_meth_release(Context * self, PyObject * arg) {
    const GLMethods * const gl = &self->gl;
    if (Py_TYPE(arg) == self->module_state->Buffer_type) {
        Buffer * buffer = (Buffer *)arg;
        buffer->gc_prev->gc_next = buffer->gc_next;
        buffer->gc_next->gc_prev = buffer->gc_prev;
        gl->DeleteBuffers(1, (unsigned int *)&buffer->buffer);
        Py_DECREF(buffer);
    } else if (Py_TYPE(arg) == self->module_state->Image_type) {
        Image * image = (Image *)arg;
        image->gc_prev->gc_next = image->gc_next;
        image->gc_next->gc_prev = image->gc_prev;
        if (image->faces) {
            PyObject * key = NULL;
            PyObject * value = NULL;
            Py_ssize_t pos = 0;
            while (PyDict_Next(image->faces, &pos, &key, &value)) {
                ImageFace * face = (ImageFace *)value;
                release_framebuffer(self, face->framebuffer);
            }
            PyDict_Clear(image->faces);
        }
        if (image->renderbuffer) {
            gl->DeleteRenderbuffers(1, (unsigned int *)&image->image);
        } else {
            gl->DeleteTextures(1, (unsigned int *)&image->image);
        }
        Py_DECREF(image);
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
        }
        Py_DECREF(pipeline);
    } else if (PyUnicode_CheckExact(arg) && !PyUnicode_CompareWithASCIIString(arg, "shader_cache")) {
        PyObject * key = NULL;
        PyObject * value = NULL;
        Py_ssize_t pos = 0;
        while (PyDict_Next(self->shader_cache, &pos, &key, &value)) {
            GLObject * shader = (GLObject *)value;
            gl->DeleteShader(shader->obj);
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

static PyObject * Context_meth_gc(Context * self, PyObject * arg) {
    PyObject * res = PyList_New(0);
    GCHeader * it = self->gc_next;
    while (it != (GCHeader *)self) {
        GCHeader * next = it->gc_next;
        PyList_Append(res, (PyObject *)it);
        it = next;
    }
    return res;
}

static PyObject * Context_get_screen(Context * self, void * closure) {
    return PyLong_FromLong(self->default_framebuffer->obj);
}

static int Context_set_screen(Context * self, PyObject * value, void * closure) {
    if (!PyLong_CheckExact(value)) {
        PyErr_Format(PyExc_TypeError, "the clear value must be an int");
        return -1;
    }

    self->default_framebuffer->obj = PyLong_AsLong(value);
    return 0;
}

static PyObject * Buffer_meth_write(Buffer * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"data", "offset", NULL};

    PyObject * data;
    int offset = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", keywords, &data, &offset)) {
        return NULL;
    }

    if (self->mapped) {
        PyErr_Format(PyExc_RuntimeError, "already mapped");
        return NULL;
    }

    if (offset < 0 || offset > self->size) {
        PyErr_Format(PyExc_ValueError, "invalid offset");
        return NULL;
    }

    PyObject * mem = contiguous(data);
    if (!mem) {
        return NULL;
    }

    Py_buffer * view = PyMemoryView_GET_BUFFER(mem);

    if ((int)view->len + offset > self->size) {
        PyErr_Format(PyExc_ValueError, "invalid size");
        return NULL;
    }

    const GLMethods * const gl = &self->ctx->gl;

    if (view->len) {
        gl->BindBuffer(GL_ARRAY_BUFFER, self->buffer);
        gl->BufferSubData(GL_ARRAY_BUFFER, offset, (int)view->len, view->buf);
    }

    Py_DECREF(mem);
    Py_RETURN_NONE;
}

static PyObject * Buffer_meth_map(Buffer * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"size", "offset", "discard", NULL};

    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;
    int discard = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOp", keywords, &size_arg, &offset_arg, &discard)) {
        return NULL;
    }

    int size = self->size;
    int offset = 0;

    if (size_arg != Py_None && !PyLong_CheckExact(size_arg)) {
        PyErr_Format(PyExc_TypeError, "the size must be an int or None");
        return NULL;
    }

    if (offset_arg != Py_None && !PyLong_CheckExact(offset_arg)) {
        PyErr_Format(PyExc_TypeError, "the offset must be an int or None");
        return NULL;
    }

    if (size_arg != Py_None) {
        size = PyLong_AsLong(size_arg);
    }

    if (offset_arg != Py_None) {
        offset = PyLong_AsLong(offset_arg);
    }

    if (self->mapped) {
        PyErr_Format(PyExc_RuntimeError, "already mapped");
        return NULL;
    }

    if (size_arg == Py_None && offset_arg != Py_None) {
        PyErr_Format(PyExc_ValueError, "the size is required when the offset is not None");
        return NULL;
    }

    if (size <= 0 || size > self->size) {
        PyErr_Format(PyExc_ValueError, "invalid size");
        return NULL;
    }

    if (offset < 0 || offset + size > self->size) {
        PyErr_Format(PyExc_ValueError, "invalid offset");
        return NULL;
    }

    const GLMethods * const gl = &self->ctx->gl;

    self->mapped = 1;
    self->ctx->mapped_buffers += 1;
    const int access = discard ? GL_MAP_READ_BIT | GL_MAP_WRITE_BIT : GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_WRITE_BIT;
    gl->BindBuffer(GL_ARRAY_BUFFER, self->buffer);
    void * ptr = gl->MapBufferRange(GL_ARRAY_BUFFER, offset, size, access);
    return PyMemoryView_FromMemory((char *)ptr, size, PyBUF_WRITE);
}

static PyObject * Buffer_meth_unmap(Buffer * self, PyObject * args) {
    const GLMethods * const gl = &self->ctx->gl;
    if (self->mapped) {
        self->mapped = 0;
        self->ctx->mapped_buffers -= 1;
        gl->BindBuffer(GL_ARRAY_BUFFER, self->buffer);
        gl->UnmapBuffer(GL_ARRAY_BUFFER);
    }
    Py_RETURN_NONE;
}

static PyObject * Image_meth_clear(Image * self, PyObject * args) {
    const int count = (int)PyTuple_Size(self->layers);
    for (int i = 0; i < count; ++i) {
        ImageFace * face = (ImageFace *)PyTuple_GetItem(self->layers, i);
        bind_framebuffer(self->ctx, face->framebuffer->obj);
        clear_bound_image(self);
    }
    Py_RETURN_NONE;
}

static PyObject * Image_meth_write(Image * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"data", "size", "offset", "layer", "level", NULL};

    PyObject * data;
    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;
    PyObject * layer_arg = Py_None;
    int level = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOOi", keywords, &data, &size_arg, &offset_arg, &layer_arg, &level)) {
        return NULL;
    }

    if (layer_arg != Py_None && !PyLong_CheckExact(layer_arg)) {
        PyErr_Format(PyExc_TypeError, "the layer must be an int or None");
        return NULL;
    }

    int layer = 0;
    if (layer_arg != Py_None) {
        layer = PyLong_AsLong(layer_arg);
    }

    IntPair size = to_int_pair(size_arg, least_one(self->width >> level), least_one(self->height >> level));
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "the size must be a tuple of 2 ints");
        return NULL;
    }

    IntPair offset = to_int_pair(offset_arg, 0, 0);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "the offset must be a tuple of 2 ints");
        return NULL;
    }

    if (size_arg == Py_None && offset_arg != Py_None) {
        PyErr_Format(PyExc_ValueError, "the size is required when the offset is not None");
        return NULL;
    }

    if (size.x <= 0 || size.y <= 0 || size.x > self->width || size.y > self->height) {
        PyErr_Format(PyExc_ValueError, "invalid size");
        return NULL;
    }

    if (offset.x < 0 || offset.y < 0 || size.x + offset.x > self->width || size.y + offset.y > self->height) {
        PyErr_Format(PyExc_ValueError, "invalid offset");
        return NULL;
    }

    if (layer < 0 || layer >= self->layer_count) {
        PyErr_Format(PyExc_ValueError, "invalid layer");
        return NULL;
    }

    if (level < 0 || level > self->level_count) {
        PyErr_Format(PyExc_ValueError, "invalid level");
        return NULL;
    }

    if (!self->cubemap && !self->array && layer_arg != Py_None) {
        PyErr_Format(PyExc_TypeError, "the image is not layered");
        return NULL;
    }

    if (!self->fmt->color) {
        PyErr_Format(PyExc_TypeError, "cannot write to depth or stencil images");
        return NULL;
    }

    if (self->samples != 1) {
        PyErr_Format(PyExc_TypeError, "cannot write to multisampled images");
        return NULL;
    }

    int padded_row = (size.x * self->fmt->pixel_size + 3) & ~3;
    int expected_size = padded_row * size.y;

    if (layer_arg == Py_None) {
        expected_size *= self->layer_count;
    }

    PyObject * mem = contiguous(data);
    if (!mem) {
        return NULL;
    }

    Py_buffer * view = PyMemoryView_GET_BUFFER(mem);

    if ((int)view->len != expected_size) {
        PyErr_Format(PyExc_ValueError, "invalid data size, expected %d, got %d", expected_size, (int)view->len);
        return NULL;
    }

    const GLMethods * const gl = &self->ctx->gl;

    gl->ActiveTexture(self->ctx->default_texture_unit);
    gl->BindTexture(self->target, self->image);
    if (self->cubemap) {
        int padded_row = (size.x * self->fmt->pixel_size + 3) & ~3;
        int stride = padded_row * size.y;
        if (layer_arg != Py_None) {
            int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer;
            gl->TexSubImage2D(face, level, offset.x, offset.y, size.x, size.y, self->fmt->format, self->fmt->type, view->buf);
        } else {
            for (int i = 0; i < 6; ++i) {
                int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i;
                gl->TexSubImage2D(face, level, offset.x, offset.y, size.x, size.y, self->fmt->format, self->fmt->type, (char *)view->buf + stride * i);
            }
        }
    } else if (self->array) {
        if (layer_arg != Py_None) {
            gl->TexSubImage3D(self->target, level, offset.x, offset.y, layer, size.x, size.y, 1, self->fmt->format, self->fmt->type, view->buf);
        } else {
            gl->TexSubImage3D(self->target, level, offset.x, offset.y, 0, size.x, size.y, self->array, self->fmt->format, self->fmt->type, view->buf);
        }
    } else {
        gl->TexSubImage2D(self->target, level, offset.x, offset.y, size.x, size.y, self->fmt->format, self->fmt->type, view->buf);
    }

    Py_DECREF(mem);
    Py_RETURN_NONE;
}

static PyObject * Image_meth_mipmaps(Image * self, PyObject * args) {
    const GLMethods * const gl = &self->ctx->gl;
    gl->ActiveTexture(self->ctx->default_texture_unit);
    gl->BindTexture(self->target, self->image);
    gl->GenerateMipmap(self->target);
    Py_RETURN_NONE;
}

static PyObject * Image_meth_read(Image * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"size", "offset", NULL};

    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", keywords, &size_arg, &offset_arg)) {
        return NULL;
    }

    if (self->array || self->cubemap) {
        PyObject * chunks = PyTuple_New(self->layer_count);
        for (int i = 0; i < self->layer_count; ++i) {
            ImageFace * src = (ImageFace *)PyTuple_GetItem(self->layers, i);
            PyObject * chunk = read_image_face(src, size_arg, offset_arg);
            if (!chunk) {
                return NULL;
            }
            PyTuple_SetItem(chunks, i, chunk);
        }
        PyObject * sep = PyBytes_FromStringAndSize(NULL, 0);
        PyObject * res = PyObject_CallMethod(sep, "join", "(N)", chunks);
        Py_DECREF(sep);
        return res;
    }

    ImageFace * src = (ImageFace *)PyTuple_GetItem(self->layers, 0);
    return read_image_face(src, size_arg, offset_arg);
}

static PyObject * Image_meth_blit(Image * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"target", "target_viewport", "source_viewport", "filter", "srgb", NULL};

    PyObject * target = Py_None;
    PyObject * target_viewport = Py_None;
    PyObject * source_viewport = Py_None;
    int filter = 1;
    PyObject * srgb = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOpO", keywords, &target, &target_viewport, &source_viewport, &filter, &srgb)) {
        return NULL;
    }

    ImageFace * src = (ImageFace *)PyTuple_GetItem(self->layers, 0);
    return blit_image_face(src, target, source_viewport, target_viewport, filter, srgb);
}

static ImageFace * Image_meth_face(Image * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"layer", "level", NULL};

    int layer = 0;
    int level = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii", keywords, &layer, &level)) {
        return NULL;
    }

    if (layer < 0 || layer >= self->layer_count) {
        PyErr_Format(PyExc_ValueError, "invalid layer");
        return NULL;
    }

    if (level > self->level_count) {
        PyErr_Format(PyExc_ValueError, "invalid level");
        return NULL;
    }

    PyObject * key = Py_BuildValue("(ii)", layer, level);
    ImageFace * res = build_image_face(self, key);
    Py_DECREF(key);
    return res;
}

static PyObject * Image_get_clear_value(Image * self, void * closure) {
    if (self->fmt->clear_type == 'x') {
        return Py_BuildValue("fI", self->clear_value.clear_floats[0], self->clear_value.clear_uints[1]);
    }
    if (self->fmt->components == 1) {
        if (self->fmt->clear_type == 'f') {
            return PyFloat_FromDouble(self->clear_value.clear_floats[0]);
        } else if (self->fmt->clear_type == 'i') {
            return PyLong_FromLong(self->clear_value.clear_ints[0]);
        } else if (self->fmt->clear_type == 'u') {
            return PyLong_FromUnsignedLong(self->clear_value.clear_uints[0]);
        }
    }
    PyObject * res = PyTuple_New(self->fmt->components);
    for (int i = 0; i < self->fmt->components; ++i) {
        if (self->fmt->clear_type == 'f') {
            PyTuple_SetItem(res, i, PyFloat_FromDouble(self->clear_value.clear_floats[i]));
        } else if (self->fmt->clear_type == 'i') {
            PyTuple_SetItem(res, i, PyLong_FromLong(self->clear_value.clear_ints[i]));
        } else if (self->fmt->clear_type == 'u') {
            PyTuple_SetItem(res, i, PyLong_FromUnsignedLong(self->clear_value.clear_uints[i]));
        }
    }
    return res;
}

static int Image_set_clear_value(Image * self, PyObject * value, void * closure) {
    if (self->fmt->components == 1) {
        if (self->fmt->clear_type == 'f' && !PyFloat_CheckExact(value)) {
            PyErr_Format(PyExc_TypeError, "the clear value must be a float");
            return -1;
        }
        if (self->fmt->clear_type == 'i' && !PyLong_CheckExact(value)) {
            PyErr_Format(PyExc_TypeError, "the clear value must be an int");
            return -1;
        }
        if (self->fmt->clear_type == 'f') {
            self->clear_value.clear_floats[0] = (float)PyFloat_AsDouble(value);
        } else if (self->fmt->clear_type == 'i') {
            self->clear_value.clear_ints[0] = PyLong_AsLong(value);
        } else if (self->fmt->clear_type == 'u') {
            self->clear_value.clear_uints[0] = PyLong_AsUnsignedLong(value);
        }
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

    if (size != self->fmt->components) {
        Py_DECREF(values);
        PyErr_Format(PyExc_ValueError, "invalid clear value size");
        return -1;
    }

    if (self->fmt->clear_type == 'f') {
        for (int i = 0; i < self->fmt->components; ++i) {
            self->clear_value.clear_floats[i] = (float)PyFloat_AsDouble(seq[i]);
        }
    } else if (self->fmt->clear_type == 'i') {
        for (int i = 0; i < self->fmt->components; ++i) {
            self->clear_value.clear_ints[i] = PyLong_AsLong(seq[i]);
        }
    } else if (self->fmt->clear_type == 'u') {
        for (int i = 0; i < self->fmt->components; ++i) {
            self->clear_value.clear_uints[i] = PyLong_AsUnsignedLong(seq[i]);
        }
    } else if (self->fmt->clear_type == 'x') {
        self->clear_value.clear_floats[0] = (float)PyFloat_AsDouble(seq[0]);
        self->clear_value.clear_ints[1] = PyLong_AsLong(seq[1]);
    }
    if (PyErr_Occurred()) {
        Py_DECREF(values);
        return -1;
    }
    Py_DECREF(values);
    return 0;
}

static PyObject * Pipeline_meth_render(Pipeline * self, PyObject * args) {
    const GLMethods * const gl = &self->ctx->gl;
    if (memcmp(&self->viewport, &self->ctx->current_viewport, sizeof(Viewport))) {
        gl->Viewport(self->viewport.x, self->viewport.y, self->viewport.width, self->viewport.height);
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
        gl->DrawElementsInstanced(self->topology, self->vertex_count, self->index_type, (void *)offset, self->instance_count);
    } else {
        gl->DrawArraysInstanced(self->topology, self->first_vertex, self->vertex_count, self->instance_count);
    }
    Py_RETURN_NONE;
}

static PyObject * Pipeline_get_viewport(Pipeline * self, void * closure) {
    return Py_BuildValue("iiii", self->viewport.x, self->viewport.y, self->viewport.width, self->viewport.height);
}

static int Pipeline_set_viewport(Pipeline * self, PyObject * viewport, void * closure) {
    self->viewport = to_viewport(viewport, 0, 0, 0, 0);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "the viewport must be a tuple of 4 ints");
        return -1;
    }
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
        return Py_BuildValue("{sssi}", "type", "image", gltype, image->image);
    } else if (Py_TYPE(arg) == module_state->ImageFace_type) {
        ImageFace * face = (ImageFace *)arg;
        return Py_BuildValue("{sssi}", "type", "image_face", "framebuffer", face->framebuffer->obj);
    } else if (Py_TYPE(arg) == module_state->Pipeline_type) {
        Pipeline * pipeline = (Pipeline *)arg;
        return Py_BuildValue(
            "{sssOsNsisisi}",
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

static PyObject * ImageFace_meth_read(ImageFace * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"size", "offset", NULL};

    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", keywords, &size_arg, &offset_arg)) {
        return NULL;
    }

    return read_image_face(self, size_arg, offset_arg);
}

static PyObject * ImageFace_meth_blit(ImageFace * self, PyObject * args, PyObject * kwargs) {
    static char * keywords[] = {"target", "target_viewport", "source_viewport", "filter", "srgb", NULL};

    PyObject * target = Py_None;
    PyObject * target_viewport = Py_None;
    PyObject * source_viewport = Py_None;
    int filter = 1;
    PyObject * srgb = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOpO", keywords, &target, &target_viewport, &source_viewport, &filter, &srgb)) {
        return NULL;
    }

    return blit_image_face(self, target, source_viewport, target_viewport, filter, srgb);
}

typedef struct vec3 {
    double x, y, z;
} vec3;

static vec3 sub(const vec3 a, const vec3 b) {
    vec3 res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    res.z = a.z - b.z;
    return res;
}

static vec3 normalize(const vec3 a) {
    const double l = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    vec3 res;
    res.x = a.x / l;
    res.y = a.y / l;
    res.z = a.z / l;
    return res;
}

static vec3 cross(const vec3 a, const vec3 b) {
    vec3 res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;
    return res;
}

static double dot(const vec3 a, const vec3 b) {
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
    int clip = 0;

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

    const vec3 f = normalize(sub(target, eye));
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
    Py_DECREF(self->loader);
    Py_DECREF(self->descriptor_set_cache);
    Py_DECREF(self->global_settings_cache);
    Py_DECREF(self->sampler_cache);
    Py_DECREF(self->vertex_array_cache);
    Py_DECREF(self->framebuffer_cache);
    Py_DECREF(self->program_cache);
    Py_DECREF(self->shader_cache);
    Py_DECREF(self->includes);
    Py_DECREF(self->default_framebuffer);
    Py_DECREF(self->before_frame_callback);
    Py_DECREF(self->after_frame_callback);
    Py_DECREF(self->limits_dict);
    Py_DECREF(self->info_dict);
    Py_TYPE(self)->tp_free(self);
}

static void Buffer_dealloc(Buffer * self) {
    Py_TYPE(self)->tp_free(self);
}

static void Image_dealloc(Image * self) {
    Py_DECREF(self->size);
    Py_DECREF(self->format);
    Py_DECREF(self->faces);
    Py_DECREF(self->layers);
    Py_TYPE(self)->tp_free(self);
}

static void Pipeline_dealloc(Pipeline * self) {
    Py_DECREF(self->descriptor_set);
    Py_DECREF(self->global_settings);
    Py_DECREF(self->framebuffer);
    Py_DECREF(self->vertex_array);
    Py_DECREF(self->program);
    if (self->uniform_map) {
        Py_DECREF(self->uniform_map);
    }
    Py_TYPE(self)->tp_free(self);
}

static void ImageFace_dealloc(ImageFace * self) {
    Py_DECREF(self->framebuffer);
    Py_DECREF(self->size);
    Py_TYPE(self)->tp_free(self);
}

static void DescriptorSet_dealloc(DescriptorSet * self) {
    Py_TYPE(self)->tp_free(self);
}

static void GlobalSettings_dealloc(GlobalSettings * self) {
    Py_TYPE(self)->tp_free(self);
}

static void GLObject_dealloc(GLObject * self) {
    if (self->extra) {
        Py_DECREF(self->extra);
    }
    Py_TYPE(self)->tp_free(self);
}

static PyMethodDef Context_methods[] = {
    {"buffer", (PyCFunction)Context_meth_buffer, METH_VARARGS | METH_KEYWORDS},
    {"image", (PyCFunction)Context_meth_image, METH_VARARGS | METH_KEYWORDS},
    {"pipeline", (PyCFunction)Context_meth_pipeline, METH_VARARGS | METH_KEYWORDS},
    {"new_frame", (PyCFunction)Context_meth_new_frame, METH_VARARGS | METH_KEYWORDS},
    {"end_frame", (PyCFunction)Context_meth_end_frame, METH_VARARGS | METH_KEYWORDS},
    {"release", (PyCFunction)Context_meth_release, METH_O},
    {"gc", (PyCFunction)Context_meth_gc, METH_NOARGS},
    {NULL},
};

static PyGetSetDef Context_getset[] = {
    {"screen", (getter)Context_get_screen, (setter)Context_set_screen},
    {NULL},
};

static PyMemberDef Context_members[] = {
    {"includes", T_OBJECT, offsetof(Context, includes), READONLY},
    {"limits", T_OBJECT, offsetof(Context, limits_dict), READONLY},
    {"info", T_OBJECT, offsetof(Context, info_dict), READONLY},
    {"before_frame", T_OBJECT, offsetof(Context, before_frame_callback), 0},
    {"after_frame", T_OBJECT, offsetof(Context, after_frame_callback), 0},
    {"frame_time", T_INT, offsetof(Context, frame_time), READONLY},
    {NULL},
};

static PyMethodDef Buffer_methods[] = {
    {"write", (PyCFunction)Buffer_meth_write, METH_VARARGS | METH_KEYWORDS},
    {"map", (PyCFunction)Buffer_meth_map, METH_VARARGS | METH_KEYWORDS},
    {"unmap", (PyCFunction)Buffer_meth_unmap, METH_NOARGS},
    {NULL},
};

static PyMemberDef Buffer_members[] = {
    {"size", T_INT, offsetof(Buffer, size), READONLY},
    {NULL},
};

static PyMethodDef Image_methods[] = {
    {"clear", (PyCFunction)Image_meth_clear, METH_NOARGS},
    {"write", (PyCFunction)Image_meth_write, METH_VARARGS | METH_KEYWORDS},
    {"read", (PyCFunction)Image_meth_read, METH_VARARGS | METH_KEYWORDS},
    {"mipmaps", (PyCFunction)Image_meth_mipmaps, METH_NOARGS},
    {"blit", (PyCFunction)Image_meth_blit, METH_VARARGS | METH_KEYWORDS},
    {"face", (PyCFunction)Image_meth_face, METH_VARARGS | METH_KEYWORDS},
    {NULL},
};

static PyGetSetDef Image_getset[] = {
    {"clear_value", (getter)Image_get_clear_value, (setter)Image_set_clear_value},
    {NULL},
};

static PyMemberDef Image_members[] = {
    {"size", T_OBJECT, offsetof(Image, size), READONLY},
    {"format", T_OBJECT, offsetof(Image, format), READONLY},
    {"samples", T_INT, offsetof(Image, samples), READONLY},
    {"array", T_INT, offsetof(Image, array), READONLY},
    {NULL},
};

static PyMethodDef Pipeline_methods[] = {
    {"render", (PyCFunction)Pipeline_meth_render, METH_NOARGS},
    {NULL},
};

static PyGetSetDef Pipeline_getset[] = {
    {"viewport", (getter)Pipeline_get_viewport, (setter)Pipeline_set_viewport},
    {NULL},
};

static PyMemberDef Pipeline_members[] = {
    {"vertex_count", T_INT, offsetof(Pipeline, vertex_count), 0},
    {"instance_count", T_INT, offsetof(Pipeline, instance_count), 0},
    {"first_vertex", T_INT, offsetof(Pipeline, first_vertex), 0},
    {"uniforms", T_OBJECT_EX, offsetof(Pipeline, uniform_map), READONLY},
    {NULL},
};

static PyMethodDef ImageFace_methods[] = {
    {"clear", (PyCFunction)ImageFace_meth_clear, METH_NOARGS},
    {"read", (PyCFunction)ImageFace_meth_read, METH_VARARGS | METH_KEYWORDS},
    {"blit", (PyCFunction)ImageFace_meth_blit, METH_VARARGS | METH_KEYWORDS},
    {NULL},
};

static PyMemberDef ImageFace_members[] = {
    {"image", T_OBJECT, offsetof(ImageFace, image), READONLY},
    {"size", T_OBJECT, offsetof(ImageFace, size), READONLY},
    {"layer", T_INT, offsetof(ImageFace, layer), READONLY},
    {"level", T_INT, offsetof(ImageFace, level), READONLY},
    {"samples", T_INT, offsetof(ImageFace, samples), READONLY},
    {"flags", T_INT, offsetof(ImageFace, flags), READONLY},
    {NULL},
};

static PyType_Slot Context_slots[] = {
    {Py_tp_methods, Context_methods},
    {Py_tp_getset, Context_getset},
    {Py_tp_members, Context_members},
    {Py_tp_dealloc, (void *)Context_dealloc},
    {0},
};

static PyType_Slot Buffer_slots[] = {
    {Py_tp_methods, Buffer_methods},
    {Py_tp_members, Buffer_members},
    {Py_tp_dealloc, (void *)Buffer_dealloc},
    {0},
};

static PyType_Slot Image_slots[] = {
    {Py_tp_methods, Image_methods},
    {Py_tp_getset, Image_getset},
    {Py_tp_members, Image_members},
    {Py_tp_dealloc, (void *)Image_dealloc},
    {0},
};

static PyType_Slot Pipeline_slots[] = {
    {Py_tp_methods, Pipeline_methods},
    {Py_tp_getset, Pipeline_getset},
    {Py_tp_members, Pipeline_members},
    {Py_tp_dealloc, (void *)Pipeline_dealloc},
    {0},
};

static PyType_Slot ImageFace_slots[] = {
    {Py_tp_methods, ImageFace_methods},
    {Py_tp_members, ImageFace_members},
    {Py_tp_dealloc, (void *)ImageFace_dealloc},
    {0},
};

static PyType_Slot DescriptorSet_slots[] = {
    {Py_tp_dealloc, (void *)DescriptorSet_dealloc},
    {0},
};

static PyType_Slot GlobalSettings_slots[] = {
    {Py_tp_dealloc, (void *)GlobalSettings_dealloc},
    {0},
};

static PyType_Slot GLObject_slots[] = {
    {Py_tp_dealloc, (void *)GLObject_dealloc},
    {0},
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

    PyModule_AddObject(self, "__version__", PyUnicode_FromString("1.12.2"));

    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, (void *)module_exec},
    {0},
};

static PyMethodDef module_methods[] = {
    {"context", (PyCFunction)meth_context, METH_VARARGS | METH_KEYWORDS},
    {"inspect", (PyCFunction)meth_inspect, METH_O},
    {"camera", (PyCFunction)meth_camera, METH_VARARGS | METH_KEYWORDS},
    {NULL},
};

static void module_free(PyObject * self) {
    ModuleState * state = (ModuleState *)PyModule_GetState(self);
    if (state) {
        Py_DECREF(state->empty_tuple);
        Py_DECREF(state->str_none);
        Py_DECREF(state->Context_type);
        Py_DECREF(state->Buffer_type);
        Py_DECREF(state->Image_type);
        Py_DECREF(state->Pipeline_type);
        Py_DECREF(state->ImageFace_type);
        Py_DECREF(state->DescriptorSet_type);
        Py_DECREF(state->GlobalSettings_type);
        Py_DECREF(state->GLObject_type);
    }
}

static PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "zengl", NULL, sizeof(ModuleState), module_methods, module_slots, NULL, NULL, (freefunc)module_free,
};

extern PyObject * PyInit_zengl() {
    return PyModuleDef_Init(&module_def);
}
