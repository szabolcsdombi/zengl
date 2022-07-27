#include <Python.h>
#include <structmember.h>

const int MAX_ATTACHMENTS = 16;
const int MAX_UNIFORM_BUFFER_BINDINGS = 16;
const int MAX_SAMPLER_BINDINGS = 64;
const int MAX_UNIFORM_BINDINGS = 64;

#if defined(_WIN32) || defined(_WIN64)
#define GLAPI __stdcall
#else
#define GLAPI
#endif

#if defined(__x86_64__) || defined(_WIN64)
typedef long long int sizeiptr;
#else
typedef int sizeiptr;
#endif

// GL_VERSION_1_0
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

// GL_VERSION_1_1
#define GL_POLYGON_OFFSET_POINT 0x2A01
#define GL_POLYGON_OFFSET_LINE 0x2A02
#define GL_POLYGON_OFFSET_FILL 0x8037
#define GL_RGBA8 0x8058

// GL_VERSION_1_2
#define GL_TEXTURE_WRAP_R 0x8072
#define GL_BGRA 0x80E1
#define GL_TEXTURE_MIN_LOD 0x813A
#define GL_TEXTURE_MAX_LOD 0x813B
#define GL_TEXTURE_BASE_LEVEL 0x813C
#define GL_TEXTURE_MAX_LEVEL 0x813D

// GL_VERSION_1_3
#define GL_TEXTURE0 0x84C0
#define GL_TEXTURE_CUBE_MAP 0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X 0x8515

// GL_VERSION_1_4
#define GL_DEPTH_COMPONENT16 0x81A5
#define GL_DEPTH_COMPONENT24 0x81A6
#define GL_TEXTURE_LOD_BIAS 0x8501
#define GL_TEXTURE_COMPARE_MODE 0x884C
#define GL_TEXTURE_COMPARE_FUNC 0x884D

// GL_VERSION_1_5
#define GL_ARRAY_BUFFER 0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_DRAW 0x88E8

// GL_VERSION_2_0
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

// GL_VERSION_2_1
#define GL_SRGB8_ALPHA8 0x8C43

// GL_VERSION_3_0
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

// GL_VERSION_3_1
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

// GL_VERSION_3_2
#define GL_PROGRAM_POINT_SIZE 0x8642
#define GL_TEXTURE_CUBE_MAP_SEAMLESS 0x884F

// EXTENSION
#define GL_TEXTURE_MAX_ANISOTROPY 0x84FE

// GL_VERSION_1_0
typedef void (GLAPI * glCullFaceProc)(unsigned int mode);
typedef void (GLAPI * glFrontFaceProc)(unsigned int mode);
typedef void (GLAPI * glLineWidthProc)(float width);
typedef void (GLAPI * glTexParameteriProc)(unsigned int target, unsigned int pname, int param);
typedef void (GLAPI * glTexImage2DProc)(unsigned int target, int level, int internalformat, int width, int height, int border, unsigned int format, unsigned int type, const void * pixels);
typedef void (GLAPI * glDepthMaskProc)(unsigned char flag);
typedef void (GLAPI * glDisableProc)(unsigned int cap);
typedef void (GLAPI * glEnableProc)(unsigned int cap);
typedef void (GLAPI * glFlushProc)();
typedef void (GLAPI * glDepthFuncProc)(unsigned int func);
typedef void (GLAPI * glReadBufferProc)(unsigned int src);
typedef void (GLAPI * glReadPixelsProc)(int x, int y, int width, int height, unsigned int format, unsigned int type, void * pixels);
typedef unsigned int (GLAPI * glGetErrorProc)();
typedef void (GLAPI * glGetIntegervProc)(unsigned int pname, int * data);
typedef const unsigned char * (GLAPI * glGetStringProc)(unsigned int name);
typedef void (GLAPI * glViewportProc)(int x, int y, int width, int height);

// GL_VERSION_1_1
typedef void (GLAPI * glPolygonOffsetProc)(float factor, float units);
typedef void (GLAPI * glTexSubImage2DProc)(unsigned int target, int level, int xoffset, int yoffset, int width, int height, unsigned int format, unsigned int type, const void * pixels);
typedef void (GLAPI * glBindTextureProc)(unsigned int target, unsigned int texture);
typedef void (GLAPI * glDeleteTexturesProc)(int n, const unsigned int * textures);
typedef void (GLAPI * glGenTexturesProc)(int n, unsigned int * textures);

// GL_VERSION_1_2
typedef void (GLAPI * glTexImage3DProc)(unsigned int target, int level, int internalformat, int width, int height, int depth, int border, unsigned int format, unsigned int type, const void * pixels);
typedef void (GLAPI * glTexSubImage3DProc)(unsigned int target, int level, int xoffset, int yoffset, int zoffset, int width, int height, int depth, unsigned int format, unsigned int type, const void * pixels);

// GL_VERSION_1_3
typedef void (GLAPI * glActiveTextureProc)(unsigned int texture);

// GL_VERSION_1_4
typedef void (GLAPI * glBlendFuncSeparateProc)(unsigned int sfactorRGB, unsigned int dfactorRGB, unsigned int sfactorAlpha, unsigned int dfactorAlpha);

// GL_VERSION_1_5
typedef void (GLAPI * glBindBufferProc)(unsigned int target, unsigned int buffer);
typedef void (GLAPI * glDeleteBuffersProc)(int n, const unsigned int * buffers);
typedef void (GLAPI * glGenBuffersProc)(int n, unsigned int * buffers);
typedef void (GLAPI * glBufferDataProc)(unsigned int target, sizeiptr size, const void * data, unsigned int usage);
typedef void (GLAPI * glBufferSubDataProc)(unsigned int target, sizeiptr offset, sizeiptr size, const void * data);
typedef unsigned char (GLAPI * glUnmapBufferProc)(unsigned int target);

// GL_VERSION_2_0
typedef void (GLAPI * glDrawBuffersProc)(int n, const unsigned int * bufs);
typedef void (GLAPI * glStencilOpSeparateProc)(unsigned int face, unsigned int sfail, unsigned int dpfail, unsigned int dppass);
typedef void (GLAPI * glStencilFuncSeparateProc)(unsigned int face, unsigned int func, int ref, unsigned int mask);
typedef void (GLAPI * glStencilMaskSeparateProc)(unsigned int face, unsigned int mask);
typedef void (GLAPI * glAttachShaderProc)(unsigned int program, unsigned int shader);
typedef void (GLAPI * glCompileShaderProc)(unsigned int shader);
typedef unsigned int (GLAPI * glCreateProgramProc)();
typedef unsigned int (GLAPI * glCreateShaderProc)(unsigned int type);
typedef void (GLAPI * glDeleteProgramProc)(unsigned int program);
typedef void (GLAPI * glDeleteShaderProc)(unsigned int shader);
typedef void (GLAPI * glEnableVertexAttribArrayProc)(unsigned int index);
typedef void (GLAPI * glGetActiveAttribProc)(unsigned int program, unsigned int index, int bufSize, int * length, int * size, unsigned int * type, char * name);
typedef void (GLAPI * glGetActiveUniformProc)(unsigned int program, unsigned int index, int bufSize, int * length, int * size, unsigned int * type, char * name);
typedef int (GLAPI * glGetAttribLocationProc)(unsigned int program, const char * name);
typedef void (GLAPI * glGetProgramivProc)(unsigned int program, unsigned int pname, int * params);
typedef void (GLAPI * glGetProgramInfoLogProc)(unsigned int program, int bufSize, int * length, char * infoLog);
typedef void (GLAPI * glGetShaderivProc)(unsigned int shader, unsigned int pname, int * params);
typedef void (GLAPI * glGetShaderInfoLogProc)(unsigned int shader, int bufSize, int * length, char * infoLog);
typedef int (GLAPI * glGetUniformLocationProc)(unsigned int program, const char * name);
typedef void (GLAPI * glLinkProgramProc)(unsigned int program);
typedef void (GLAPI * glShaderSourceProc)(unsigned int shader, int count, const char * const * string, const int * length);
typedef void (GLAPI * glUseProgramProc)(unsigned int program);
typedef void (GLAPI * glUniform1iProc)(int location, int v0);
typedef void (GLAPI * glUniform1ivProc)(int location, int count, int * value);
typedef void (GLAPI * glUniform2ivProc)(int location, int count, int * value);
typedef void (GLAPI * glUniform3ivProc)(int location, int count, int * value);
typedef void (GLAPI * glUniform4ivProc)(int location, int count, int * value);
typedef void (GLAPI * glUniform1fvProc)(int location, int count, float * value);
typedef void (GLAPI * glUniform2fvProc)(int location, int count, float * value);
typedef void (GLAPI * glUniform3fvProc)(int location, int count, float * value);
typedef void (GLAPI * glUniform4fvProc)(int location, int count, float * value);
typedef void (GLAPI * glUniformMatrix2fvProc)(int location, int count, unsigned char transpose, float * value);
typedef void (GLAPI * glUniformMatrix2x3fvProc)(int location, int count, unsigned char transpose, float * value);
typedef void (GLAPI * glUniformMatrix2x4fvProc)(int location, int count, unsigned char transpose, float * value);
typedef void (GLAPI * glUniformMatrix3x2fvProc)(int location, int count, unsigned char transpose, float * value);
typedef void (GLAPI * glUniformMatrix3fvProc)(int location, int count, unsigned char transpose, float * value);
typedef void (GLAPI * glUniformMatrix3x4fvProc)(int location, int count, unsigned char transpose, float * value);
typedef void (GLAPI * glUniformMatrix4x2fvProc)(int location, int count, unsigned char transpose, float * value);
typedef void (GLAPI * glUniformMatrix4x3fvProc)(int location, int count, unsigned char transpose, float * value);
typedef void (GLAPI * glUniformMatrix4fvProc)(int location, int count, unsigned char transpose, float * value);
typedef void (GLAPI * glVertexAttribPointerProc)(unsigned int index, int size, unsigned int type, unsigned char normalized, int stride, const void * pointer);

// GL_VERSION_3_0
typedef void (GLAPI * glColorMaskiProc)(unsigned int index, unsigned char r, unsigned char g, unsigned char b, unsigned char a);
typedef void (GLAPI * glEnableiProc)(unsigned int target, unsigned int index);
typedef void (GLAPI * glDisableiProc)(unsigned int target, unsigned int index);
typedef void (GLAPI * glBindBufferRangeProc)(unsigned int target, unsigned int index, unsigned int buffer, sizeiptr offset, sizeiptr size);
typedef void (GLAPI * glVertexAttribIPointerProc)(unsigned int index, int size, unsigned int type, int stride, const void * pointer);
typedef void (GLAPI * glUniform1uivProc)(int location, int count, unsigned int * value);
typedef void (GLAPI * glUniform2uivProc)(int location, int count, unsigned int * value);
typedef void (GLAPI * glUniform3uivProc)(int location, int count, unsigned int * value);
typedef void (GLAPI * glUniform4uivProc)(int location, int count, unsigned int * value);
typedef void (GLAPI * glClearBufferivProc)(unsigned int buffer, int drawbuffer, const int * value);
typedef void (GLAPI * glClearBufferuivProc)(unsigned int buffer, int drawbuffer, const unsigned int * value);
typedef void (GLAPI * glClearBufferfvProc)(unsigned int buffer, int drawbuffer, const float * value);
typedef void (GLAPI * glClearBufferfiProc)(unsigned int buffer, int drawbuffer, float depth, int stencil);
typedef void (GLAPI * glBindRenderbufferProc)(unsigned int target, unsigned int renderbuffer);
typedef void (GLAPI * glDeleteRenderbuffersProc)(int n, const unsigned int * renderbuffers);
typedef void (GLAPI * glGenRenderbuffersProc)(int n, unsigned int * renderbuffers);
typedef void (GLAPI * glBindFramebufferProc)(unsigned int target, unsigned int framebuffer);
typedef void (GLAPI * glDeleteFramebuffersProc)(int n, const unsigned int * framebuffers);
typedef void (GLAPI * glGenFramebuffersProc)(int n, unsigned int * framebuffers);
typedef void (GLAPI * glFramebufferTexture2DProc)(unsigned int target, unsigned int attachment, unsigned int textarget, unsigned int texture, int level);
typedef void (GLAPI * glFramebufferRenderbufferProc)(unsigned int target, unsigned int attachment, unsigned int renderbuffertarget, unsigned int renderbuffer);
typedef void (GLAPI * glGenerateMipmapProc)(unsigned int target);
typedef void (GLAPI * glBlitFramebufferProc)(int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0, int dstX1, int dstY1, unsigned int mask, unsigned int filter);
typedef void (GLAPI * glRenderbufferStorageMultisampleProc)(unsigned int target, int samples, unsigned int internalformat, int width, int height);
typedef void (GLAPI * glFramebufferTextureLayerProc)(unsigned int target, unsigned int attachment, unsigned int texture, int level, int layer);
typedef void * (GLAPI * glMapBufferRangeProc)(unsigned int target, sizeiptr offset, sizeiptr length, unsigned int access);
typedef void (GLAPI * glBindVertexArrayProc)(unsigned int array);
typedef void (GLAPI * glDeleteVertexArraysProc)(int n, const unsigned int * arrays);
typedef void (GLAPI * glGenVertexArraysProc)(int n, unsigned int * arrays);

// GL_VERSION_3_1
typedef void (GLAPI * glDrawArraysInstancedProc)(unsigned int mode, int first, int count, int instancecount);
typedef void (GLAPI * glDrawElementsInstancedProc)(unsigned int mode, int count, unsigned int type, const void * indices, int instancecount);
typedef void (GLAPI * glPrimitiveRestartIndexProc)(unsigned int index);
typedef unsigned int (GLAPI * glGetUniformBlockIndexProc)(unsigned int program, const char * uniformBlockName);
typedef void (GLAPI * glGetActiveUniformBlockivProc)(unsigned int program, unsigned int uniformBlockIndex, unsigned int pname, int * params);
typedef void (GLAPI * glGetActiveUniformBlockNameProc)(unsigned int program, unsigned int uniformBlockIndex, int bufSize, int * length, char * uniformBlockName);
typedef void (GLAPI * glUniformBlockBindingProc)(unsigned int program, unsigned int uniformBlockIndex, unsigned int uniformBlockBinding);

// GL_VERSION_3_3
typedef void (GLAPI * glGenSamplersProc)(int count, unsigned int * samplers);
typedef void (GLAPI * glDeleteSamplersProc)(int count, const unsigned int * samplers);
typedef void (GLAPI * glBindSamplerProc)(unsigned int unit, unsigned int sampler);
typedef void (GLAPI * glSamplerParameteriProc)(unsigned int sampler, unsigned int pname, int param);
typedef void (GLAPI * glSamplerParameterfProc)(unsigned int sampler, unsigned int pname, float param);
typedef void (GLAPI * glSamplerParameterfvProc)(unsigned int sampler, unsigned int pname, const float * param);
typedef void (GLAPI * glVertexAttribDivisorProc)(unsigned int index, unsigned int divisor);

struct GLMethods {
    // GL_VERSION_1_0
    glCullFaceProc CullFace;
    glFrontFaceProc FrontFace;
    glLineWidthProc LineWidth;
    glTexParameteriProc TexParameteri;
    glTexImage2DProc TexImage2D;
    glDepthMaskProc DepthMask;
    glDisableProc Disable;
    glEnableProc Enable;
    glFlushProc Flush;
    glDepthFuncProc DepthFunc;
    glReadBufferProc ReadBuffer;
    glReadPixelsProc ReadPixels;
    glGetErrorProc GetError;
    glGetIntegervProc GetIntegerv;
    glGetStringProc GetString;
    glViewportProc Viewport;

    // GL_VERSION_1_1
    glPolygonOffsetProc PolygonOffset;
    glTexSubImage2DProc TexSubImage2D;
    glBindTextureProc BindTexture;
    glDeleteTexturesProc DeleteTextures;
    glGenTexturesProc GenTextures;

    // GL_VERSION_1_2
    glTexImage3DProc TexImage3D;
    glTexSubImage3DProc TexSubImage3D;

    // GL_VERSION_1_3
    glActiveTextureProc ActiveTexture;

    // GL_VERSION_1_4
    glBlendFuncSeparateProc BlendFuncSeparate;

    // GL_VERSION_1_5
    glBindBufferProc BindBuffer;
    glDeleteBuffersProc DeleteBuffers;
    glGenBuffersProc GenBuffers;
    glBufferDataProc BufferData;
    glBufferSubDataProc BufferSubData;
    glUnmapBufferProc UnmapBuffer;

    // GL_VERSION_2_0
    glDrawBuffersProc DrawBuffers;
    glStencilOpSeparateProc StencilOpSeparate;
    glStencilFuncSeparateProc StencilFuncSeparate;
    glStencilMaskSeparateProc StencilMaskSeparate;
    glAttachShaderProc AttachShader;
    glCompileShaderProc CompileShader;
    glCreateProgramProc CreateProgram;
    glCreateShaderProc CreateShader;
    glDeleteProgramProc DeleteProgram;
    glDeleteShaderProc DeleteShader;
    glEnableVertexAttribArrayProc EnableVertexAttribArray;
    glGetActiveAttribProc GetActiveAttrib;
    glGetActiveUniformProc GetActiveUniform;
    glGetAttribLocationProc GetAttribLocation;
    glGetProgramivProc GetProgramiv;
    glGetProgramInfoLogProc GetProgramInfoLog;
    glGetShaderivProc GetShaderiv;
    glGetShaderInfoLogProc GetShaderInfoLog;
    glGetUniformLocationProc GetUniformLocation;
    glLinkProgramProc LinkProgram;
    glShaderSourceProc ShaderSource;
    glUseProgramProc UseProgram;
    glUniform1iProc Uniform1i;
    glUniform1ivProc Uniform1iv;
    glUniform2ivProc Uniform2iv;
    glUniform3ivProc Uniform3iv;
    glUniform4ivProc Uniform4iv;
    glUniform1fvProc Uniform1fv;
    glUniform2fvProc Uniform2fv;
    glUniform3fvProc Uniform3fv;
    glUniform4fvProc Uniform4fv;
    glUniformMatrix2fvProc UniformMatrix2fv;
    glUniformMatrix2x3fvProc UniformMatrix2x3fv;
    glUniformMatrix2x4fvProc UniformMatrix2x4fv;
    glUniformMatrix3x2fvProc UniformMatrix3x2fv;
    glUniformMatrix3fvProc UniformMatrix3fv;
    glUniformMatrix3x4fvProc UniformMatrix3x4fv;
    glUniformMatrix4x2fvProc UniformMatrix4x2fv;
    glUniformMatrix4x3fvProc UniformMatrix4x3fv;
    glUniformMatrix4fvProc UniformMatrix4fv;
    glVertexAttribPointerProc VertexAttribPointer;

    // GL_VERSION_3_0
    glColorMaskiProc ColorMaski;
    glEnableiProc Enablei;
    glDisableiProc Disablei;
    glBindBufferRangeProc BindBufferRange;
    glVertexAttribIPointerProc VertexAttribIPointer;
    glUniform1uivProc Uniform1uiv;
    glUniform2uivProc Uniform2uiv;
    glUniform3uivProc Uniform3uiv;
    glUniform4uivProc Uniform4uiv;
    glClearBufferivProc ClearBufferiv;
    glClearBufferuivProc ClearBufferuiv;
    glClearBufferfvProc ClearBufferfv;
    glClearBufferfiProc ClearBufferfi;
    glBindRenderbufferProc BindRenderbuffer;
    glDeleteRenderbuffersProc DeleteRenderbuffers;
    glGenRenderbuffersProc GenRenderbuffers;
    glBindFramebufferProc BindFramebuffer;
    glDeleteFramebuffersProc DeleteFramebuffers;
    glGenFramebuffersProc GenFramebuffers;
    glFramebufferTexture2DProc FramebufferTexture2D;
    glFramebufferRenderbufferProc FramebufferRenderbuffer;
    glGenerateMipmapProc GenerateMipmap;
    glBlitFramebufferProc BlitFramebuffer;
    glRenderbufferStorageMultisampleProc RenderbufferStorageMultisample;
    glFramebufferTextureLayerProc FramebufferTextureLayer;
    glMapBufferRangeProc MapBufferRange;
    glBindVertexArrayProc BindVertexArray;
    glDeleteVertexArraysProc DeleteVertexArrays;
    glGenVertexArraysProc GenVertexArrays;

    // GL_VERSION_3_1
    glDrawArraysInstancedProc DrawArraysInstanced;
    glDrawElementsInstancedProc DrawElementsInstanced;
    glPrimitiveRestartIndexProc PrimitiveRestartIndex;
    glGetUniformBlockIndexProc GetUniformBlockIndex;
    glGetActiveUniformBlockivProc GetActiveUniformBlockiv;
    glGetActiveUniformBlockNameProc GetActiveUniformBlockName;
    glUniformBlockBindingProc UniformBlockBinding;

    // GL_VERSION_3_3
    glGenSamplersProc GenSamplers;
    glDeleteSamplersProc DeleteSamplers;
    glBindSamplerProc BindSampler;
    glSamplerParameteriProc SamplerParameteri;
    glSamplerParameterfProc SamplerParameterf;
    glSamplerParameterfvProc SamplerParameterfv;
    glVertexAttribDivisorProc VertexAttribDivisor;
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

union Viewport {
    unsigned long long viewport;
    struct {
        short x;
        short y;
        short width;
        short height;
    };
};

union ClearValue {
    float clear_floats[4];
    int clear_ints[4];
    unsigned int clear_uints[4];
};

union IntPair {
    unsigned long long pair;
    struct {
        int x;
        int y;
    };
};

VertexFormat get_vertex_format(const char * format) {
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

ImageFormat get_image_format(const char * format) {
    if (!strcmp(format, "r8unorm")) return {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1, 1, GL_COLOR, true, 'f'};
    if (!strcmp(format, "rg8unorm")) return {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2, 2, GL_COLOR, true, 'f'};
    if (!strcmp(format, "rgba8unorm")) return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, true, 'f'};
    if (!strcmp(format, "bgra8unorm")) return {GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, true, 'f'};
    if (!strcmp(format, "r8snorm")) return {GL_R8_SNORM, GL_RED, GL_UNSIGNED_BYTE, 1, 1, GL_COLOR, true, 'f'};
    if (!strcmp(format, "rg8snorm")) return {GL_RG8_SNORM, GL_RG, GL_UNSIGNED_BYTE, 2, 2, GL_COLOR, true, 'f'};
    if (!strcmp(format, "rgba8snorm")) return {GL_RGBA8_SNORM, GL_RGBA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, true, 'f'};
    if (!strcmp(format, "r8uint")) return {GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, 1, 1, GL_COLOR, true, 'u'};
    if (!strcmp(format, "rg8uint")) return {GL_RG8UI, GL_RG_INTEGER, GL_UNSIGNED_BYTE, 2, 2, GL_COLOR, true, 'u'};
    if (!strcmp(format, "rgba8uint")) return {GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, true, 'u'};
    if (!strcmp(format, "r16uint")) return {GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT, 1, 2, GL_COLOR, true, 'u'};
    if (!strcmp(format, "rg16uint")) return {GL_RG16UI, GL_RG_INTEGER, GL_UNSIGNED_SHORT, 2, 4, GL_COLOR, true, 'u'};
    if (!strcmp(format, "rgba16uint")) return {GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, 4, 8, GL_COLOR, true, 'u'};
    if (!strcmp(format, "r32uint")) return {GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 1, 4, GL_COLOR, true, 'u'};
    if (!strcmp(format, "rg32uint")) return {GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT, 2, 8, GL_COLOR, true, 'u'};
    if (!strcmp(format, "rgba32uint")) return {GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 4, 16, GL_COLOR, true, 'u'};
    if (!strcmp(format, "r8sint")) return {GL_R8I, GL_RED_INTEGER, GL_BYTE, 1, 1, GL_COLOR, true, 'i'};
    if (!strcmp(format, "rg8sint")) return {GL_RG8I, GL_RG_INTEGER, GL_BYTE, 2, 2, GL_COLOR, true, 'i'};
    if (!strcmp(format, "rgba8sint")) return {GL_RGBA8I, GL_RGBA_INTEGER, GL_BYTE, 4, 4, GL_COLOR, true, 'i'};
    if (!strcmp(format, "r16sint")) return {GL_R16I, GL_RED_INTEGER, GL_SHORT, 1, 2, GL_COLOR, true, 'i'};
    if (!strcmp(format, "rg16sint")) return {GL_RG16I, GL_RG_INTEGER, GL_SHORT, 2, 4, GL_COLOR, true, 'i'};
    if (!strcmp(format, "rgba16sint")) return {GL_RGBA16I, GL_RGBA_INTEGER, GL_SHORT, 4, 8, GL_COLOR, true, 'i'};
    if (!strcmp(format, "r32sint")) return {GL_R32I, GL_RED_INTEGER, GL_INT, 1, 4, GL_COLOR, true, 'i'};
    if (!strcmp(format, "rg32sint")) return {GL_RG32I, GL_RG_INTEGER, GL_INT, 2, 8, GL_COLOR, true, 'i'};
    if (!strcmp(format, "rgba32sint")) return {GL_RGBA32I, GL_RGBA_INTEGER, GL_INT, 4, 16, GL_COLOR, true, 'i'};
    if (!strcmp(format, "r16float")) return {GL_R16F, GL_RED, GL_FLOAT, 1, 2, GL_COLOR, true, 'f'};
    if (!strcmp(format, "rg16float")) return {GL_RG16F, GL_RG, GL_FLOAT, 2, 4, GL_COLOR, true, 'f'};
    if (!strcmp(format, "rgba16float")) return {GL_RGBA16F, GL_RGBA, GL_FLOAT, 4, 8, GL_COLOR, true, 'f'};
    if (!strcmp(format, "r32float")) return {GL_R32F, GL_RED, GL_FLOAT, 1, 4, GL_COLOR, true, 'f'};
    if (!strcmp(format, "rg32float")) return {GL_RG32F, GL_RG, GL_FLOAT, 2, 8, GL_COLOR, true, 'f'};
    if (!strcmp(format, "rgba32float")) return {GL_RGBA32F, GL_RGBA, GL_FLOAT, 4, 16, GL_COLOR, true, 'f'};
    if (!strcmp(format, "rgba8unorm-srgb")) return {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, true, 'f'};
    if (!strcmp(format, "bgra8unorm-srgb")) return {GL_SRGB8_ALPHA8, GL_BGRA, GL_UNSIGNED_BYTE, 4, 4, GL_COLOR, true, 'f'};
    if (!strcmp(format, "stencil8")) return {GL_STENCIL_INDEX8, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, 1, 1, GL_STENCIL, false, 'i'};
    if (!strcmp(format, "depth16unorm")) return {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, 1, 2, GL_DEPTH, false, 'f'};
    if (!strcmp(format, "depth24plus")) return {GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 1, 4, GL_DEPTH, false, 'f'};
    if (!strcmp(format, "depth24plus-stencil8")) return {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 2, 4, GL_DEPTH_STENCIL, false, 'x'};
    if (!strcmp(format, "depth32float")) return {GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, 4, GL_DEPTH, false, 'f'};
    return {};
}

int get_topology(const char * topology) {
    if (!strcmp(topology, "points")) return GL_POINTS;
    if (!strcmp(topology, "lines")) return GL_LINES;
    if (!strcmp(topology, "line_loop")) return GL_LINE_LOOP;
    if (!strcmp(topology, "line_strip")) return GL_LINE_STRIP;
    if (!strcmp(topology, "triangles")) return GL_TRIANGLES;
    if (!strcmp(topology, "triangle_strip")) return GL_TRIANGLE_STRIP;
    if (!strcmp(topology, "triangle_fan")) return GL_TRIANGLE_FAN;
    return -1;
}

int topology_converter(PyObject * arg, int * value) {
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

int count_mipmaps(int width, int height) {
    int size = width > height ? width : height;
    for (int i = 0; i < 32; ++i) {
        if (size <= (1 << i)) {
            return i;
        }
    }
    return 32;
}

void remove_dict_value(PyObject * dict, PyObject * obj) {
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

void * new_ref(void * obj) {
    Py_INCREF(obj);
    return obj;
}

PyObject * to_str(const unsigned char * ptr) {
    if (!ptr) {
        return PyUnicode_FromString("");
    }
    return PyUnicode_FromString((char *)ptr);
}

bool is_int_pair(PyObject * obj) {
    return (
        PySequence_Check(obj) && PySequence_Size(obj) == 2 &&
        PyLong_CheckExact(PySequence_GetItem(obj, 0)) &&
        PyLong_CheckExact(PySequence_GetItem(obj, 1))
    );
}

bool is_viewport(PyObject * obj) {
    return (
        PySequence_Check(obj) && PySequence_Size(obj) == 4 &&
        PyLong_CheckExact(PySequence_GetItem(obj, 0)) &&
        PyLong_CheckExact(PySequence_GetItem(obj, 1)) &&
        PyLong_CheckExact(PySequence_GetItem(obj, 2)) &&
        PyLong_CheckExact(PySequence_GetItem(obj, 3))
    );
}

IntPair to_int_pair(PyObject * obj) {
    IntPair res = {};
    res.x = PyLong_AsLong(PySequence_GetItem(obj, 0));
    res.y = PyLong_AsLong(PySequence_GetItem(obj, 1));
    return res;
}

Viewport to_viewport(PyObject * obj) {
    Viewport res = {};
    res.x = (short)PyLong_AsLong(PySequence_GetItem(obj, 0));
    res.y = (short)PyLong_AsLong(PySequence_GetItem(obj, 1));
    res.width = (short)PyLong_AsLong(PySequence_GetItem(obj, 2));
    res.height = (short)PyLong_AsLong(PySequence_GetItem(obj, 3));
    return res;
}

int max(int a, int b) {
    return a > b ? a : b;
}

void * load_method(PyObject * context, const char * method) {
    PyObject * res = PyObject_CallMethod(context, "load", "s", method);
    if (!res) {
        return NULL;
    }
    return PyLong_AsVoidPtr(res);
}

GLMethods load_gl(PyObject * context) {
    GLMethods res = {};

    #define check(name) if (!res.name) return {}
    #define load(name) res.name = (gl ## name ## Proc)load_method(context, "gl" # name); check(name)

    // GL_VERSION_1_0
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

    // GL_VERSION_1_1
    load(PolygonOffset);
    load(TexSubImage2D);
    load(BindTexture);
    load(DeleteTextures);
    load(GenTextures);

    // GL_VERSION_1_2
    load(TexImage3D);
    load(TexSubImage3D);

    // GL_VERSION_1_3
    load(ActiveTexture);

    // GL_VERSION_1_4
    load(BlendFuncSeparate);

    // GL_VERSION_1_5
    load(BindBuffer);
    load(DeleteBuffers);
    load(GenBuffers);
    load(BufferData);
    load(BufferSubData);
    load(UnmapBuffer);

    // GL_VERSION_2_0
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
    load(Uniform1iv);
    load(Uniform2iv);
    load(Uniform3iv);
    load(Uniform4iv);
    load(Uniform1fv);
    load(Uniform2fv);
    load(Uniform3fv);
    load(Uniform4fv);
    load(UniformMatrix2fv);
    load(UniformMatrix2x3fv);
    load(UniformMatrix2x4fv);
    load(UniformMatrix3x2fv);
    load(UniformMatrix3fv);
    load(UniformMatrix3x4fv);
    load(UniformMatrix4x2fv);
    load(UniformMatrix4x3fv);
    load(UniformMatrix4fv);
    load(VertexAttribPointer);

    // GL_VERSION_3_0
    load(ColorMaski);
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

    // GL_VERSION_3_1
    load(DrawArraysInstanced);
    load(DrawElementsInstanced);
    load(PrimitiveRestartIndex);
    load(GetUniformBlockIndex);
    load(GetActiveUniformBlockiv);
    load(GetActiveUniformBlockName);
    load(UniformBlockBinding);

    // GL_VERSION_3_3
    load(GenSamplers);
    load(DeleteSamplers);
    load(BindSampler);
    load(SamplerParameteri);
    load(SamplerParameterf);
    load(SamplerParameterfv);
    load(VertexAttribDivisor);

    #undef load
    #undef check
    return res;
}
