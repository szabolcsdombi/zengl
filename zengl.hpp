#include <Python.h>
#include <structmember.h>

const int MAX_ATTACHMENTS = 16;
const int MAX_BUFFER_BINDINGS = 32;
const int MAX_TEXTURE_BINDINGS = 128;
const int MAX_IMAGE_BINDINGS = 32;

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
#define GL_RGBA8 0x8058
#define GL_TEXTURE_WRAP_R 0x8072
#define GL_TEXTURE_MIN_LOD 0x813A
#define GL_TEXTURE_MAX_LOD 0x813B
#define GL_TEXTURE_BASE_LEVEL 0x813C
#define GL_TEXTURE_MAX_LEVEL 0x813D
#define GL_TEXTURE0 0x84C0
#define GL_TEXTURE_CUBE_MAP 0x8513
#define GL_DEPTH_COMPONENT16 0x81A5
#define GL_DEPTH_COMPONENT24 0x81A6
#define GL_TEXTURE_LOD_BIAS 0x8501
#define GL_TEXTURE_COMPARE_MODE 0x884C
#define GL_TEXTURE_COMPARE_FUNC 0x884D
#define GL_QUERY_RESULT 0x8866
#define GL_MAX_DRAW_BUFFERS 0x8824
#define GL_MAX_VERTEX_ATTRIBS 0x8869
#define GL_MAX_TEXTURE_IMAGE_UNITS 0x8872
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_VERTEX_SHADER 0x8B31
#define GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS 0x8B4D
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
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
#define GL_DRAW_INDIRECT_BUFFER 0x8F3F
#define GL_ALL_BARRIER_BITS 0xFFFFFFFF
#define GL_MAX_IMAGE_UNITS 0x8F38
#define GL_PRIMITIVE_RESTART_FIXED_INDEX 0x8D69
#define GL_COMPUTE_SHADER 0x91B9
#define GL_FRAMEBUFFER_DEFAULT_WIDTH 0x9310
#define GL_FRAMEBUFFER_DEFAULT_HEIGHT 0x9311
#define GL_UNIFORM 0x92E1
#define GL_UNIFORM_BLOCK 0x92E2
#define GL_PROGRAM_INPUT 0x92E3
#define GL_PROGRAM_OUTPUT 0x92E4
#define GL_SHADER_STORAGE_BLOCK 0x92E6
#define GL_ACTIVE_RESOURCES 0x92F5
#define GL_TYPE 0x92FA
#define GL_ARRAY_SIZE 0x92FB
#define GL_BUFFER_BINDING 0x9302
#define GL_BUFFER_DATA_SIZE 0x9303
#define GL_LOCATION 0x930E
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS 0x90DC
#define GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS 0x90DD
#define GL_MAX_SHADER_STORAGE_BLOCK_SIZE 0x90DE
#define GL_DYNAMIC_STORAGE_BIT 0x0100
#define GL_TEXTURE_MAX_ANISOTROPY 0x84FE

typedef void (GLAPI * glCullFaceProc)(unsigned mode);
typedef void (GLAPI * glDepthMaskProc)(unsigned char flag);
typedef void (GLAPI * glDisableProc)(unsigned cap);
typedef void (GLAPI * glEnableProc)(unsigned cap);
typedef void (GLAPI * glFlushProc)(void);
typedef void (GLAPI * glDepthFuncProc)(unsigned func);
typedef void (GLAPI * glReadPixelsProc)(int x, int y, int width, int height, unsigned format, unsigned type, void * pixels);
typedef unsigned (GLAPI * glGetErrorProc)(void);
typedef void (GLAPI * glGetIntegervProc)(unsigned pname, int * data);
typedef const unsigned char *(GLAPI * glGetStringProc)(unsigned name);
typedef void (GLAPI * glViewportProc)(int x, int y, int width, int height);
typedef void (GLAPI * glDeleteTexturesProc)(int n, const unsigned * textures);
typedef void (GLAPI * glGenQueriesProc)(int n, unsigned * ids);
typedef void (GLAPI * glDeleteQueriesProc)(int n, const unsigned * ids);
typedef void (GLAPI * glBeginQueryProc)(unsigned target, unsigned id);
typedef void (GLAPI * glEndQueryProc)(unsigned target);
typedef void (GLAPI * glGetQueryObjectuivProc)(unsigned id, unsigned pname, unsigned * params);
typedef void (GLAPI * glBindBufferProc)(unsigned target, unsigned buffer);
typedef void (GLAPI * glDeleteBuffersProc)(int n, const unsigned * buffers);
typedef void (GLAPI * glStencilOpSeparateProc)(unsigned face, unsigned sfail, unsigned dpfail, unsigned dppass);
typedef void (GLAPI * glStencilFuncSeparateProc)(unsigned face, unsigned func, int ref, unsigned mask);
typedef void (GLAPI * glStencilMaskSeparateProc)(unsigned face, unsigned mask);
typedef void (GLAPI * glAttachShaderProc)(unsigned program, unsigned shader);
typedef void (GLAPI * glCompileShaderProc)(unsigned shader);
typedef unsigned (GLAPI * glCreateProgramProc)(void);
typedef unsigned (GLAPI * glCreateShaderProc)(unsigned type);
typedef void (GLAPI * glDeleteProgramProc)(unsigned program);
typedef void (GLAPI * glDeleteShaderProc)(unsigned shader);
typedef void (GLAPI * glGetProgramivProc)(unsigned program, unsigned pname, int * params);
typedef void (GLAPI * glGetProgramInfoLogProc)(unsigned program, int bufSize, int * length, char * infoLog);
typedef void (GLAPI * glGetShaderivProc)(unsigned shader, unsigned pname, int * params);
typedef void (GLAPI * glGetShaderInfoLogProc)(unsigned shader, int bufSize, int * length, char * infoLog);
typedef void (GLAPI * glGetUniformivProc)(unsigned program, int location, int * params);
typedef void (GLAPI * glLinkProgramProc)(unsigned program);
typedef void (GLAPI * glShaderSourceProc)(unsigned shader, int count, const char * const * string, const int * length);
typedef void (GLAPI * glUseProgramProc)(unsigned program);
typedef void (GLAPI * glEnableiProc)(unsigned target, unsigned index);
typedef void (GLAPI * glDisableiProc)(unsigned target, unsigned index);
typedef void (GLAPI * glClearBufferivProc)(unsigned buffer, int drawbuffer, const int * value);
typedef void (GLAPI * glClearBufferuivProc)(unsigned buffer, int drawbuffer, const unsigned * value);
typedef void (GLAPI * glClearBufferfvProc)(unsigned buffer, int drawbuffer, const float * value);
typedef void (GLAPI * glClearBufferfiProc)(unsigned buffer, int drawbuffer, float depth, int stencil);
typedef void (GLAPI * glDeleteRenderbuffersProc)(int n, const unsigned * renderbuffers);
typedef void (GLAPI * glBindFramebufferProc)(unsigned target, unsigned framebuffer);
typedef void (GLAPI * glDeleteFramebuffersProc)(int n, const unsigned * framebuffers);
typedef void (GLAPI * glBlitFramebufferProc)(int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0, int dstX1, int dstY1, unsigned mask, unsigned filter);
typedef void (GLAPI * glBindVertexArrayProc)(unsigned array);
typedef void (GLAPI * glDeleteVertexArraysProc)(int n, const unsigned * arrays);
typedef void (GLAPI * glDrawArraysInstancedProc)(unsigned mode, int first, int count, int instancecount);
typedef void (GLAPI * glDrawElementsInstancedProc)(unsigned mode, int count, unsigned type, const void * indices, int instancecount);
typedef void * (GLAPI * glFenceSyncProc)(unsigned condition, unsigned flags);
typedef void (GLAPI * glDeleteSyncProc)(void * sync);
typedef unsigned (GLAPI * glClientWaitSyncProc)(void * sync, unsigned flags, unsigned long long timeout);
typedef void (GLAPI * glDeleteSamplersProc)(int count, const unsigned * samplers);
typedef void (GLAPI * glSamplerParameteriProc)(unsigned sampler, unsigned pname, int param);
typedef void (GLAPI * glSamplerParameterfProc)(unsigned sampler, unsigned pname, float param);
typedef void (GLAPI * glBlendEquationSeparateiProc)(unsigned buf, unsigned modeRGB, unsigned modeAlpha);
typedef void (GLAPI * glBlendFunciProc)(unsigned buf, unsigned src, unsigned dst);
typedef void (GLAPI * glBlendFuncSeparateiProc)(unsigned buf, unsigned srcRGB, unsigned dstRGB, unsigned srcAlpha, unsigned dstAlpha);
typedef void (GLAPI * glProgramUniform1iProc)(unsigned program, int location, int v0);
typedef void (GLAPI * glProgramUniform1ivProc)(unsigned program, int location, int count, const int * value);
typedef void (GLAPI * glProgramUniform1fvProc)(unsigned program, int location, int count, const float * value);
typedef void (GLAPI * glProgramUniform1uivProc)(unsigned program, int location, int count, const unsigned * value);
typedef void (GLAPI * glProgramUniform2ivProc)(unsigned program, int location, int count, const int * value);
typedef void (GLAPI * glProgramUniform2fvProc)(unsigned program, int location, int count, const float * value);
typedef void (GLAPI * glProgramUniform2uivProc)(unsigned program, int location, int count, const unsigned * value);
typedef void (GLAPI * glProgramUniform3ivProc)(unsigned program, int location, int count, const int * value);
typedef void (GLAPI * glProgramUniform3fvProc)(unsigned program, int location, int count, const float * value);
typedef void (GLAPI * glProgramUniform3uivProc)(unsigned program, int location, int count, const unsigned * value);
typedef void (GLAPI * glProgramUniform4ivProc)(unsigned program, int location, int count, const int * value);
typedef void (GLAPI * glProgramUniform4fvProc)(unsigned program, int location, int count, const float * value);
typedef void (GLAPI * glProgramUniform4uivProc)(unsigned program, int location, int count, const unsigned * value);
typedef void (GLAPI * glProgramUniformMatrix2fvProc)(unsigned program, int location, int count, unsigned char transpose, const float * value);
typedef void (GLAPI * glProgramUniformMatrix3fvProc)(unsigned program, int location, int count, unsigned char transpose, const float * value);
typedef void (GLAPI * glProgramUniformMatrix4fvProc)(unsigned program, int location, int count, unsigned char transpose, const float * value);
typedef void (GLAPI * glProgramUniformMatrix2x3fvProc)(unsigned program, int location, int count, unsigned char transpose, const float * value);
typedef void (GLAPI * glProgramUniformMatrix3x2fvProc)(unsigned program, int location, int count, unsigned char transpose, const float * value);
typedef void (GLAPI * glProgramUniformMatrix2x4fvProc)(unsigned program, int location, int count, unsigned char transpose, const float * value);
typedef void (GLAPI * glProgramUniformMatrix4x2fvProc)(unsigned program, int location, int count, unsigned char transpose, const float * value);
typedef void (GLAPI * glProgramUniformMatrix3x4fvProc)(unsigned program, int location, int count, unsigned char transpose, const float * value);
typedef void (GLAPI * glProgramUniformMatrix4x3fvProc)(unsigned program, int location, int count, unsigned char transpose, const float * value);
typedef void (GLAPI * glMemoryBarrierProc)(unsigned barriers);
typedef void (GLAPI * glDispatchComputeProc)(unsigned num_groups_x, unsigned num_groups_y, unsigned num_groups_z);
typedef void (GLAPI * glMultiDrawArraysIndirectProc)(unsigned mode, const void * indirect, int drawcount, int stride);
typedef void (GLAPI * glMultiDrawElementsIndirectProc)(unsigned mode, unsigned type, const void * indirect, int drawcount, int stride);
typedef void (GLAPI * glGetProgramInterfaceivProc)(unsigned program, unsigned programInterface, unsigned pname, int * params);
typedef void (GLAPI * glGetProgramResourceNameProc)(unsigned program, unsigned programInterface, unsigned index, int bufSize, int * length, char * name);
typedef void (GLAPI * glGetProgramResourceivProc)(unsigned program, unsigned programInterface, unsigned index, int propCount, const unsigned * props, int count, int * length, int * params);
typedef void (GLAPI * glBindBuffersRangeProc)(unsigned target, unsigned first, int count, const unsigned * buffers, const sizeiptr * offsets, const sizeiptr * sizes);
typedef void (GLAPI * glBindTexturesProc)(unsigned first, int count, const unsigned * textures);
typedef void (GLAPI * glBindSamplersProc)(unsigned first, int count, const unsigned * samplers);
typedef void (GLAPI * glBindImageTexturesProc)(unsigned first, int count, const unsigned * textures);
typedef void (GLAPI * glCreateBuffersProc)(int n, unsigned * buffers);
typedef void (GLAPI * glNamedBufferStorageProc)(unsigned buffer, sizeiptr size, const void * data, unsigned flags);
typedef void (GLAPI * glNamedBufferSubDataProc)(unsigned buffer, sizeiptr offset, sizeiptr size, const void * data);
typedef void *(GLAPI * glMapNamedBufferRangeProc)(unsigned buffer, sizeiptr offset, sizeiptr length, unsigned access);
typedef unsigned char (GLAPI * glUnmapNamedBufferProc)(unsigned buffer);
typedef void (GLAPI * glGetNamedBufferSubDataProc)(unsigned buffer, sizeiptr offset, sizeiptr size, void * data);
typedef void (GLAPI * glCreateFramebuffersProc)(int n, unsigned * framebuffers);
typedef void (GLAPI * glNamedFramebufferRenderbufferProc)(unsigned framebuffer, unsigned attachment, unsigned renderbuffertarget, unsigned renderbuffer);
typedef void (GLAPI * glNamedFramebufferParameteriProc)(unsigned framebuffer, unsigned pname, int param);
typedef void (GLAPI * glNamedFramebufferTextureProc)(unsigned framebuffer, unsigned attachment, unsigned texture, int level);
typedef void (GLAPI * glNamedFramebufferTextureLayerProc)(unsigned framebuffer, unsigned attachment, unsigned texture, int level, int layer);
typedef void (GLAPI * glNamedFramebufferDrawBuffersProc)(unsigned framebuffer, int n, const unsigned * bufs);
typedef void (GLAPI * glNamedFramebufferReadBufferProc)(unsigned framebuffer, unsigned src);
typedef void (GLAPI * glCreateRenderbuffersProc)(int n, unsigned * renderbuffers);
typedef void (GLAPI * glNamedRenderbufferStorageMultisampleProc)(unsigned renderbuffer, int samples, unsigned internalformat, int width, int height);
typedef void (GLAPI * glCreateTexturesProc)(unsigned target, int n, unsigned * textures);
typedef void (GLAPI * glTextureStorage2DProc)(unsigned texture, int levels, unsigned internalformat, int width, int height);
typedef void (GLAPI * glTextureStorage3DProc)(unsigned texture, int levels, unsigned internalformat, int width, int height, int depth);
typedef void (GLAPI * glTextureSubImage2DProc)(unsigned texture, int level, int xoffset, int yoffset, int width, int height, unsigned format, unsigned type, const void * pixels);
typedef void (GLAPI * glTextureSubImage3DProc)(unsigned texture, int level, int xoffset, int yoffset, int zoffset, int width, int height, int depth, unsigned format, unsigned type, const void * pixels);
typedef void (GLAPI * glTextureParameteriProc)(unsigned texture, unsigned pname, int param);
typedef void (GLAPI * glGenerateTextureMipmapProc)(unsigned texture);
typedef void (GLAPI * glCreateVertexArraysProc)(int n, unsigned * arrays);
typedef void (GLAPI * glEnableVertexArrayAttribProc)(unsigned vaobj, unsigned index);
typedef void (GLAPI * glVertexArrayElementBufferProc)(unsigned vaobj, unsigned buffer);
typedef void (GLAPI * glVertexArrayVertexBufferProc)(unsigned vaobj, unsigned bindingindex, unsigned buffer, sizeiptr offset, int stride);
typedef void (GLAPI * glVertexArrayAttribFormatProc)(unsigned vaobj, unsigned attribindex, int size, unsigned type, unsigned char normalized, unsigned relativeoffset);
typedef void (GLAPI * glVertexArrayAttribIFormatProc)(unsigned vaobj, unsigned attribindex, int size, unsigned type, unsigned relativeoffset);
typedef void (GLAPI * glVertexArrayBindingDivisorProc)(unsigned vaobj, unsigned bindingindex, unsigned divisor);
typedef void (GLAPI * glCreateSamplersProc)(int n, unsigned * samplers);

struct GLMethods {
    glCullFaceProc CullFace;
    glDepthMaskProc DepthMask;
    glDisableProc Disable;
    glEnableProc Enable;
    glFlushProc Flush;
    glDepthFuncProc DepthFunc;
    glReadPixelsProc ReadPixels;
    glGetErrorProc GetError;
    glGetIntegervProc GetIntegerv;
    glGetStringProc GetString;
    glViewportProc Viewport;
    glDeleteTexturesProc DeleteTextures;
    glGenQueriesProc GenQueries;
    glDeleteQueriesProc DeleteQueries;
    glBeginQueryProc BeginQuery;
    glEndQueryProc EndQuery;
    glGetQueryObjectuivProc GetQueryObjectuiv;
    glBindBufferProc BindBuffer;
    glDeleteBuffersProc DeleteBuffers;
    glStencilOpSeparateProc StencilOpSeparate;
    glStencilFuncSeparateProc StencilFuncSeparate;
    glStencilMaskSeparateProc StencilMaskSeparate;
    glAttachShaderProc AttachShader;
    glCompileShaderProc CompileShader;
    glCreateProgramProc CreateProgram;
    glCreateShaderProc CreateShader;
    glDeleteProgramProc DeleteProgram;
    glDeleteShaderProc DeleteShader;
    glGetProgramivProc GetProgramiv;
    glGetProgramInfoLogProc GetProgramInfoLog;
    glGetShaderivProc GetShaderiv;
    glGetShaderInfoLogProc GetShaderInfoLog;
    glGetUniformivProc GetUniformiv;
    glLinkProgramProc LinkProgram;
    glShaderSourceProc ShaderSource;
    glUseProgramProc UseProgram;
    glEnableiProc Enablei;
    glDisableiProc Disablei;
    glClearBufferivProc ClearBufferiv;
    glClearBufferuivProc ClearBufferuiv;
    glClearBufferfvProc ClearBufferfv;
    glClearBufferfiProc ClearBufferfi;
    glDeleteRenderbuffersProc DeleteRenderbuffers;
    glBindFramebufferProc BindFramebuffer;
    glDeleteFramebuffersProc DeleteFramebuffers;
    glBlitFramebufferProc BlitFramebuffer;
    glBindVertexArrayProc BindVertexArray;
    glDeleteVertexArraysProc DeleteVertexArrays;
    glDrawArraysInstancedProc DrawArraysInstanced;
    glDrawElementsInstancedProc DrawElementsInstanced;
    glFenceSyncProc FenceSync;
    glDeleteSyncProc DeleteSync;
    glClientWaitSyncProc ClientWaitSync;
    glDeleteSamplersProc DeleteSamplers;
    glSamplerParameteriProc SamplerParameteri;
    glSamplerParameterfProc SamplerParameterf;
    glBlendEquationSeparateiProc BlendEquationSeparatei;
    glBlendFunciProc BlendFunci;
    glBlendFuncSeparateiProc BlendFuncSeparatei;
    glProgramUniform1iProc ProgramUniform1i;
    glProgramUniform1ivProc ProgramUniform1iv;
    glProgramUniform1fvProc ProgramUniform1fv;
    glProgramUniform1uivProc ProgramUniform1uiv;
    glProgramUniform2ivProc ProgramUniform2iv;
    glProgramUniform2fvProc ProgramUniform2fv;
    glProgramUniform2uivProc ProgramUniform2uiv;
    glProgramUniform3ivProc ProgramUniform3iv;
    glProgramUniform3fvProc ProgramUniform3fv;
    glProgramUniform3uivProc ProgramUniform3uiv;
    glProgramUniform4ivProc ProgramUniform4iv;
    glProgramUniform4fvProc ProgramUniform4fv;
    glProgramUniform4uivProc ProgramUniform4uiv;
    glProgramUniformMatrix2fvProc ProgramUniformMatrix2fv;
    glProgramUniformMatrix3fvProc ProgramUniformMatrix3fv;
    glProgramUniformMatrix4fvProc ProgramUniformMatrix4fv;
    glProgramUniformMatrix2x3fvProc ProgramUniformMatrix2x3fv;
    glProgramUniformMatrix3x2fvProc ProgramUniformMatrix3x2fv;
    glProgramUniformMatrix2x4fvProc ProgramUniformMatrix2x4fv;
    glProgramUniformMatrix4x2fvProc ProgramUniformMatrix4x2fv;
    glProgramUniformMatrix3x4fvProc ProgramUniformMatrix3x4fv;
    glProgramUniformMatrix4x3fvProc ProgramUniformMatrix4x3fv;
    glMemoryBarrierProc MemoryBarrier;
    glDispatchComputeProc DispatchCompute;
    glMultiDrawArraysIndirectProc MultiDrawArraysIndirect;
    glMultiDrawElementsIndirectProc MultiDrawElementsIndirect;
    glGetProgramInterfaceivProc GetProgramInterfaceiv;
    glGetProgramResourceNameProc GetProgramResourceName;
    glGetProgramResourceivProc GetProgramResourceiv;
    glBindBuffersRangeProc BindBuffersRange;
    glBindTexturesProc BindTextures;
    glBindSamplersProc BindSamplers;
    glBindImageTexturesProc BindImageTextures;
    glCreateBuffersProc CreateBuffers;
    glNamedBufferStorageProc NamedBufferStorage;
    glNamedBufferSubDataProc NamedBufferSubData;
    glMapNamedBufferRangeProc MapNamedBufferRange;
    glUnmapNamedBufferProc UnmapNamedBuffer;
    glGetNamedBufferSubDataProc GetNamedBufferSubData;
    glCreateFramebuffersProc CreateFramebuffers;
    glNamedFramebufferRenderbufferProc NamedFramebufferRenderbuffer;
    glNamedFramebufferParameteriProc NamedFramebufferParameteri;
    glNamedFramebufferTextureProc NamedFramebufferTexture;
    glNamedFramebufferTextureLayerProc NamedFramebufferTextureLayer;
    glNamedFramebufferDrawBuffersProc NamedFramebufferDrawBuffers;
    glNamedFramebufferReadBufferProc NamedFramebufferReadBuffer;
    glCreateRenderbuffersProc CreateRenderbuffers;
    glNamedRenderbufferStorageMultisampleProc NamedRenderbufferStorageMultisample;
    glCreateTexturesProc CreateTextures;
    glTextureStorage2DProc TextureStorage2D;
    glTextureStorage3DProc TextureStorage3D;
    glTextureSubImage2DProc TextureSubImage2D;
    glTextureSubImage3DProc TextureSubImage3D;
    glTextureParameteriProc TextureParameteri;
    glGenerateTextureMipmapProc GenerateTextureMipmap;
    glCreateVertexArraysProc CreateVertexArrays;
    glEnableVertexArrayAttribProc EnableVertexArrayAttrib;
    glVertexArrayElementBufferProc VertexArrayElementBuffer;
    glVertexArrayVertexBufferProc VertexArrayVertexBuffer;
    glVertexArrayAttribFormatProc VertexArrayAttribFormat;
    glVertexArrayAttribIFormatProc VertexArrayAttribIFormat;
    glVertexArrayBindingDivisorProc VertexArrayBindingDivisor;
    glCreateSamplersProc CreateSamplers;
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

struct DynamicState {
    int vertex_count;
    int instance_count;
    int indirect_count;
    int first_vertex;
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

int min(int a, int b) {
    return a < b ? a : b;
}

bool is_uniform_variable(int t) {
    return (0x1404 <= t && t <= 0x8B5C) || (0x8B65 <= t && t <= 0x8B6A) || (0x8DC6 <= t && t <= 0x8DC8) || (0x8F46 <= t && t <= 0x8FFE);
}

bool is_uniform_sampler(int t) {
    return (0x8B5D <= t && t <= 0x8B64) || (0x8DC0 <= t && t <= 0x8DC5) || (0x8DC9 <= t && t <= 0x8DD8) || (0x900C <= t && t <= 0x900F) || (0x9108 <= t && t <= 0x910D);
}

bool is_uniform_image(int t) {
    return 0x904C <= t && t <= 0x906C;
}

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
    res.x = PyLong_AsLong(PySequence_GetItem(obj, 0));
    res.y = PyLong_AsLong(PySequence_GetItem(obj, 1));
    res.width = PyLong_AsLong(PySequence_GetItem(obj, 2));
    res.height = PyLong_AsLong(PySequence_GetItem(obj, 3));
    return res;
}

int max(int a, int b) {
    return a > b ? a : b;
}

void * load_opengl_function(PyObject * loader, const char * method) {
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

GLMethods load_gl(PyObject * loader) {
    GLMethods res = {};
    PyObject * missing = PyList_New(0);

    #define check(name) if (!res.name) { if (PyErr_Occurred()) return {}; PyList_Append(missing, PyUnicode_FromString("gl" # name)); }
    #define load(name) res.name = (gl ## name ## Proc)load_opengl_function(loader, "gl" # name); check(name)

    load(CullFace);
    load(DepthMask);
    load(Disable);
    load(Enable);
    load(Flush);
    load(DepthFunc);
    load(ReadPixels);
    load(GetError);
    load(GetIntegerv);
    load(GetString);
    load(Viewport);
    load(DeleteTextures);
    load(GenQueries);
    load(DeleteQueries);
    load(BeginQuery);
    load(EndQuery);
    load(GetQueryObjectuiv);
    load(BindBuffer);
    load(DeleteBuffers);
    load(StencilOpSeparate);
    load(StencilFuncSeparate);
    load(StencilMaskSeparate);
    load(AttachShader);
    load(CompileShader);
    load(CreateProgram);
    load(CreateShader);
    load(DeleteProgram);
    load(DeleteShader);
    load(GetProgramiv);
    load(GetProgramInfoLog);
    load(GetShaderiv);
    load(GetShaderInfoLog);
    load(GetUniformiv);
    load(LinkProgram);
    load(ShaderSource);
    load(UseProgram);
    load(Enablei);
    load(Disablei);
    load(ClearBufferiv);
    load(ClearBufferuiv);
    load(ClearBufferfv);
    load(ClearBufferfi);
    load(DeleteRenderbuffers);
    load(BindFramebuffer);
    load(DeleteFramebuffers);
    load(BlitFramebuffer);
    load(BindVertexArray);
    load(DeleteVertexArrays);
    load(DrawArraysInstanced);
    load(DrawElementsInstanced);
    load(FenceSync);
    load(DeleteSync);
    load(ClientWaitSync);
    load(DeleteSamplers);
    load(SamplerParameteri);
    load(SamplerParameterf);
    load(BlendEquationSeparatei);
    load(BlendFunci);
    load(BlendFuncSeparatei);
    load(ProgramUniform1i);
    load(ProgramUniform1iv);
    load(ProgramUniform1fv);
    load(ProgramUniform1uiv);
    load(ProgramUniform2iv);
    load(ProgramUniform2fv);
    load(ProgramUniform2uiv);
    load(ProgramUniform3iv);
    load(ProgramUniform3fv);
    load(ProgramUniform3uiv);
    load(ProgramUniform4iv);
    load(ProgramUniform4fv);
    load(ProgramUniform4uiv);
    load(ProgramUniformMatrix2fv);
    load(ProgramUniformMatrix3fv);
    load(ProgramUniformMatrix4fv);
    load(ProgramUniformMatrix2x3fv);
    load(ProgramUniformMatrix3x2fv);
    load(ProgramUniformMatrix2x4fv);
    load(ProgramUniformMatrix4x2fv);
    load(ProgramUniformMatrix3x4fv);
    load(ProgramUniformMatrix4x3fv);
    load(MemoryBarrier);
    load(DispatchCompute);
    load(MultiDrawArraysIndirect);
    load(MultiDrawElementsIndirect);
    load(GetProgramInterfaceiv);
    load(GetProgramResourceName);
    load(GetProgramResourceiv);
    load(BindBuffersRange);
    load(BindTextures);
    load(BindSamplers);
    load(BindImageTextures);
    load(CreateBuffers);
    load(NamedBufferStorage);
    load(NamedBufferSubData);
    load(MapNamedBufferRange);
    load(UnmapNamedBuffer);
    load(GetNamedBufferSubData);
    load(CreateFramebuffers);
    load(NamedFramebufferRenderbuffer);
    load(NamedFramebufferParameteri);
    load(NamedFramebufferTexture);
    load(NamedFramebufferTextureLayer);
    load(NamedFramebufferDrawBuffers);
    load(NamedFramebufferReadBuffer);
    load(CreateRenderbuffers);
    load(NamedRenderbufferStorageMultisample);
    load(CreateTextures);
    load(TextureStorage2D);
    load(TextureStorage3D);
    load(TextureSubImage2D);
    load(TextureSubImage3D);
    load(TextureParameteri);
    load(GenerateTextureMipmap);
    load(CreateVertexArrays);
    load(EnableVertexArrayAttrib);
    load(VertexArrayElementBuffer);
    load(VertexArrayVertexBuffer);
    load(VertexArrayAttribFormat);
    load(VertexArrayAttribIFormat);
    load(VertexArrayBindingDivisor);
    load(CreateSamplers);

    #undef load
    #undef check

    if (PyList_Size(missing)) {
        PyErr_Format(PyExc_RuntimeError, "cannot load opengl %R", missing);
        return {};
    }

    Py_DECREF(missing);
    return res;
}
