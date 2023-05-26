#include <Python.h>
#include <structmember.h>

#define MAX_ATTACHMENTS 16
#define MAX_BUFFER_BINDINGS 16
#define MAX_SAMPLER_BINDINGS 64

#ifdef _WIN32
#define GL __stdcall *
#else
#define GL *
#endif

typedef Py_ssize_t intptr;

#define GL_COLOR_BUFFER_BIT 0x4000
#define GL_FRONT 0x0404
#define GL_BACK 0x0405
#define GL_CULL_FACE 0x0B44
#define GL_DEPTH_TEST 0x0B71
#define GL_STENCIL_TEST 0x0B90
#define GL_BLEND 0x0BE2
#define GL_TEXTURE_2D 0x0DE1
#define GL_UNSIGNED_SHORT 0x1403
#define GL_UNSIGNED_INT 0x1405
#define GL_DEPTH 0x1801
#define GL_STENCIL 0x1802
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_NEAREST 0x2600
#define GL_LINEAR 0x2601
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_WRAP_S 0x2802
#define GL_TEXTURE_WRAP_T 0x2803
#define GL_TEXTURE_WRAP_R 0x8072
#define GL_TEXTURE_MIN_LOD 0x813A
#define GL_TEXTURE_MAX_LOD 0x813B
#define GL_TEXTURE0 0x84C0
#define GL_TEXTURE_CUBE_MAP 0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X 0x8515
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
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#define GL_DEPTH_STENCIL_ATTACHMENT 0x821A
#define GL_DEPTH_STENCIL 0x84F9
#define GL_READ_FRAMEBUFFER 0x8CA8
#define GL_DRAW_FRAMEBUFFER 0x8CA9
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_DEPTH_ATTACHMENT 0x8D00
#define GL_STENCIL_ATTACHMENT 0x8D20
#define GL_FRAMEBUFFER 0x8D40
#define GL_RENDERBUFFER 0x8D41
#define GL_MAX_SAMPLES 0x8D57
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#define GL_MAP_READ_BIT 0x0001
#define GL_MAP_WRITE_BIT 0x0002
#define GL_MAP_INVALIDATE_RANGE_BIT 0x0004
#define GL_UNIFORM_BUFFER 0x8A11
#define GL_MAX_COMBINED_UNIFORM_BLOCKS 0x8A2E
#define GL_MAX_UNIFORM_BUFFER_BINDINGS 0x8A2F
#define GL_MAX_UNIFORM_BLOCK_SIZE 0x8A30
#define GL_ACTIVE_UNIFORM_BLOCKS 0x8A36
#define GL_UNIFORM_BLOCK_DATA_SIZE 0x8A40
#define GL_PROGRAM_POINT_SIZE 0x8642
#define GL_TEXTURE_CUBE_MAP_SEAMLESS 0x884F
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#define GL_SYNC_FLUSH_COMMANDS_BIT 0x0001
#define GL_TIME_ELAPSED 0x88BF
#define GL_PRIMITIVE_RESTART_FIXED_INDEX 0x8D69
#define GL_TEXTURE_MAX_ANISOTROPY 0x84FE

typedef struct GLMethods {
    void (GL CullFace)(int);
    void (GL Clear)(int);
    void (GL TexParameteri)(int, int, int);
    void (GL TexImage2D)(int, int, int, int, int, int, int, int, const void *);
    void (GL DepthMask)(int);
    void (GL Disable)(int);
    void (GL Enable)(int);
    void (GL Flush)();
    void (GL DepthFunc)(int);
    void (GL ReadBuffer)(int);
    void (GL ReadPixels)(int, int, int, int, int, int, void *);
    int (GL GetError)();
    void (GL GetIntegerv)(int, int *);
    const char * (GL GetString)(int);
    void (GL Viewport)(int, int, int, int);
    void (GL TexSubImage2D)(int, int, int, int, int, int, int, int, const void *);
    void (GL BindTexture)(int, int);
    void (GL DeleteTextures)(int, const int *);
    void (GL GenTextures)(int, int *);
    void (GL TexImage3D)(int, int, int, int, int, int, int, int, int, const void *);
    void (GL TexSubImage3D)(int, int, int, int, int, int, int, int, int, int, const void *);
    void (GL ActiveTexture)(int);
    void (GL BlendFuncSeparate)(int, int, int, int);
    void (GL GenQueries)(int, int *);
    void (GL BeginQuery)(int, int);
    void (GL EndQuery)(int);
    void (GL GetQueryObjectuiv)(int, int, void *);
    void (GL BindBuffer)(int, int);
    void (GL DeleteBuffers)(int, const int *);
    void (GL GenBuffers)(int, int *);
    void (GL BufferData)(int, intptr, const void *, int);
    void (GL BufferSubData)(int, intptr, intptr, const void *);
    int (GL UnmapBuffer)(int);
    void (GL BlendEquationSeparate)(int, int);
    void (GL DrawBuffers)(int, const int *);
    void (GL StencilOpSeparate)(int, int, int, int);
    void (GL StencilFuncSeparate)(int, int, int, int);
    void (GL StencilMaskSeparate)(int, int);
    void (GL AttachShader)(int, int);
    void (GL CompileShader)(int);
    int (GL CreateProgram)();
    int (GL CreateShader)(int);
    void (GL DeleteProgram)(int);
    void (GL DeleteShader)(int);
    void (GL EnableVertexAttribArray)(int);
    void (GL GetActiveAttrib)(int, int, int, int *, int *, int *, char *);
    void (GL GetActiveUniform)(int, int, int, int *, int *, int *, char *);
    int (GL GetAttribLocation)(int, const char *);
    void (GL GetProgramiv)(int, int, int *);
    void (GL GetProgramInfoLog)(int, int, int *, char *);
    void (GL GetShaderiv)(int, int, int *);
    void (GL GetShaderInfoLog)(int, int, int *, char *);
    int (GL GetUniformLocation)(int, const char *);
    void (GL LinkProgram)(int);
    void (GL ShaderSource)(int, int, const void *, const int *);
    void (GL UseProgram)(int);
    void (GL Uniform1i)(int, int);
    void (GL Uniform1fv)(int, int, const void *);
    void (GL Uniform2fv)(int, int, const void *);
    void (GL Uniform3fv)(int, int, const void *);
    void (GL Uniform4fv)(int, int, const void *);
    void (GL Uniform1iv)(int, int, const void *);
    void (GL Uniform2iv)(int, int, const void *);
    void (GL Uniform3iv)(int, int, const void *);
    void (GL Uniform4iv)(int, int, const void *);
    void (GL UniformMatrix2fv)(int, int, int, const void *);
    void (GL UniformMatrix3fv)(int, int, int, const void *);
    void (GL UniformMatrix4fv)(int, int, int, const void *);
    void (GL VertexAttribPointer)(int, int, int, int, int, intptr);
    void (GL UniformMatrix2x3fv)(int, int, int, const void *);
    void (GL UniformMatrix3x2fv)(int, int, int, const void *);
    void (GL UniformMatrix2x4fv)(int, int, int, const void *);
    void (GL UniformMatrix4x2fv)(int, int, int, const void *);
    void (GL UniformMatrix3x4fv)(int, int, int, const void *);
    void (GL UniformMatrix4x3fv)(int, int, int, const void *);
    void (GL BindBufferRange)(int, int, int, intptr, intptr);
    void (GL VertexAttribIPointer)(int, int, int, int, intptr);
    void (GL Uniform1uiv)(int, int, const void *);
    void (GL Uniform2uiv)(int, int, const void *);
    void (GL Uniform3uiv)(int, int, const void *);
    void (GL Uniform4uiv)(int, int, const void *);
    void (GL ClearBufferiv)(int, int, const void *);
    void (GL ClearBufferuiv)(int, int, const void *);
    void (GL ClearBufferfv)(int, int, const void *);
    void (GL ClearBufferfi)(int, int, float, int);
    void (GL BindRenderbuffer)(int, int);
    void (GL DeleteRenderbuffers)(int, const int *);
    void (GL GenRenderbuffers)(int, int *);
    void (GL BindFramebuffer)(int, int);
    void (GL DeleteFramebuffers)(int, const int *);
    void (GL GenFramebuffers)(int, int *);
    void (GL FramebufferTexture2D)(int, int, int, int, int);
    void (GL FramebufferRenderbuffer)(int, int, int, int);
    void (GL GenerateMipmap)(int);
    void (GL BlitFramebuffer)(int, int, int, int, int, int, int, int, int, int);
    void (GL RenderbufferStorageMultisample)(int, int, int, int, int);
    void (GL FramebufferTextureLayer)(int, int, int, int, int);
    void * (GL MapBufferRange)(int, intptr, intptr, int);
    void (GL BindVertexArray)(int);
    void (GL DeleteVertexArrays)(int, const int *);
    void (GL GenVertexArrays)(int, int *);
    void (GL DrawArraysInstanced)(int, int, int, int);
    void (GL DrawElementsInstanced)(int, int, int, intptr, int);
    int (GL GetUniformBlockIndex)(int, const char *);
    void (GL GetActiveUniformBlockiv)(int, int, int, int *);
    void (GL GetActiveUniformBlockName)(int, int, int, int *, char *);
    void (GL UniformBlockBinding)(int, int, int);
    void * (GL FenceSync)(int, int);
    void (GL DeleteSync)(void *);
    int (GL ClientWaitSync)(void *, int, long long);
    void (GL GenSamplers)(int, int *);
    void (GL DeleteSamplers)(int, const int *);
    void (GL BindSampler)(int, int);
    void (GL SamplerParameteri)(int, int, int);
    void (GL SamplerParameterf)(int, int, float);
    void (GL VertexAttribDivisor)(int, int);
} GLMethods;

typedef void (GL UniformSetter)(int, int, const void *);
typedef void (GL UniformMatrixSetter)(int, int, int, const void *);

static int uniform_setter_offset[] = {
    offsetof(GLMethods, Uniform1iv),
    offsetof(GLMethods, Uniform2iv),
    offsetof(GLMethods, Uniform3iv),
    offsetof(GLMethods, Uniform4iv),
    offsetof(GLMethods, Uniform1iv),
    offsetof(GLMethods, Uniform2iv),
    offsetof(GLMethods, Uniform3iv),
    offsetof(GLMethods, Uniform4iv),
    offsetof(GLMethods, Uniform1uiv),
    offsetof(GLMethods, Uniform2uiv),
    offsetof(GLMethods, Uniform3uiv),
    offsetof(GLMethods, Uniform4uiv),
    offsetof(GLMethods, Uniform1fv),
    offsetof(GLMethods, Uniform2fv),
    offsetof(GLMethods, Uniform3fv),
    offsetof(GLMethods, Uniform4fv),
    offsetof(GLMethods, UniformMatrix2fv),
    offsetof(GLMethods, UniformMatrix2x3fv),
    offsetof(GLMethods, UniformMatrix2x4fv),
    offsetof(GLMethods, UniformMatrix3x2fv),
    offsetof(GLMethods, UniformMatrix3fv),
    offsetof(GLMethods, UniformMatrix3x4fv),
    offsetof(GLMethods, UniformMatrix4x2fv),
    offsetof(GLMethods, UniformMatrix4x3fv),
    offsetof(GLMethods, UniformMatrix4fv),
};

typedef struct VertexFormat {
    int type;
    int size;
    int normalize;
    int integer;
} VertexFormat;

typedef struct ImageFormat {
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

typedef struct UniformBinding {
    int function;
    int location;
    int count;
    int offset;
} UniformBinding;

typedef struct UniformHeader {
    int count;
    UniformBinding binding[1];
} UniformHeader;

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
    unsigned clear_uints[4];
} ClearValue;

typedef struct IntPair {
    int x;
    int y;
} IntPair;

static int to_int(PyObject * obj) {
    return (int)PyLong_AsLong(obj);
}

static unsigned to_uint(PyObject * obj) {
    return (unsigned)PyLong_AsUnsignedLong(obj);
}

static float to_float(PyObject * obj) {
    return (float)PyFloat_AsDouble(obj);
}

static int least_one(int value) {
    return value > 1 ? value : 1;
}

static int get_vertex_format(PyObject * helper, PyObject * name, VertexFormat * res) {
    PyObject * lookup = PyObject_GetAttrString(helper, "VERTEX_FORMAT");
    PyObject * tup = PyDict_GetItem(lookup, name);
    Py_DECREF(lookup);
    if (!tup) {
        return 0;
    }
    res->type = to_int(PyTuple_GetItem(tup, 0));
    res->size = to_int(PyTuple_GetItem(tup, 1));
    res->normalize = to_int(PyTuple_GetItem(tup, 2));
    res->integer = to_int(PyTuple_GetItem(tup, 3));
    return 1;
}

static int get_image_format(PyObject * helper, PyObject * name, ImageFormat * res) {
    PyObject * lookup = PyObject_GetAttrString(helper, "IMAGE_FORMAT");
    PyObject * tup = PyDict_GetItem(lookup, name);
    Py_DECREF(lookup);
    if (!tup) {
        return 0;
    }
    res->internal_format = to_int(PyTuple_GetItem(tup, 0));
    res->format = to_int(PyTuple_GetItem(tup, 1));
    res->type = to_int(PyTuple_GetItem(tup, 2));
    res->buffer = to_int(PyTuple_GetItem(tup, 3));
    res->components = to_int(PyTuple_GetItem(tup, 4));
    res->pixel_size = to_int(PyTuple_GetItem(tup, 5));
    res->color = to_int(PyTuple_GetItem(tup, 6));
    res->flags = to_int(PyTuple_GetItem(tup, 7));
    res->clear_type = PyUnicode_AsUTF8(PyTuple_GetItem(tup, 8))[0];
    return 1;
}

static int get_topology(PyObject * helper, PyObject * name, int * res) {
    PyObject * lookup = PyObject_GetAttrString(helper, "TOPOLOGY");
    PyObject * value = PyDict_GetItem(lookup, name);
    Py_DECREF(lookup);
    if (!value) {
        return 0;
    }
    *res = to_int(value);
    return 1;
}

static int count_mipmaps(int width, int height) {
    int size = width > height ? width : height;
    int levels = 0;
    while (size) {
        levels += 1;
        size /= 2;
    }
    return levels;
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

static PyObject * new_ref(void * obj) {
    Py_INCREF(obj);
    return obj;
}

static int valid_mem(PyObject * mem, Py_ssize_t size) {
    if (!PyMemoryView_Check(mem)) {
        return 0;
    }
    Py_buffer * view = PyMemoryView_GET_BUFFER(mem);
    return PyBuffer_IsContiguous(view, 'C') && (size < 0 || view->len == size);
}

static PyObject * contiguous(PyObject * data) {
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
        res.x = to_int(PySequence_GetItem(obj, 0));
        res.y = to_int(PySequence_GetItem(obj, 1));
    } else {
        res.x = x;
        res.y = y;
    }
    return res;
}

static Viewport to_viewport(PyObject * obj, int x, int y, int width, int height) {
    Viewport res;
    if (obj != Py_None) {
        res.x = to_int(PySequence_GetItem(obj, 0));
        res.y = to_int(PySequence_GetItem(obj, 1));
        res.width = to_int(PySequence_GetItem(obj, 2));
        res.height = to_int(PySequence_GetItem(obj, 3));
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
    PyObject * str_triangles;
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

typedef struct BufferBinding {
    struct Buffer * buffer;
    int offset;
    int size;
} BufferBinding;

typedef struct SamplerBinding {
    GLObject * sampler;
    struct Image * image;
} SamplerBinding;

typedef struct DescriptorSetBuffers {
    int binding_count;
    BufferBinding binding[MAX_BUFFER_BINDINGS];
} DescriptorSetBuffers;

typedef struct DescriptorSetSamplers {
    int binding_count;
    SamplerBinding binding[MAX_SAMPLER_BINDINGS];
} DescriptorSetSamplers;

typedef struct DescriptorSet {
    PyObject_HEAD
    int uses;
    DescriptorSetBuffers uniform_buffers;
    DescriptorSetSamplers samplers;
} DescriptorSet;

typedef struct BlendState {
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
    ImageFormat fmt;
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

typedef struct RenderParameters {
    int vertex_count;
    int instance_count;
    int first_vertex;
} RenderParameters;

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
    PyObject * uniforms;
    PyObject * uniform_layout;
    PyObject * uniform_data;
    PyObject * viewport_data;
    PyObject * render_data;
    RenderParameters params;
    Viewport viewport;
    int topology;
    int index_type;
    int index_size;
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
        if (set->uniform_buffers.binding_count) {
            for (int i = 0; i < set->uniform_buffers.binding_count; ++i) {
                if (set->uniform_buffers.binding[i].buffer) {
                    gl->BindBufferRange(
                        GL_UNIFORM_BUFFER,
                        i,
                        set->uniform_buffers.binding[i].buffer->buffer,
                        set->uniform_buffers.binding[i].offset,
                        set->uniform_buffers.binding[i].size
                    );
                }
            }
        }
        if (set->samplers.binding_count) {
            for (int i = 0; i < set->samplers.binding_count; ++i) {
                if (set->samplers.binding[i].image) {
                    gl->ActiveTexture(GL_TEXTURE0 + i);
                    gl->BindTexture(set->samplers.binding[i].image->target, set->samplers.binding[i].image->image);
                    gl->BindSampler(i, set->samplers.binding[i].sampler->obj);
                }
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
    gl->GenFramebuffers(1, &framebuffer);
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
        int buffer = face->image->fmt.buffer;
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

    int draw_buffers[MAX_ATTACHMENTS];
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

static void bind_uniforms(Context * self, PyObject * uniform_layout, PyObject * uniform_data) {
    const GLMethods * const gl = &self->gl;
    const UniformHeader * const header = (UniformHeader *)PyMemoryView_GET_BUFFER(uniform_layout)->buf;
    const char * const data = (char *)PyMemoryView_GET_BUFFER(uniform_data)->buf;
    for (int i = 0; i < header->count; ++i) {
        const void * func = (char *)gl + uniform_setter_offset[header->binding[i].function];
        const void * ptr = data + header->binding[i].offset;
        if (header->binding[i].function & 0x10) {
            (*(UniformMatrixSetter *)func)(header->binding[i].location, header->binding[i].count, 0, ptr);
        } else {
            (*(UniformSetter *)func)(header->binding[i].location, header->binding[i].count, ptr);
        }
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
    gl->GenVertexArrays(1, &vertex_array);
    bind_vertex_array(self, vertex_array);

    for (int i = 1; i < length; i += 6) {
        Buffer * buffer = (Buffer *)seq[i + 0];
        int location = to_int(seq[i + 1]);
        int offset = to_int(seq[i + 2]);
        int stride = to_int(seq[i + 3]);
        int divisor = to_int(seq[i + 4]);
        VertexFormat fmt;
        if (!get_vertex_format(self->module_state->helper, seq[i + 5], &fmt)) {
            PyErr_Format(PyExc_ValueError, "invalid vertex format");
            return NULL;
        }
        gl->BindBuffer(GL_ARRAY_BUFFER, buffer->buffer);
        if (fmt.integer) {
            gl->VertexAttribIPointer(location, fmt.size, fmt.type, stride, (intptr)offset);
        } else {
            gl->VertexAttribPointer(location, fmt.size, fmt.type, fmt.normalize, stride, (intptr)offset);
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
    gl->GenSamplers(1, &sampler);
    gl->SamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, to_int(seq[0]));
    gl->SamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, to_int(seq[1]));
    gl->SamplerParameterf(sampler, GL_TEXTURE_MIN_LOD, to_float(seq[2]));
    gl->SamplerParameterf(sampler, GL_TEXTURE_MAX_LOD, to_float(seq[3]));

    float lod_bias = to_float(seq[4]);
    if (lod_bias != 0.0f) {
        gl->SamplerParameterf(sampler, GL_TEXTURE_LOD_BIAS, lod_bias);
    }

    gl->SamplerParameteri(sampler, GL_TEXTURE_WRAP_S, to_int(seq[5]));
    gl->SamplerParameteri(sampler, GL_TEXTURE_WRAP_T, to_int(seq[6]));
    gl->SamplerParameteri(sampler, GL_TEXTURE_WRAP_R, to_int(seq[7]));
    gl->SamplerParameteri(sampler, GL_TEXTURE_COMPARE_MODE, to_int(seq[8]));
    gl->SamplerParameteri(sampler, GL_TEXTURE_COMPARE_FUNC, to_int(seq[9]));

    float max_anisotropy = to_float(seq[10]);
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
        int binding = to_int(seq[i + 0]);
        Buffer * buffer = (Buffer *)seq[i + 1];
        int offset = to_int(seq[i + 2]);
        int size = to_int(seq[i + 3]);
        res.binding[binding].buffer = (Buffer *)new_ref(buffer);
        res.binding[binding].offset = offset;
        res.binding[binding].size = size;
        res.binding_count = res.binding_count > (binding + 1) ? res.binding_count : (binding + 1);
    }

    return res;
}

static DescriptorSetSamplers build_descriptor_set_samplers(Context * self, PyObject * bindings) {
    DescriptorSetSamplers res;
    memset(&res, 0, sizeof(res));

    int length = (int)PyTuple_Size(bindings);
    PyObject ** seq = PySequence_Fast_ITEMS(bindings);

    for (int i = 0; i < length; i += 3) {
        int binding = to_int(seq[i + 0]);
        Image * image = (Image *)seq[i + 1];
        GLObject * sampler = build_sampler(self, seq[i + 2]);
        res.binding[binding].sampler = sampler;
        res.binding[binding].image = (Image *)new_ref(image);
        res.binding_count = res.binding_count > (binding + 1) ? res.binding_count : (binding + 1);
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
    res->attachments = to_int(seq[it++]);
    res->cull_face = to_int(seq[it++]);
    res->depth_enabled = PyObject_IsTrue(seq[it++]);
    if (res->depth_enabled) {
        res->depth_func = to_int(seq[it++]);
        res->depth_write = PyObject_IsTrue(seq[it++]);
    }
    res->stencil_enabled = PyObject_IsTrue(seq[it++]);
    if (res->stencil_enabled) {
        res->stencil_front.fail_op = to_int(seq[it++]);
        res->stencil_front.pass_op = to_int(seq[it++]);
        res->stencil_front.depth_fail_op = to_int(seq[it++]);
        res->stencil_front.compare_op = to_int(seq[it++]);
        res->stencil_front.compare_mask = to_int(seq[it++]);
        res->stencil_front.write_mask = to_int(seq[it++]);
        res->stencil_front.reference = to_int(seq[it++]);
        res->stencil_back.fail_op = to_int(seq[it++]);
        res->stencil_back.pass_op = to_int(seq[it++]);
        res->stencil_back.depth_fail_op = to_int(seq[it++]);
        res->stencil_back.compare_op = to_int(seq[it++]);
        res->stencil_back.compare_mask = to_int(seq[it++]);
        res->stencil_back.write_mask = to_int(seq[it++]);
        res->stencil_back.reference = to_int(seq[it++]);
    }
    res->blend_enabled = to_int(seq[it++]);
    if (res->blend_enabled) {
        res->blend.op_color = to_int(seq[it++]);
        res->blend.op_alpha = to_int(seq[it++]);
        res->blend.src_color = to_int(seq[it++]);
        res->blend.dst_color = to_int(seq[it++]);
        res->blend.src_alpha = to_int(seq[it++]);
        res->blend.dst_alpha = to_int(seq[it++]);
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
    int type = to_int(PyTuple_GetItem(pair, 1));
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
        gl->GetActiveAttrib(program, i, 256, &length, &size, &type, name);
        int location = gl->GetAttribLocation(program, name);
        PyList_SET_ITEM(attributes, i, Py_BuildValue("{sssisisi}", "name", name, "location", location, "gltype", type, "size", size));
    }

    for (int i = 0; i < num_uniforms; ++i) {
        int size = 0;
        int type = 0;
        int length = 0;
        char name[256] = {0};
        gl->GetActiveUniform(program, i, 256, &length, &size, &type, name);
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

    int layer = to_int(PyTuple_GetItem(key, 0));
    int level = to_int(PyTuple_GetItem(key, 1));

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
    res->flags = self->fmt.flags;

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
    return res;
}

static void clear_bound_image(Image * self) {
    const GLMethods * const gl = &self->ctx->gl;
    const int depth_mask = self->ctx->current_depth_mask != 1 && (self->fmt.buffer == GL_DEPTH || self->fmt.buffer == GL_DEPTH_STENCIL);
    const int stencil_mask = self->ctx->current_stencil_mask != 0xff && (self->fmt.buffer == GL_STENCIL || self->fmt.buffer == GL_DEPTH_STENCIL);
    if (depth_mask) {
        gl->DepthMask(1);
        self->ctx->current_depth_mask = 1;
    }
    if (stencil_mask) {
        gl->StencilMaskSeparate(GL_FRONT, 0xff);
        self->ctx->current_stencil_mask = 0xff;
    }
    if (self->fmt.clear_type == 'f') {
        gl->ClearBufferfv(self->fmt.buffer, 0, self->clear_value.clear_floats);
    } else if (self->fmt.clear_type == 'i') {
        gl->ClearBufferiv(self->fmt.buffer, 0, self->clear_value.clear_ints);
    } else if (self->fmt.clear_type == 'u') {
        gl->ClearBufferuiv(self->fmt.buffer, 0, self->clear_value.clear_uints);
    } else if (self->fmt.clear_type == 'x') {
        gl->ClearBufferfi(self->fmt.buffer, 0, self->clear_value.clear_floats[0], self->clear_value.clear_ints[1]);
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
        srgb = src->image->fmt.internal_format == GL_SRGB8_ALPHA8 ? Py_True : Py_False;
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

    if (!src->image->fmt.color) {
        PyErr_Format(PyExc_TypeError, "cannot blit depth or stencil images");
        return NULL;
    }

    if (target && !target->image->fmt.color) {
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

    PyObject * res = PyBytes_FromStringAndSize(NULL, (long long)size.x * size.y * src->image->fmt.pixel_size);
    bind_framebuffer(src->ctx, src->framebuffer->obj);
    gl->ReadPixels(offset.x, offset.y, size.x, size.y, src->image->fmt.format, src->image->fmt.type, PyBytes_AS_STRING(res));
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
    res->before_frame_callback = new_ref(Py_None);
    res->after_frame_callback = new_ref(Py_None);
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

    if (res->limits.max_uniform_buffer_bindings > MAX_BUFFER_BINDINGS) {
        res->limits.max_uniform_buffer_bindings = MAX_BUFFER_BINDINGS;
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
        size = to_int(size_arg);
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
        gl->GenBuffers(1, &buffer);
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

    ImageFormat fmt;
    if (!get_image_format(self->module_state->helper, format, &fmt)) {
        PyErr_Format(PyExc_ValueError, "invalid image format");
        return NULL;
    }

    int image = 0;
    if (external) {
        image = external;
    } else if (renderbuffer) {
        gl->GenRenderbuffers(1, &image);
        gl->BindRenderbuffer(GL_RENDERBUFFER, image);
        gl->RenderbufferStorageMultisample(GL_RENDERBUFFER, samples > 1 ? samples : 0, fmt.internal_format, width, height);
    } else {
        gl->GenTextures(1, &image);
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
                    gl->TexImage2D(face, level, fmt.internal_format, w, h, 0, fmt.format, fmt.type, NULL);
                }
            } else if (array) {
                gl->TexImage3D(target, level, fmt.internal_format, w, h, array, 0, fmt.format, fmt.type, NULL);
            } else {
                gl->TexImage2D(target, level, fmt.internal_format, w, h, 0, fmt.format, fmt.type, NULL);
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
    res->format = new_ref(format);
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

    if (fmt.buffer == GL_DEPTH || fmt.buffer == GL_DEPTH_STENCIL) {
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
        "uniform_data",
        "viewport_data",
        "render_data",
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
    PyObject * topology_arg = self->module_state->str_triangles;
    int vertex_count = 0;
    int instance_count = 1;
    int first_vertex = 0;
    PyObject * viewport = Py_None;
    PyObject * uniform_data = Py_None;
    PyObject * viewport_data = Py_None;
    PyObject * render_data = Py_None;
    PyObject * includes = Py_None;

    int args_ok = PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "|O!O!OOOOOOOOOpOOiiiOOOOO",
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
        &topology_arg,
        &vertex_count,
        &instance_count,
        &first_vertex,
        &viewport,
        &uniform_data,
        &viewport_data,
        &render_data,
        &includes
    );

    if (!args_ok) {
        return NULL;
    }

    if (uniforms == Py_None) {
        uniforms = NULL;
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

    if (uniform_data != Py_None && !valid_mem(uniform_data, -1)) {
        PyErr_Format(PyExc_TypeError, "uniform_data must be a contiguous memoryview");
        return NULL;
    }

    if (viewport_data != Py_None && !valid_mem(viewport_data, 16)) {
        PyErr_Format(PyExc_TypeError, "viewport_data must be a contiguous memoryview with a size of 16 bytes");
        return NULL;
    }

    if (render_data != Py_None && !valid_mem(render_data, 12)) {
        PyErr_Format(PyExc_TypeError, "render_data must be a contiguous memoryview with a size of 12 bytes");
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

    PyObject * uniform_layout = NULL;

    if (uniforms) {
        PyObject * tuple = PyObject_CallMethod(self->module_state->helper, "uniforms", "(OOO)", program->extra, uniforms, uniform_data);
        if (!tuple) {
            return NULL;
        }

        uniforms = PyDictProxy_New(PyTuple_GetItem(tuple, 0));
        uniform_layout = PyTuple_GetItem(tuple, 1);
        uniform_data = PyTuple_GetItem(tuple, 2);
        Py_INCREF(uniform_layout);
        Py_INCREF(uniform_data);
        Py_DECREF(tuple);
    }


    PyObject * validate = PyObject_CallMethod(
        self->module_state->helper,
        "validate",
        "(OOOOO)",
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

    PyObject * layout_bindings = PyObject_CallMethod(self->module_state->helper, "layout_bindings", "(O)", layout);
    if (!layout_bindings) {
        return NULL;
    }

    int layout_count = (int)PyList_Size(layout_bindings);
    for (int i = 0; i < layout_count; ++i) {
        PyObject * obj = PyList_GetItem(layout_bindings, i);
        PyObject * name = PyTuple_GetItem(obj, 0);
        int binding = to_int(PyTuple_GetItem(obj, 1));
        int location = gl->GetUniformLocation(program->obj, PyUnicode_AsUTF8(name));
        if (location >= 0) {
            gl->Uniform1i(location, binding);
        } else {
            int index = gl->GetUniformBlockIndex(program->obj, PyUnicode_AsUTF8(name));
            gl->UniformBlockBinding(program->obj, index, binding);
        }
    }

    Py_DECREF(layout_bindings);

    PyObject * attachments = PyObject_CallMethod(self->module_state->helper, "framebuffer_attachments", "(O)", framebuffer_attachments);
    if (!attachments) {
        return NULL;
    }
    if (attachments != Py_None && viewport == Py_None) {
        PyObject * size = PyTuple_GetItem(attachments, 0);
        viewport_value.width = to_int(PyTuple_GetItem(size, 0));
        viewport_value.height = to_int(PyTuple_GetItem(size, 1));
    }

    GLObject * framebuffer = build_framebuffer(self, attachments);

    PyObject * bindings = PyObject_CallMethod(self->module_state->helper, "vertex_array_bindings", "(OO)", vertex_buffers, index_buffer);
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

    PyObject * settings = PyObject_CallMethod(self->module_state->helper, "settings", "(OOOON)", cull_face, depth, stencil, blend, attachments);
    if (!settings) {
        return NULL;
    }

    GlobalSettings * global_settings = build_global_settings(self, settings);
    Py_DECREF(settings);

    int topology;
    if (!get_topology(self->module_state->helper, topology_arg, &topology)) {
        PyErr_Format(PyExc_ValueError, "invalid topology");
        return NULL;
    }

    Pipeline * res = PyObject_New(Pipeline, self->module_state->Pipeline_type);
    res->gc_prev = self->gc_prev;
    res->gc_next = (GCHeader *)self;
    res->gc_prev->gc_next = (GCHeader *)res;
    res->gc_next->gc_prev = (GCHeader *)res;
    Py_INCREF(res);

    if (viewport_data == Py_None) {
        viewport_data = PyMemoryView_FromMemory((char *)&res->viewport, sizeof(res->viewport), PyBUF_WRITE);
    }

    if (render_data == Py_None) {
        render_data = PyMemoryView_FromMemory((char *)&res->params, sizeof(res->params), PyBUF_WRITE);
    }

    res->ctx = self;
    res->framebuffer = framebuffer;
    res->vertex_array = vertex_array;
    res->program = program;
    res->uniforms = uniforms;
    res->uniform_layout = uniform_layout;
    res->uniform_data = uniform_data;
    res->viewport_data = viewport_data;
    res->render_data = render_data;
    res->topology = topology;
    res->viewport = viewport_value;
    res->params.vertex_count = vertex_count;
    res->params.instance_count = instance_count;
    res->params.first_vertex = first_vertex;
    res->index_type = index_type;
    res->index_size = index_size;
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
            gl->GenQueries(1, &self->frame_time_query);
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
        gl->GetQueryObjectuiv(self->frame_time_query, GL_QUERY_RESULT, &self->frame_time);
        self->frame_time_query_running = 0;
    } else {
        self->frame_time = 0;
    }

    if (flush) {
        gl->Flush();
    }

    if (sync) {
        void * fence = gl->FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl->ClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, -1);
        gl->DeleteSync(fence);
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
        for (int i = 0; i < set->samplers.binding_count; ++i) {
            GLObject * sampler = set->samplers.binding[i].sampler;
            if (sampler) {
                sampler->uses -= 1;
                if (!sampler->uses) {
                    remove_dict_value(self->sampler_cache, (PyObject *)sampler);
                    gl->DeleteSamplers(1, &sampler->obj);
                }
            }
        }
        for (int i = 0; i < set->uniform_buffers.binding_count; ++i) {
            Py_XDECREF(set->uniform_buffers.binding[i].buffer);
        }
        for (int i = 0; i < set->samplers.binding_count; ++i) {
            Py_XDECREF(set->samplers.binding[i].sampler);
            Py_XDECREF(set->samplers.binding[i].image);
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
            gl->DeleteFramebuffers(1, &framebuffer->obj);
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
        gl->DeleteVertexArrays(1, &vertex_array->obj);
    }
}

static PyObject * Context_meth_release(Context * self, PyObject * arg) {
    const GLMethods * const gl = &self->gl;
    if (Py_TYPE(arg) == self->module_state->Buffer_type) {
        Buffer * buffer = (Buffer *)arg;
        buffer->gc_prev->gc_next = buffer->gc_next;
        buffer->gc_next->gc_prev = buffer->gc_prev;
        gl->DeleteBuffers(1, &buffer->buffer);
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
            gl->DeleteRenderbuffers(1, &image->image);
        } else {
            gl->DeleteTextures(1, &image->image);
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

    self->default_framebuffer->obj = to_int(value);
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
    int size = (int)view->len;

    if (size + offset > self->size) {
        PyErr_Format(PyExc_ValueError, "invalid size");
        return NULL;
    }

    const GLMethods * const gl = &self->ctx->gl;

    if (view->len) {
        gl->BindBuffer(GL_ARRAY_BUFFER, self->buffer);
        gl->BufferSubData(GL_ARRAY_BUFFER, offset, size, view->buf);
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
        size = to_int(size_arg);
    }

    if (offset_arg != Py_None) {
        offset = to_int(offset_arg);
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
        layer = to_int(layer_arg);
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

    if (!self->fmt.color) {
        PyErr_Format(PyExc_TypeError, "cannot write to depth or stencil images");
        return NULL;
    }

    if (self->samples != 1) {
        PyErr_Format(PyExc_TypeError, "cannot write to multisampled images");
        return NULL;
    }

    int padded_row = (size.x * self->fmt.pixel_size + 3) & ~3;
    int expected_size = padded_row * size.y;

    if (layer_arg == Py_None) {
        expected_size *= self->layer_count;
    }

    PyObject * mem = contiguous(data);
    if (!mem) {
        return NULL;
    }

    Py_buffer * view = PyMemoryView_GET_BUFFER(mem);
    int data_size = (int)view->len;

    if (data_size != expected_size) {
        PyErr_Format(PyExc_ValueError, "invalid data size, expected %d, got %d", expected_size, data_size);
        return NULL;
    }

    const GLMethods * const gl = &self->ctx->gl;

    gl->ActiveTexture(self->ctx->default_texture_unit);
    gl->BindTexture(self->target, self->image);
    if (self->cubemap) {
        int stride = padded_row * size.y;
        if (layer_arg != Py_None) {
            int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer;
            gl->TexSubImage2D(face, level, offset.x, offset.y, size.x, size.y, self->fmt.format, self->fmt.type, view->buf);
        } else {
            for (int i = 0; i < 6; ++i) {
                int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i;
                gl->TexSubImage2D(face, level, offset.x, offset.y, size.x, size.y, self->fmt.format, self->fmt.type, (char *)view->buf + stride * i);
            }
        }
    } else if (self->array) {
        if (layer_arg != Py_None) {
            gl->TexSubImage3D(self->target, level, offset.x, offset.y, layer, size.x, size.y, 1, self->fmt.format, self->fmt.type, view->buf);
        } else {
            gl->TexSubImage3D(self->target, level, offset.x, offset.y, 0, size.x, size.y, self->array, self->fmt.format, self->fmt.type, view->buf);
        }
    } else {
        gl->TexSubImage2D(self->target, level, offset.x, offset.y, size.x, size.y, self->fmt.format, self->fmt.type, view->buf);
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
    if (self->fmt.clear_type == 'x') {
        return Py_BuildValue("dI", (double)self->clear_value.clear_floats[0], self->clear_value.clear_uints[1]);
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
    if (self->fmt.components == 1) {
        if (self->fmt.clear_type == 'f' && !PyFloat_CheckExact(value)) {
            PyErr_Format(PyExc_TypeError, "the clear value must be a float");
            return -1;
        }
        if (self->fmt.clear_type == 'i' && !PyLong_CheckExact(value)) {
            PyErr_Format(PyExc_TypeError, "the clear value must be an int");
            return -1;
        }
        if (self->fmt.clear_type == 'f') {
            self->clear_value.clear_floats[0] = to_float(value);
        } else if (self->fmt.clear_type == 'i') {
            self->clear_value.clear_ints[0] = to_int(value);
        } else if (self->fmt.clear_type == 'u') {
            self->clear_value.clear_uints[0] = to_uint(value);
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

    if (size != self->fmt.components) {
        Py_DECREF(values);
        PyErr_Format(PyExc_ValueError, "invalid clear value size");
        return -1;
    }

    if (self->fmt.clear_type == 'f') {
        for (int i = 0; i < self->fmt.components; ++i) {
            self->clear_value.clear_floats[i] = to_float(seq[i]);
        }
    } else if (self->fmt.clear_type == 'i') {
        for (int i = 0; i < self->fmt.components; ++i) {
            self->clear_value.clear_ints[i] = to_int(seq[i]);
        }
    } else if (self->fmt.clear_type == 'u') {
        for (int i = 0; i < self->fmt.components; ++i) {
            self->clear_value.clear_uints[i] = to_uint(seq[i]);
        }
    } else if (self->fmt.clear_type == 'x') {
        self->clear_value.clear_floats[0] = to_float(seq[0]);
        self->clear_value.clear_ints[1] = to_int(seq[1]);
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
    Viewport * viewport = (Viewport *)PyMemoryView_GET_BUFFER(self->viewport_data)->buf;
    if (memcmp(viewport, &self->ctx->current_viewport, sizeof(Viewport))) {
        gl->Viewport(viewport->x, viewport->y, viewport->width, viewport->height);
        self->ctx->current_viewport = *viewport;
    }
    bind_global_settings(self->ctx, self->global_settings);
    bind_framebuffer(self->ctx, self->framebuffer->obj);
    bind_program(self->ctx, self->program->obj);
    bind_vertex_array(self->ctx, self->vertex_array->obj);
    bind_descriptor_set(self->ctx, self->descriptor_set);
    if (self->uniforms) {
        bind_uniforms(self->ctx, self->uniform_layout, self->uniform_data);
    }
    RenderParameters * params = (RenderParameters *)PyMemoryView_GET_BUFFER(self->render_data)->buf;
    if (self->index_type) {
        intptr offset = (intptr)params->first_vertex * (intptr)self->index_size;
        gl->DrawElementsInstanced(self->topology, params->vertex_count, self->index_type, offset, params->instance_count);
    } else {
        gl->DrawArraysInstanced(self->topology, params->first_vertex, params->vertex_count, params->instance_count);
    }
    Py_RETURN_NONE;
}

static PyObject * Pipeline_get_viewport(Pipeline * self, void * closure) {
    return Py_BuildValue("(iiii)", self->viewport.x, self->viewport.y, self->viewport.width, self->viewport.height);
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
    for (int i = 0; i < set->uniform_buffers.binding_count; ++i) {
        if (set->uniform_buffers.binding[i].buffer) {
            PyObject * obj = Py_BuildValue(
                "{sssisisisi}",
                "type", "uniform_buffer",
                "binding", i,
                "buffer", set->uniform_buffers.binding[i].buffer->buffer,
                "offset", set->uniform_buffers.binding[i].offset,
                "size", set->uniform_buffers.binding[i].size
            );
            PyList_Append(res, obj);
            Py_DECREF(obj);
        }
    }
    for (int i = 0; i < set->samplers.binding_count; ++i) {
        if (set->samplers.binding[i].sampler) {
            PyObject * obj = Py_BuildValue(
                "{sssisisi}",
                "type", "sampler",
                "binding", i,
                "sampler", set->samplers.binding[i].sampler->obj,
                "texture", set->samplers.binding[i].image->image
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
    Py_XDECREF(self->uniforms);
    Py_XDECREF(self->uniform_layout);
    Py_XDECREF(self->uniform_data);
    Py_DECREF(self->viewport_data);
    Py_DECREF(self->render_data);
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
    {"buffer", (PyCFunction)Context_meth_buffer, METH_VARARGS | METH_KEYWORDS, NULL},
    {"image", (PyCFunction)Context_meth_image, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pipeline", (PyCFunction)Context_meth_pipeline, METH_VARARGS | METH_KEYWORDS, NULL},
    {"new_frame", (PyCFunction)Context_meth_new_frame, METH_VARARGS | METH_KEYWORDS, NULL},
    {"end_frame", (PyCFunction)Context_meth_end_frame, METH_VARARGS | METH_KEYWORDS, NULL},
    {"release", (PyCFunction)Context_meth_release, METH_O, NULL},
    {"gc", (PyCFunction)Context_meth_gc, METH_NOARGS, NULL},
    {0},
};

static PyGetSetDef Context_getset[] = {
    {"screen", (getter)Context_get_screen, (setter)Context_set_screen, NULL, NULL},
    {0},
};

static PyMemberDef Context_members[] = {
    {"includes", T_OBJECT, offsetof(Context, includes), READONLY, NULL},
    {"limits", T_OBJECT, offsetof(Context, limits_dict), READONLY, NULL},
    {"info", T_OBJECT, offsetof(Context, info_dict), READONLY, NULL},
    {"before_frame", T_OBJECT, offsetof(Context, before_frame_callback), 0, NULL},
    {"after_frame", T_OBJECT, offsetof(Context, after_frame_callback), 0, NULL},
    {"frame_time", T_INT, offsetof(Context, frame_time), READONLY, NULL},
    {0},
};

static PyMethodDef Buffer_methods[] = {
    {"write", (PyCFunction)Buffer_meth_write, METH_VARARGS | METH_KEYWORDS, NULL},
    {"map", (PyCFunction)Buffer_meth_map, METH_VARARGS | METH_KEYWORDS, NULL},
    {"unmap", (PyCFunction)Buffer_meth_unmap, METH_NOARGS, NULL},
    {0},
};

static PyMemberDef Buffer_members[] = {
    {"size", T_INT, offsetof(Buffer, size), READONLY, NULL},
    {0},
};

static PyMethodDef Image_methods[] = {
    {"clear", (PyCFunction)Image_meth_clear, METH_NOARGS, NULL},
    {"write", (PyCFunction)Image_meth_write, METH_VARARGS | METH_KEYWORDS, NULL},
    {"read", (PyCFunction)Image_meth_read, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mipmaps", (PyCFunction)Image_meth_mipmaps, METH_NOARGS, NULL},
    {"blit", (PyCFunction)Image_meth_blit, METH_VARARGS | METH_KEYWORDS, NULL},
    {"face", (PyCFunction)Image_meth_face, METH_VARARGS | METH_KEYWORDS, NULL},
    {0},
};

static PyGetSetDef Image_getset[] = {
    {"clear_value", (getter)Image_get_clear_value, (setter)Image_set_clear_value, NULL, NULL},
    {0},
};

static PyMemberDef Image_members[] = {
    {"size", T_OBJECT, offsetof(Image, size), READONLY, NULL},
    {"format", T_OBJECT, offsetof(Image, format), READONLY, NULL},
    {"samples", T_INT, offsetof(Image, samples), READONLY, NULL},
    {"array", T_INT, offsetof(Image, array), READONLY, NULL},
    {0},
};

static PyMethodDef Pipeline_methods[] = {
    {"render", (PyCFunction)Pipeline_meth_render, METH_NOARGS, NULL},
    {0},
};

static PyGetSetDef Pipeline_getset[] = {
    {"viewport", (getter)Pipeline_get_viewport, (setter)Pipeline_set_viewport, NULL, NULL},
    {0},
};

static PyMemberDef Pipeline_members[] = {
    {"vertex_count", T_INT, offsetof(Pipeline, params.vertex_count), 0, NULL},
    {"instance_count", T_INT, offsetof(Pipeline, params.instance_count), 0, NULL},
    {"first_vertex", T_INT, offsetof(Pipeline, params.first_vertex), 0, NULL},
    {"uniforms", T_OBJECT, offsetof(Pipeline, uniforms), READONLY, NULL},
    {0},
};

static PyMethodDef ImageFace_methods[] = {
    {"clear", (PyCFunction)ImageFace_meth_clear, METH_NOARGS, NULL},
    {"read", (PyCFunction)ImageFace_meth_read, METH_VARARGS | METH_KEYWORDS, NULL},
    {"blit", (PyCFunction)ImageFace_meth_blit, METH_VARARGS | METH_KEYWORDS, NULL},
    {0},
};

static PyMemberDef ImageFace_members[] = {
    {"image", T_OBJECT, offsetof(ImageFace, image), READONLY, NULL},
    {"size", T_OBJECT, offsetof(ImageFace, size), READONLY, NULL},
    {"layer", T_INT, offsetof(ImageFace, layer), READONLY, NULL},
    {"level", T_INT, offsetof(ImageFace, level), READONLY, NULL},
    {"samples", T_INT, offsetof(ImageFace, samples), READONLY, NULL},
    {"flags", T_INT, offsetof(ImageFace, flags), READONLY, NULL},
    {0},
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
    state->str_triangles = PyUnicode_FromString("triangles");
    state->Context_type = (PyTypeObject *)PyType_FromSpec(&Context_spec);
    state->Buffer_type = (PyTypeObject *)PyType_FromSpec(&Buffer_spec);
    state->Image_type = (PyTypeObject *)PyType_FromSpec(&Image_spec);
    state->Pipeline_type = (PyTypeObject *)PyType_FromSpec(&Pipeline_spec);
    state->ImageFace_type = (PyTypeObject *)PyType_FromSpec(&ImageFace_spec);
    state->DescriptorSet_type = (PyTypeObject *)PyType_FromSpec(&DescriptorSet_spec);
    state->GlobalSettings_type = (PyTypeObject *)PyType_FromSpec(&GlobalSettings_spec);
    state->GLObject_type = (PyTypeObject *)PyType_FromSpec(&GLObject_spec);

    PyModule_AddObject(self, "Context", new_ref(state->Context_type));
    PyModule_AddObject(self, "Buffer", new_ref(state->Buffer_type));
    PyModule_AddObject(self, "Image", new_ref(state->Image_type));
    PyModule_AddObject(self, "ImageFace", new_ref(state->ImageFace_type));
    PyModule_AddObject(self, "Pipeline", new_ref(state->Pipeline_type));

    PyModule_AddObject(self, "loader", PyObject_GetAttrString(state->helper, "loader"));
    PyModule_AddObject(self, "calcsize", PyObject_GetAttrString(state->helper, "calcsize"));
    PyModule_AddObject(self, "bind", PyObject_GetAttrString(state->helper, "bind"));

    PyModule_AddObject(self, "__version__", PyUnicode_FromString("1.13.0"));

    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, (void *)module_exec},
    {0},
};

static PyMethodDef module_methods[] = {
    {"context", (PyCFunction)meth_context, METH_VARARGS | METH_KEYWORDS, NULL},
    {"inspect", (PyCFunction)meth_inspect, METH_O, NULL},
    {"camera", (PyCFunction)meth_camera, METH_VARARGS | METH_KEYWORDS, NULL},
    {0},
};

static void module_free(PyObject * self) {
    ModuleState * state = (ModuleState *)PyModule_GetState(self);
    if (state) {
        Py_DECREF(state->empty_tuple);
        Py_DECREF(state->str_none);
        Py_DECREF(state->str_triangles);
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
