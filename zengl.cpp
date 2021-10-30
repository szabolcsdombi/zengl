#include "zengl.hpp"

struct ModuleState {
    PyObject * helper;
    PyObject * empty_tuple;
    PyObject * str_none;
    PyObject * str_ccw;
    PyObject * float_one;
    PyObject * default_color_mask;
    PyTypeObject * Instance_type;
    PyTypeObject * Buffer_type;
    PyTypeObject * Image_type;
    PyTypeObject * Renderer_type;
    PyTypeObject * DescriptorSetBuffers_type;
    PyTypeObject * DescriptorSetImages_type;
    PyTypeObject * GlobalSettings_type;
    PyTypeObject * GLObject_type;
};

struct GLObject {
    PyObject_HEAD
    int uses;
    int obj;
};

struct DescriptorSetBuffers {
    PyObject_HEAD
    int uses;
    int buffers;
    UniformBufferBinding binding[MAX_UNIFORM_BUFFER_BINDINGS];
};

struct DescriptorSetImages {
    PyObject_HEAD
    int uses;
    int samplers;
    SamplerBinding binding[MAX_SAMPLER_BINDINGS];
    GLObject * sampler[MAX_SAMPLER_BINDINGS];
};

struct GlobalSettings {
    PyObject_HEAD
    int uses;
    unsigned long long color_mask;
    int primitive_restart;
    float line_width;
    int front_face;
    int cull_face;
    int depth_test;
    int depth_write;
    int depth_func;
    int stencil_test;
    StencilSettings stencil_front;
    StencilSettings stencil_back;
    int blend_enable;
    int blend_src_color;
    int blend_dst_color;
    int blend_src_alpha;
    int blend_dst_alpha;
    int polygon_offset;
    float polygon_offset_factor;
    float polygon_offset_units;
    int attachments;
};

struct Instance {
    PyObject_HEAD
    ModuleState * module_state;
    PyObject * descriptor_set_buffers_cache;
    PyObject * descriptor_set_images_cache;
    PyObject * global_settings_cache;
    PyObject * sampler_cache;
    PyObject * vertex_array_cache;
    PyObject * framebuffer_cache;
    PyObject * program_cache;
    PyObject * shader_cache;
    PyObject * files;
    PyObject * info;
    DescriptorSetBuffers * current_buffers;
    DescriptorSetImages * current_images;
    GlobalSettings * current_global_settings;
    Viewport viewport;
    int current_framebuffer;
    int current_program;
    int current_vertex_array;
    int default_texture_unit;
    int mapped_buffers;
    GLMethods gl;
};

struct Buffer {
    PyObject_HEAD
    Instance * instance;
    int buffer;
    int size;
    int mapped;
};

struct Image {
    PyObject_HEAD
    Instance * instance;
    PyObject * size;
    GLObject * framebuffer;
    ClearValue clear_value;
    ImageFormat format;
    int image;
    int width;
    int height;
    int samples;
    int array;
    int cubemap;
    int target;
    int renderbuffer;
};

struct Renderer {
    PyObject_HEAD
    Instance * instance;
    DescriptorSetBuffers * descriptor_set_buffers;
    DescriptorSetImages * descriptor_set_images;
    GlobalSettings * global_settings;
    GLObject * framebuffer;
    GLObject * vertex_array;
    GLObject * program;
    int topology;
    int vertex_count;
    int instance_count;
    int first_vertex;
    int index_type;
    int index_size;
    Viewport viewport;
};

void bind_descriptor_set_buffers(Instance * self, DescriptorSetBuffers * set) {
    const GLMethods & gl = self->gl;
    if (self->current_buffers != set) {
        self->current_buffers = set;
        for (int i = 0; i < set->buffers; ++i) {
            gl.BindBufferRange(
                GL_UNIFORM_BUFFER,
                i,
                set->binding[i].buffer,
                set->binding[i].offset,
                set->binding[i].size
            );
        }
    }
}

void bind_descriptor_set_images(Instance * self, DescriptorSetImages * set) {
    const GLMethods & gl = self->gl;
    if (self->current_images != set) {
        self->current_images = set;
        for (int i = 0; i < set->samplers; ++i) {
            gl.ActiveTexture(GL_TEXTURE0 + i);
            gl.BindTexture(set->binding[i].target, set->binding[i].image);
            gl.BindSampler(i, set->binding[i].sampler);
        }
    }
}

void bind_global_settings(Instance * self, GlobalSettings * settings) {
    const GLMethods & gl = self->gl;
    if (settings->primitive_restart) {
        gl.Enable(GL_PRIMITIVE_RESTART);
    } else {
        gl.Disable(GL_PRIMITIVE_RESTART);
    }
    if (settings->polygon_offset) {
        gl.Enable(GL_POLYGON_OFFSET_FILL);
        gl.Enable(GL_POLYGON_OFFSET_LINE);
        gl.Enable(GL_POLYGON_OFFSET_POINT);
    } else {
        gl.Disable(GL_POLYGON_OFFSET_FILL);
        gl.Disable(GL_POLYGON_OFFSET_LINE);
        gl.Disable(GL_POLYGON_OFFSET_POINT);
    }
    if (settings->stencil_test) {
        gl.Enable(GL_STENCIL_TEST);
    } else {
        gl.Disable(GL_STENCIL_TEST);
    }
    if (settings->depth_test) {
        gl.Enable(GL_DEPTH_TEST);
    } else {
        gl.Disable(GL_DEPTH_TEST);
    }
    if (settings->cull_face) {
        gl.Enable(GL_CULL_FACE);
        gl.CullFace(settings->cull_face);
    } else {
        gl.Disable(GL_CULL_FACE);
    }
    gl.LineWidth(settings->line_width);
    gl.FrontFace(settings->front_face);
    gl.DepthMask(settings->depth_write);
    gl.StencilMaskSeparate(GL_FRONT, settings->stencil_front.write_mask);
    gl.StencilMaskSeparate(GL_BACK, settings->stencil_back.write_mask);
    gl.StencilFuncSeparate(GL_FRONT, settings->stencil_front.compare_op, settings->stencil_front.reference, settings->stencil_front.compare_mask);
    gl.StencilFuncSeparate(GL_BACK, settings->stencil_back.compare_op, settings->stencil_back.reference, settings->stencil_back.compare_mask);
    gl.StencilOpSeparate(GL_FRONT, settings->stencil_front.fail_op, settings->stencil_front.pass_op, settings->stencil_front.depth_fail_op);
    gl.StencilOpSeparate(GL_BACK, settings->stencil_back.fail_op, settings->stencil_back.pass_op, settings->stencil_back.depth_fail_op);
    gl.BlendFuncSeparate(settings->blend_src_color, settings->blend_dst_color, settings->blend_src_alpha, settings->blend_dst_alpha);
    gl.PolygonOffset(settings->polygon_offset_factor, settings->polygon_offset_units);
    for (int i = 0; i < settings->attachments; ++i) {
        if (settings->blend_enable >> i & 1) {
            gl.Enablei(GL_BLEND, i);
        } else {
            gl.Disablei(GL_BLEND, i);
        }
        gl.ColorMaski(
            i,
            settings->color_mask >> (i * 4 + 0) & 1,
            settings->color_mask >> (i * 4 + 1) & 1,
            settings->color_mask >> (i * 4 + 2) & 1,
            settings->color_mask >> (i * 4 + 3) & 1
        );
    }
}

void bind_framebuffer(Instance * self, int framebuffer) {
    if (self->current_framebuffer != framebuffer) {
        self->current_framebuffer = framebuffer;
        self->gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    }
}

void bind_program(Instance * self, int program) {
    if (self->current_program != program) {
        self->current_program = program;
        self->gl.UseProgram(program);
    }
}

void bind_vertex_array(Instance * self, int vertex_array) {
    if (self->current_vertex_array != vertex_array) {
        self->current_vertex_array = vertex_array;
        self->gl.BindVertexArray(vertex_array);
    }
}

GLObject * build_framebuffer(Instance * self, PyObject * attachments) {
    if (GLObject * cache = (GLObject *)PyDict_GetItem(self->framebuffer_cache, attachments)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    PyObject * color_attachments = PyTuple_GetItem(attachments, 0);
    PyObject * depth_stencil_attachment = PyTuple_GetItem(attachments, 1);

    const GLMethods & gl = self->gl;

    int framebuffer = 0;
    gl.GenFramebuffers(1, (unsigned *)&framebuffer);
    bind_framebuffer(self, framebuffer);
    int color_attachment_count = (int)PyTuple_Size(color_attachments);
    for (int i = 0; i < color_attachment_count; ++i) {
        Image * image = (Image *)PyTuple_GetItem(color_attachments, i);
        if (image->renderbuffer) {
            gl.FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_RENDERBUFFER, image->image);
        } else {
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, image->image, 0);
        }
    }

    if (depth_stencil_attachment != Py_None) {
        Image * image = (Image *)depth_stencil_attachment;
        int buffer = image->format.buffer;
        int attachment = buffer == GL_DEPTH ? GL_DEPTH_ATTACHMENT : buffer == GL_STENCIL ? GL_STENCIL_ATTACHMENT : GL_DEPTH_STENCIL_ATTACHMENT;
        if (image->renderbuffer) {
            gl.FramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, image->image);
        } else {
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, image->image, 0);
        }
    }

    unsigned int draw_buffers[MAX_ATTACHMENTS];
    for (int i = 0; i < color_attachment_count; ++i) {
        draw_buffers[i] = GL_COLOR_ATTACHMENT0 + i;
    }

    gl.DrawBuffers(color_attachment_count, draw_buffers);
    gl.ReadBuffer(GL_COLOR_ATTACHMENT0);

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = framebuffer;
    res->uses = 1;

    PyDict_SetItem(self->framebuffer_cache, attachments, (PyObject *)res);
    return res;
}

GLObject * build_vertex_array(Instance * self, PyObject * bindings) {
    if (GLObject * cache = (GLObject *)PyDict_GetItem(self->framebuffer_cache, bindings)) {
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

GLObject * build_sampler(Instance * self, PyObject * params) {
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
    gl.SamplerParameteri(sampler, GL_TEXTURE_WRAP_S, PyLong_AsLong(seq[4]));
    gl.SamplerParameteri(sampler, GL_TEXTURE_WRAP_T, PyLong_AsLong(seq[5]));
    gl.SamplerParameteri(sampler, GL_TEXTURE_WRAP_R, PyLong_AsLong(seq[6]));
    gl.SamplerParameteri(sampler, GL_TEXTURE_COMPARE_MODE, PyLong_AsLong(seq[7]));
    gl.SamplerParameteri(sampler, GL_TEXTURE_COMPARE_FUNC, PyLong_AsLong(seq[8]));

    float color[] = {
        (float)PyFloat_AsDouble(seq[9]),
        (float)PyFloat_AsDouble(seq[10]),
        (float)PyFloat_AsDouble(seq[11]),
        (float)PyFloat_AsDouble(seq[12]),
    };
    gl.SamplerParameterfv(sampler, GL_TEXTURE_BORDER_COLOR, color);

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = sampler;
    res->uses = 1;

    PyDict_SetItem(self->sampler_cache, params, (PyObject *)res);
    return res;
}

DescriptorSetBuffers * build_descriptor_set_buffers(Instance * self, PyObject * bindings) {
    if (DescriptorSetBuffers * cache = (DescriptorSetBuffers *)PyDict_GetItem(self->descriptor_set_buffers_cache, bindings)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    const GLMethods & gl = self->gl;

    int length = (int)PyTuple_Size(bindings);
    PyObject ** seq = PySequence_Fast_ITEMS(bindings);

    DescriptorSetBuffers * res = PyObject_New(DescriptorSetBuffers, self->module_state->DescriptorSetBuffers_type);
    memset(res->binding, 0, sizeof(res->binding));
    res->buffers = 0;
    res->uses = 1;

    for (int i = 0; i < length; i += 4) {
        int binding = PyLong_AsLong(seq[i + 0]);
        Buffer * buffer = (Buffer *)seq[i + 1];
        int offset = PyLong_AsLong(seq[i + 2]);
        int size = PyLong_AsLong(seq[i + 3]);
        res->binding[binding] = {buffer->buffer, 0, buffer->size};
        res->buffers = res->buffers > (binding + 1) ? res->buffers : (binding + 1);
    }

    PyDict_SetItem(self->descriptor_set_buffers_cache, bindings, (PyObject *)res);
    return res;
}

DescriptorSetImages * build_descriptor_set_images(Instance * self, PyObject * bindings) {
    if (DescriptorSetImages * cache = (DescriptorSetImages *)PyDict_GetItem(self->descriptor_set_images_cache, bindings)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    const GLMethods & gl = self->gl;

    int length = (int)PyTuple_Size(bindings);
    PyObject ** seq = PySequence_Fast_ITEMS(bindings);

    DescriptorSetImages * res = PyObject_New(DescriptorSetImages, self->module_state->DescriptorSetImages_type);
    memset(res->binding, 0, sizeof(res->binding));
    res->samplers = 0;
    res->uses = 1;

    for (int i = 0; i < length; i += 3) {
        int binding = PyLong_AsLong(seq[i + 0]);
        Image * image = (Image *)seq[i + 1];
        res->sampler[binding] = build_sampler(self, seq[i + 2]);
        res->binding[binding] = {res->sampler[binding]->obj, image->target, image->image};
        res->samplers = res->samplers > (binding + 1) ? res->samplers : (binding + 1);
    }

    PyDict_SetItem(self->descriptor_set_images_cache, bindings, (PyObject *)res);
    return res;
}

GlobalSettings * build_global_settings(Instance * self, PyObject * settings) {
    if (GlobalSettings * cache = (GlobalSettings *)PyDict_GetItem(self->global_settings_cache, settings)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    PyObject ** seq = PySequence_Fast_ITEMS(settings);

    GlobalSettings * res = PyObject_New(GlobalSettings, self->module_state->GlobalSettings_type);
    res->uses = 1;

    res->primitive_restart = PyObject_IsTrue(seq[0]);
    res->line_width = (float)PyFloat_AsDouble(seq[1]);
    res->front_face = PyLong_AsLong(seq[2]);
    res->cull_face = PyLong_AsLong(seq[3]);
    res->color_mask = PyLong_AsUnsignedLongLong(seq[4]);
    res->depth_test = PyObject_IsTrue(seq[5]);
    res->depth_write = PyObject_IsTrue(seq[6]);
    res->depth_func = PyLong_AsLong(seq[7]);
    res->stencil_test = PyObject_IsTrue(seq[8]);
    res->stencil_front = {
        PyLong_AsLong(seq[9]),
        PyLong_AsLong(seq[10]),
        PyLong_AsLong(seq[11]),
        PyLong_AsLong(seq[12]),
        PyLong_AsLong(seq[13]),
        PyLong_AsLong(seq[14]),
        PyLong_AsLong(seq[15]),
    };
    res->stencil_back = {
        PyLong_AsLong(seq[16]),
        PyLong_AsLong(seq[17]),
        PyLong_AsLong(seq[18]),
        PyLong_AsLong(seq[19]),
        PyLong_AsLong(seq[20]),
        PyLong_AsLong(seq[21]),
        PyLong_AsLong(seq[22]),
    };
    res->blend_enable = PyLong_AsLong(seq[23]);
    res->blend_src_color = PyLong_AsLong(seq[24]);
    res->blend_dst_color = PyLong_AsLong(seq[25]);
    res->blend_src_alpha = PyLong_AsLong(seq[26]);
    res->blend_dst_alpha = PyLong_AsLong(seq[27]);
    res->polygon_offset = PyObject_IsTrue(seq[28]);
    res->polygon_offset_factor = (float)PyFloat_AsDouble(seq[29]);
    res->polygon_offset_units = (float)PyFloat_AsDouble(seq[30]);
    res->attachments = PyLong_AsLong(seq[31]);

    PyDict_SetItem(self->global_settings_cache, settings, (PyObject *)res);
    return res;
}

GLObject * compile_shader(Instance * self, PyObject * code, int type, const char * name) {
    if (GLObject * cache = (GLObject *)PyDict_GetItem(self->shader_cache, code)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    const GLMethods & gl = self->gl;

    int shader = gl.CreateShader(type);
    const char * src = PyBytes_AsString(code);
    gl.ShaderSource(shader, 1, &src, 0);
    gl.CompileShader(shader);

    int shader_compiled = false;
    gl.GetShaderiv(shader, GL_COMPILE_STATUS, &shader_compiled);

    if (!shader_compiled) {
        int log_size = 0;
        gl.GetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_size);
        char * log_text = (char *)malloc(log_size + 1);
        gl.GetShaderInfoLog(shader, log_size, &log_size, log_text);
        log_text[log_size] = 0;
        PyErr_Format(PyExc_ValueError, "%s Error\n\n%s", name, log_text);
        free(log_text);
        return 0;
    }

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = shader;
    res->uses = 1;

    PyDict_SetItem(self->shader_cache, code, (PyObject *)res);
    return res;
}

GLObject * compile_program(Instance * self, PyObject * vert, PyObject * frag, PyObject * layout) {
    const GLMethods & gl = self->gl;

    PyObject * pair = PyObject_CallMethod(self->module_state->helper, "program", "OOOO", vert, frag, layout, self->files);
    if (!pair) {
        return NULL;
    }

    if (GLObject * cache = (GLObject *)PyDict_GetItem(self->program_cache, pair)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    PyObject * vert_code = PyTuple_GetItem(pair, 0);
    PyObject * frag_code = PyTuple_GetItem(pair, 1);

    GLObject * vertex_shader = compile_shader(self, vert_code, GL_VERTEX_SHADER, "Vertex Shader");
    if (!vertex_shader) {
        Py_DECREF(pair);
        return NULL;
    }
    int vertex_shader_obj = vertex_shader->obj;
    Py_DECREF(vertex_shader);

    GLObject * fragment_shader = compile_shader(self, frag_code, GL_FRAGMENT_SHADER, "Fragment Shader");
    if (!fragment_shader) {
        Py_DECREF(pair);
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
        char * log_text = (char *)malloc(log_size + 1);
        gl.GetProgramInfoLog(program, log_size, &log_size, log_text);
        log_text[log_size] = 0;
        Py_DECREF(pair);
        PyErr_Format(PyExc_ValueError, "Linker Error\n\n%s", log_text);
        free(log_text);
        return 0;
    }

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = program;
    res->uses = 1;

    PyDict_SetItem(self->program_cache, pair, (PyObject *)res);
    Py_DECREF(pair);
    return res;
}

Instance * meth_instance(PyObject * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"context", NULL};

    PyObject * context;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "O", keywords, &context)) {
        return NULL;
    }

    ModuleState * module_state = (ModuleState *)PyModule_GetState(self);

    GLMethods gl = load_gl(context);
    if (PyErr_Occurred()) {
        return NULL;
    }

    int max_texture_image_units = 0;
    gl.GetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &max_texture_image_units);
    int default_texture_unit = GL_TEXTURE0 + max_texture_image_units - 1;
    gl.PrimitiveRestartIndex(-1);
    gl.Enable(GL_PROGRAM_POINT_SIZE);
    gl.Enable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    gl.Enable(GL_FRAMEBUFFER_SRGB);

    PyObject * info = PyTuple_New(3);
    PyTuple_SetItem(info, 0, to_str(gl.GetString(GL_VENDOR)));
    PyTuple_SetItem(info, 1, to_str(gl.GetString(GL_RENDERER)));
    PyTuple_SetItem(info, 2, to_str(gl.GetString(GL_VERSION)));

    Instance * res = PyObject_New(Instance, module_state->Instance_type);
    res->module_state = module_state;
    res->descriptor_set_buffers_cache = PyDict_New();
    res->descriptor_set_images_cache = PyDict_New();
    res->global_settings_cache = PyDict_New();
    res->sampler_cache = PyDict_New();
    res->vertex_array_cache = PyDict_New();
    res->framebuffer_cache = PyDict_New();
    res->program_cache = PyDict_New();
    res->shader_cache = PyDict_New();
    res->files = PyDict_New();
    res->info = info;
    res->current_buffers = NULL;
    res->current_images = NULL;
    res->current_global_settings = NULL;
    res->current_framebuffer = 0;
    res->current_program = 0;
    res->current_vertex_array = 0;
    res->viewport = {};
    res->default_texture_unit = default_texture_unit;
    res->mapped_buffers = 0;
    res->gl = gl;
    return res;
}

Buffer * Instance_meth_buffer(Instance * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "size", "dynamic", NULL};

    PyObject * data = Py_None;
    PyObject * size_arg = Py_None;
    int dynamic = true;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|O$Op", keywords, &data, &size_arg, &dynamic)) {
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
    gl.GenBuffers(1, (unsigned *)&buffer);
    gl.BindBuffer(GL_ARRAY_BUFFER, buffer);
    gl.BufferData(GL_ARRAY_BUFFER, size, view.buf, dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);

    Buffer * res = PyObject_New(Buffer, self->module_state->Buffer_type);
    res->instance = (Instance *)new_ref(self);
    res->buffer = buffer;
    res->size = size;
    res->mapped = false;

    if (data != Py_None) {
        PyBuffer_Release(&view);
    }

    Py_INCREF(res);
    return res;
}

Image * Instance_meth_image(Instance * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"size", "format", "data", "samples", "texture", NULL};

    int width;
    int height;
    const char * format_str;
    PyObject * data = Py_None;
    int samples = 1;
    int array = 0;
    PyObject * texture = Py_None;
    int cubemap = false;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "(ii)s|O$iiOp",
        keywords,
        &width,
        &height,
        &format_str,
        &data,
        &samples,
        &array,
        &texture,
        &cubemap
    );

    if (!args_ok) {
        return NULL;
    }

    const GLMethods & gl = self->gl;

    const bool invalid_texture_parameter = texture != Py_True && texture != Py_False && texture != Py_None;
    const bool samples_but_texture = samples > 1 && texture == Py_True;
    const bool cubemap_array = cubemap && array;
    const bool cubemap_or_array_renderbuffer = (array || cubemap) && (samples > 1 || texture == Py_False);

    if (invalid_texture_parameter || samples_but_texture || cubemap_array || cubemap_or_array_renderbuffer) {
        if (invalid_texture_parameter) {
            PyErr_Format(PyExc_TypeError, "invalid texture parameter");
        } else if (samples_but_texture) {
            PyErr_Format(PyExc_TypeError, "for multisampled images texture must be False");
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
        }
        return NULL;
    }

    int renderbuffer = samples > 1 || texture == Py_False;
    int target = cubemap ? GL_TEXTURE_CUBE_MAP : array ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;

    Py_buffer view = {};

    if (data != Py_None) {
        if (PyObject_GetBuffer(data, &view, PyBUF_SIMPLE)) {
            return NULL;
        }
    }

    ImageFormat format = get_image_format(format_str);

    int image = 0;
    if (renderbuffer) {
        gl.GenRenderbuffers(1, (unsigned *)&image);
        gl.BindRenderbuffer(GL_RENDERBUFFER, image);
        gl.RenderbufferStorageMultisample(GL_RENDERBUFFER, samples, format.internal_format, width, height);
    } else {
        gl.GenTextures(1, (unsigned *)&image);
        gl.ActiveTexture(self->default_texture_unit);
        gl.BindTexture(target, image);
        if (cubemap) {
            int stride = width * height * format.pixel_size / 6;
            for (int i = 0; i < 6; ++i) {
                int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i;
                gl.TexImage2D(face, 0, format.internal_format, width, height, 0, format.format, format.type, (char *)view.buf + stride * i);
            }
        } else if (array) {
            gl.TexImage3D(target, 0, format.internal_format, width, height, array, 0, format.format, format.type, view.buf);
        } else {
            gl.TexImage2D(target, 0, format.internal_format, width, height, 0, format.format, format.type, view.buf);
        }
    }

    ClearValue clear_value = {};
    if (format.buffer == GL_DEPTH || format.buffer == GL_DEPTH_STENCIL) {
        clear_value.clear_floats[0] = 1.0f;
    }

    Image * res = PyObject_New(Image, self->module_state->Image_type);
    res->instance = (Instance *)new_ref(self);
    res->size = Py_BuildValue("(ii)", width, height);
    res->clear_value = clear_value;
    res->format = format;
    res->image = image;
    res->width = width;
    res->height = height;
    res->samples = samples;
    res->array = array;
    res->cubemap = cubemap;
    res->target = target;
    res->renderbuffer = renderbuffer;

    res->framebuffer = 0;
    if (!cubemap && !array) {
        if (format.color) {
            PyObject * attachments = Py_BuildValue("((O)O)", res, Py_None);
            res->framebuffer = build_framebuffer(self, attachments);
            Py_DECREF(attachments);
        } else {
            PyObject * attachments = Py_BuildValue("(()O)", res);
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

Renderer * Instance_meth_renderer(Instance * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {
        "vertex_shader",
        "fragment_shader",
        "layout",
        "resources",
        "depth",
        "stencil",
        "blending",
        "polygon_offset",
        "color_mask",
        "framebuffer",
        "vertex_buffers",
        "index_buffer",
        "short_index",
        "primitive_restart",
        "front_face",
        "cull_face",
        "topology",
        "vertex_count",
        "instance_count",
        "first_vertex",
        "line_width",
        "viewport",
        NULL,
    };

    PyObject * vertex_shader = NULL;
    PyObject * fragment_shader = NULL;
    PyObject * layout = self->module_state->empty_tuple;
    PyObject * resources = self->module_state->empty_tuple;
    PyObject * depth = Py_True;
    PyObject * stencil = Py_False;
    PyObject * blending = Py_False;
    PyObject * polygon_offset = Py_False;
    PyObject * color_mask = self->module_state->default_color_mask;
    PyObject * framebuffer_images = NULL;
    PyObject * vertex_buffers = self->module_state->empty_tuple;
    PyObject * index_buffer = Py_None;
    int short_index = false;
    PyObject * primitive_restart = Py_True;
    PyObject * front_face = self->module_state->str_ccw;
    PyObject * cull_face = self->module_state->str_none;
    const char * topology = "triangles";
    int vertex_count = 0;
    int instance_count = 1;
    int first_vertex = 0;
    PyObject * line_width = self->module_state->float_one;
    PyObject * viewport = Py_None;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "|$OOOOOOOOOOOOpOOOsiiiOO",
        keywords,
        &vertex_shader,
        &fragment_shader,
        &layout,
        &resources,
        &depth,
        &stencil,
        &blending,
        &polygon_offset,
        &color_mask,
        &framebuffer_images,
        &vertex_buffers,
        &index_buffer,
        &short_index,
        &primitive_restart,
        &front_face,
        &cull_face,
        &topology,
        &vertex_count,
        &instance_count,
        &first_vertex,
        &line_width,
        &viewport
    );

    if (!args_ok) {
        return NULL;
    }

    if (viewport != Py_None && !is_viewport(viewport)) {
        PyErr_Format(PyExc_TypeError, "the viewport must be a tuple of 4 ints");
        return NULL;
    }

    const GLMethods & gl = self->gl;

    int index_size = short_index ? 2 : 4;
    int index_type = index_buffer != Py_None ? (short_index ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT) : 0;

    GLObject * program = compile_program(self, vertex_shader, fragment_shader, layout);
    if (!program) {
        return NULL;
    }

    int attribs = 0;
    int uniforms = 0;
    int uniform_buffers = 0;
    gl.GetProgramiv(program->obj, GL_ACTIVE_ATTRIBUTES, &attribs);
    gl.GetProgramiv(program->obj, GL_ACTIVE_UNIFORMS, &uniforms);
    gl.GetProgramiv(program->obj, GL_ACTIVE_UNIFORM_BLOCKS, &uniform_buffers);

    PyObject * program_attributes = PyList_New(attribs);
    PyObject * program_uniforms = PyList_New(uniforms);
    PyObject * program_uniform_buffers = PyList_New(uniform_buffers);

    for (int i = 0; i < attribs; ++i) {
        int size = 0;
        int type = 0;
        int length = 0;
        char name[256] = {};
        gl.GetActiveAttrib(program->obj, i, 256, &length, &size, (unsigned *)&type, name);
        int location = gl.GetAttribLocation(program->obj, name);
        PyList_SET_ITEM(program_attributes, i, Py_BuildValue("{sssi}", "name", name, "location", location));
    }

    for (int i = 0; i < uniforms; ++i) {
        int size = 0;
        int type = 0;
        int length = 0;
        char name[256] = {};
        gl.GetActiveUniform(program->obj, i, 256, &length, &size, (unsigned *)&type, name);
        int location = gl.GetUniformLocation(program->obj, name);
        PyList_SET_ITEM(program_uniforms, i, Py_BuildValue("{sssi}", "name", name, "location", location));
    }

    for (int i = 0; i < uniform_buffers; ++i) {
        int size = 0;
        int length = 0;
        char name[256] = {};
        gl.GetActiveUniformBlockiv(program->obj, i, GL_UNIFORM_BLOCK_DATA_SIZE, &size);
        gl.GetActiveUniformBlockName(program->obj, i, 256, &length, name);
        PyList_SET_ITEM(program_uniform_buffers, i, Py_BuildValue("{sssi}", "name", name, "size", size));
    }

    PyObject * validate = PyObject_CallMethod(
        self->module_state->helper,
        "validate",
        "NNNOOO",
        program_attributes,
        program_uniforms,
        program_uniform_buffers,
        vertex_buffers,
        layout,
        resources
    );

    if (!validate) {
        return NULL;
    }

    bind_program(self, program->obj);
    int layout_count = layout != Py_None ? (int)PyList_Size(layout) : 0;
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

    PyObject * attachments = PyObject_CallMethod(self->module_state->helper, "framebuffer_attachments", "(O)", framebuffer_images);
    if (!attachments) {
        return NULL;
    }

    GLObject * framebuffer = build_framebuffer(self, attachments);

    PyObject * bindings = PyObject_CallMethod(self->module_state->helper, "vertex_array_bindings", "OO", vertex_buffers, index_buffer);
    if (!bindings) {
        return NULL;
    }

    GLObject * vertex_array = build_vertex_array(self, bindings);
    Py_DECREF(bindings);

    PyObject * buffer_bindings = PyObject_CallMethod(self->module_state->helper, "buffer_bindings", "(O)", resources);
    if (!buffer_bindings) {
        return NULL;
    }

    DescriptorSetBuffers * descriptor_set_buffers = build_descriptor_set_buffers(self, buffer_bindings);
    Py_DECREF(buffer_bindings);

    PyObject * sampler_bindings = PyObject_CallMethod(self->module_state->helper, "sampler_bindings", "(O)", resources);
    if (!sampler_bindings) {
        return NULL;
    }

    DescriptorSetImages * descriptor_set_images = build_descriptor_set_images(self, sampler_bindings);
    Py_DECREF(sampler_bindings);

    PyObject * settings = PyObject_CallMethod(
        self->module_state->helper,
        "settings",
        "OOOOOOOOON",
        primitive_restart,
        line_width,
        front_face,
        cull_face,
        color_mask,
        depth,
        stencil,
        blending,
        polygon_offset,
        attachments
    );

    if (!settings) {
        return NULL;
    }

    GlobalSettings * global_settings = build_global_settings(self, settings);
    Py_DECREF(settings);

    Viewport viewport_value = {};
    if (viewport != Py_None) {
        viewport_value = to_viewport(viewport);
    } else {
        Image * first_image = (Image *)PySequence_GetItem(framebuffer_images, 0);
        if (!first_image) {
            return NULL;
        }
        viewport_value.width = (short)first_image->width;
        viewport_value.height = (short)first_image->height;
        Py_DECREF(first_image);
    }

    Renderer * res = PyObject_New(Renderer, self->module_state->Renderer_type);
    res->instance = (Instance *)new_ref(self);
    res->framebuffer = framebuffer;
    res->vertex_array = vertex_array;
    res->program = program;
    res->topology = get_topology(topology);
    res->vertex_count = vertex_count;
    res->instance_count = instance_count;
    res->first_vertex = first_vertex;
    res->index_type = index_type;
    res->index_size = index_size;
    res->viewport = viewport_value;
    res->descriptor_set_buffers = descriptor_set_buffers;
    res->descriptor_set_images = descriptor_set_images;
    res->global_settings = global_settings;
    Py_INCREF(res);
    return res;
}

PyObject * Instance_meth_clear_shader_cache(Instance * self) {
    const GLMethods & gl = self->gl;
    PyObject * key = NULL;
    PyObject * value = NULL;
    Py_ssize_t pos = 0;
    while (PyDict_Next(self->shader_cache, &pos, &key, &value)) {
        GLObject * shader = (GLObject *)value;
        gl.DeleteShader(shader->obj);
    }
    PyDict_Clear(self->shader_cache);
    Py_RETURN_NONE;
}

PyObject * Instance_meth_release(Instance * self, PyObject * arg) {
    const GLMethods & gl = self->gl;
    if (Py_TYPE(arg) == self->module_state->Buffer_type) {
        Buffer * buffer = (Buffer *)arg;
        gl.DeleteBuffers(1, (unsigned int *)&buffer->buffer);
        Py_DECREF(arg);
    } else if (Py_TYPE(arg) == self->module_state->Image_type) {
        Image * image = (Image *)arg;
        image->framebuffer->uses -= 1;
        if (!image->framebuffer->uses) {
            remove_dict_value(self->framebuffer_cache, (PyObject *)image->framebuffer);
            gl.DeleteFramebuffers(1, (unsigned int *)&image->framebuffer->obj);
        }
        if (image->renderbuffer) {
            gl.DeleteRenderbuffers(1, (unsigned int *)&image->image);
        } else {
            gl.DeleteTextures(1, (unsigned int *)&image->image);
        }
        Py_DECREF(arg);
    } else if (Py_TYPE(arg) == self->module_state->Renderer_type) {
        Renderer * renderer = (Renderer *)arg;
        renderer->descriptor_set_buffers->uses -= 1;
        if (!renderer->descriptor_set_buffers->uses) {
            remove_dict_value(self->descriptor_set_buffers_cache, (PyObject *)renderer->descriptor_set_buffers);
        }
        renderer->descriptor_set_images->uses -= 1;
        if (!renderer->descriptor_set_images->uses) {
            for (int i = 0; i < renderer->descriptor_set_images->samplers; ++i) {
                GLObject * sampler = renderer->descriptor_set_images->sampler[i];
                sampler->uses -= 1;
                if (!sampler->uses) {
                    remove_dict_value(self->sampler_cache, (PyObject *)sampler);
                    gl.DeleteSamplers(1, (unsigned int *)&sampler->obj);
                }
            }
            remove_dict_value(self->descriptor_set_images_cache, (PyObject *)renderer->descriptor_set_images);
        }
        renderer->global_settings->uses -= 1;
        if (!renderer->global_settings->uses) {
            remove_dict_value(self->global_settings_cache, (PyObject *)renderer->global_settings);
        }
        renderer->framebuffer->uses -= 1;
        if (!renderer->framebuffer->uses) {
            remove_dict_value(self->framebuffer_cache, (PyObject *)renderer->framebuffer);
            gl.DeleteFramebuffers(1, (unsigned int *)&renderer->framebuffer->obj);
        }
        renderer->program->uses -= 1;
        if (!renderer->program->uses) {
            remove_dict_value(self->program_cache, (PyObject *)renderer->program);
            gl.DeleteProgram(renderer->program->obj);
        }
        renderer->vertex_array->uses -= 1;
        if (!renderer->vertex_array->uses) {
            remove_dict_value(self->vertex_array_cache, (PyObject *)renderer->vertex_array);
            gl.DeleteVertexArrays(1, (unsigned int *)&renderer->vertex_array->obj);
        }
        Py_DECREF(renderer);
    }
    Py_RETURN_NONE;
}

PyObject * Buffer_meth_write(Buffer * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "offset", NULL};

    Py_buffer view;
    int offset = 0;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "y*|i", keywords, &view, &offset)) {
        return NULL;
    }

    const bool already_mapped = self->mapped;
    const bool invalid_offset = offset < 0 || offset > self->size;
    const bool invalid_size = (int)view.len > self->size;

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

    const GLMethods & gl = self->instance->gl;

    gl.BindBuffer(GL_ARRAY_BUFFER, self->buffer);
    gl.BufferSubData(GL_ARRAY_BUFFER, offset, (int)view.len, view.buf);

    PyBuffer_Release(&view);
    Py_RETURN_NONE;
}

PyObject * Buffer_meth_map(Buffer * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"size", "offset", "discard", NULL};

    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;
    int discard = false;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|$OOp", keywords, &size_arg, &offset_arg, &discard)) {
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

    const GLMethods & gl = self->instance->gl;

    self->mapped = true;
    self->instance->mapped_buffers += 1;
    const int access = discard ? GL_MAP_READ_BIT | GL_MAP_WRITE_BIT : GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_WRITE_BIT;
    gl.BindBuffer(GL_ARRAY_BUFFER, self->buffer);
    void * ptr = gl.MapBufferRange(GL_ARRAY_BUFFER, offset, size, access);
    return PyMemoryView_FromMemory((char *)ptr, size, PyBUF_WRITE);
}

PyObject * Buffer_meth_unmap(Buffer * self) {
    const GLMethods & gl = self->instance->gl;
    if (self->mapped) {
        self->mapped = false;
        self->instance->mapped_buffers -= 1;
        gl.BindBuffer(GL_ARRAY_BUFFER, self->buffer);
        gl.UnmapBuffer(GL_ARRAY_BUFFER);
    }
    Py_RETURN_NONE;
}

PyObject * Image_meth_clear(Image * self) {
    const GLMethods & gl = self->instance->gl;
    bind_framebuffer(self->instance, self->framebuffer->obj);
    gl.ColorMaski(0, 1, 1, 1, 1);
    gl.DepthMask(1);
    gl.StencilMaskSeparate(GL_FRONT, 0xff);
    if (self->format.clear_type == 'f') {
        self->instance->gl.ClearBufferfv(self->format.buffer, 0, self->clear_value.clear_floats);
    } else if (self->format.clear_type == 'i') {
        self->instance->gl.ClearBufferiv(self->format.buffer, 0, self->clear_value.clear_ints);
    } else if (self->format.clear_type == 'u') {
        self->instance->gl.ClearBufferuiv(self->format.buffer, 0, self->clear_value.clear_uints);
    } else if (self->format.clear_type == 'x') {
        self->instance->gl.ClearBufferfi(self->format.buffer, 0, self->clear_value.clear_floats[0], self->clear_value.clear_ints[1]);
    }
    if (GlobalSettings * settings = self->instance->current_global_settings) {
        gl.ColorMaski(0, settings->color_mask & 1, settings->color_mask & 2, settings->color_mask & 4, settings->color_mask & 8);
        gl.StencilMaskSeparate(GL_FRONT, settings->stencil_front.write_mask);
        gl.DepthMask(settings->depth_write);
    }
    Py_RETURN_NONE;
}

PyObject * Image_meth_write(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "size", "offset", "layer", NULL};

    Py_buffer view;
    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;
    PyObject * layer_arg = Py_None;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "y*|O$OO", keywords, &view, &size_arg, &offset_arg, &layer_arg)) {
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
        size.x = self->width;
        size.y = self->height;
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
    const bool invalid_layer = invalid_layer_type || layer < 0 || (self->cubemap && layer >= 6) || (self->array && layer >= self->array);
    const bool layer_but_simple = !self->cubemap && !self->array && layer;
    const bool invalid_type = !self->format.color || self->samples != 1;

    if (offset_but_no_size || invalid_size || invalid_offset || invalid_layer || layer_but_simple || invalid_type) {
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
        } else if (layer_but_simple) {
            PyErr_Format(PyExc_TypeError, "the image is not layered");
        } else if (!self->format.color) {
            PyErr_Format(PyExc_TypeError, "cannot write to depth or stencil images");
        } else if (self->samples != 1) {
            PyErr_Format(PyExc_TypeError, "cannot write to multisampled images");
        }
        return NULL;
    }

    const GLMethods & gl = self->instance->gl;

    gl.ActiveTexture(self->instance->default_texture_unit);
    gl.BindTexture(self->target, self->image);
    if (self->cubemap) {
        int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer;
        gl.TexSubImage2D(face, 0, offset.x, offset.y, size.x, size.y, self->format.format, self->format.type, view.buf);
    } else if (self->array) {
        gl.TexSubImage3D(self->target, 0, offset.x, offset.y, layer, size.x, size.y, 1, self->format.format, self->format.type, view.buf);
    } else {
        gl.TexSubImage2D(self->target, 0, offset.x, offset.y, size.x, size.y, self->format.format, self->format.type, view.buf);
    }

    PyBuffer_Release(&view);
    Py_RETURN_NONE;
}

PyObject * Image_meth_mipmaps(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"base", "levels", NULL};

    int base = 0;
    PyObject * levels_arg = Py_None;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|$iO", keywords, &base, &levels_arg)) {
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

    const GLMethods & gl = self->instance->gl;
    gl.BindTexture(self->target, self->image);
    gl.TexParameteri(self->target, GL_TEXTURE_BASE_LEVEL, base);
    gl.TexParameteri(self->target, GL_TEXTURE_MAX_LEVEL, base + levels);
    gl.GenerateMipmap(self->target);
    Py_RETURN_NONE;
}

PyObject * Image_meth_read(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"size", "offset", NULL};

    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|O$O", keywords, &size_arg, &offset_arg)) {
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

    const GLMethods & gl = self->instance->gl;

    PyObject * res = PyBytes_FromStringAndSize(NULL, size.x * size.y * self->format.pixel_size);
    bind_framebuffer(self->instance, self->framebuffer->obj);
    gl.ReadPixels(offset.x, offset.y, size.x, size.y, self->format.format, self->format.type, PyBytes_AS_STRING(res));
    return res;
}

PyObject * Image_meth_blit(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"target", "target_viewport", "source_viewport", "filter", "srgb", NULL};

    PyObject * target_arg = Py_None;
    PyObject * target_viewport_arg = Py_None;
    PyObject * source_viewport_arg = Py_None;
    int filter = true;
    int srgb = false;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "|O$OOpp",
        keywords,
        &target_arg,
        &target_viewport_arg,
        &source_viewport_arg,
        &filter,
        &srgb
    );

    if (!args_ok) {
        return NULL;
    }

    const bool invalid_target_type = target_arg != Py_None && Py_TYPE(target_arg) != self->instance->module_state->Image_type;

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

    const bool invalid_target_viewport = invalid_target_viewport_type || (
        target_viewport.x < 0 || target_viewport.y < 0 || target_viewport.width <= 0 || target_viewport.height <= 0 ||
        (target && (target_viewport.x + target_viewport.width > target->width || target_viewport.y + target_viewport.height > target->height))
    );

    const bool invalid_source_viewport = invalid_source_viewport_type || (
        source_viewport.x < 0 || source_viewport.y < 0 || source_viewport.width <= 0 || source_viewport.height <= 0 ||
        source_viewport.x + source_viewport.width > self->width || source_viewport.y + source_viewport.height > self->height
    );

    const bool invalid_target = invalid_target_type && (target->cubemap || target->array || !target->format.color);
    const bool invalid_source = self->cubemap || self->array || !self->format.color;

    const bool error = (
        invalid_target_type || invalid_target_viewport_type || invalid_source_viewport_type ||
        invalid_target_viewport || invalid_source_viewport || invalid_target || invalid_source
    );

    if (error) {
        if (invalid_target_type) {
            PyErr_Format(PyExc_TypeError, "target must be an Image or None");
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
        } else if (!self->format.color) {
            PyErr_Format(PyExc_TypeError, "cannot blit depth or stencil images");
        } else if (target && target->cubemap) {
            PyErr_Format(PyExc_TypeError, "cannot blit to cubemap images");
        } else if (target && target->array) {
            PyErr_Format(PyExc_TypeError, "cannot blit to array images");
        } else if (!target && target->format.color) {
            PyErr_Format(PyExc_TypeError, "cannot blit to depth or stencil images");
        }
        return NULL;
    }

    const GLMethods & gl = self->instance->gl;

    if (!srgb) {
        gl.Disable(GL_FRAMEBUFFER_SRGB);
    }
    gl.ColorMaski(0, 1, 1, 1, 1);
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, self->framebuffer->obj);
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, target ? target->framebuffer->obj : 0);
    gl.BlitFramebuffer(
        source_viewport.x, source_viewport.y, source_viewport.x + source_viewport.width, source_viewport.y + source_viewport.height,
        target_viewport.x, target_viewport.y, target_viewport.x + target_viewport.width, target_viewport.y + target_viewport.height,
        GL_COLOR_BUFFER_BIT, filter ? GL_LINEAR : GL_NEAREST
    );
    gl.BindFramebuffer(GL_FRAMEBUFFER, self->instance->current_framebuffer);
    if (GlobalSettings * settings = self->instance->current_global_settings) {
        gl.ColorMaski(0, settings->color_mask & 1, settings->color_mask & 2, settings->color_mask & 4, settings->color_mask & 8);
    }
    if (!srgb) {
        gl.Enable(GL_FRAMEBUFFER_SRGB);
    }
    Py_RETURN_NONE;
}

PyObject * Image_get_clear_value(Image * self) {
    if (self->format.clear_type == 'x') {
        return Py_BuildValue("fi", self->clear_value.clear_floats[0], self->clear_value.clear_ints[1]);
    }
    if (self->format.components = 1) {
        if (self->format.clear_type == 'f') {
            return PyFloat_FromDouble(self->clear_value.clear_floats[0]);
        } else if (self->format.clear_type == 'i') {
            return PyLong_FromLong(self->clear_value.clear_ints[0]);
        } else if (self->format.clear_type == 'u') {
            return PyLong_FromUnsignedLong(self->clear_value.clear_uints[0]);
        }
    }
    PyObject * res = PyTuple_New(self->format.components);
    for (int i = 0; i < self->format.components; ++i) {
        if (self->format.clear_type == 'f') {
            PyTuple_SetItem(res, i, PyFloat_FromDouble(self->clear_value.clear_floats[i]));
        } else if (self->format.clear_type == 'i') {
            PyTuple_SetItem(res, i, PyLong_FromLong(self->clear_value.clear_ints[i]));
        } else if (self->format.clear_type == 'u') {
            PyTuple_SetItem(res, i, PyLong_FromUnsignedLong(self->clear_value.clear_uints[i]));
        }
    }
    return res;
}

int Image_set_clear_value(Image * self, PyObject * value) {
    ClearValue clear_value = {};
    if (self->format.components == 1) {
        if (self->format.clear_type == 'f' ? !PyFloat_CheckExact(value) : !PyLong_CheckExact(value)) {
            if (self->format.clear_type == 'f') {
                PyErr_Format(PyExc_TypeError, "the clear value must be a float");
            } else {
                PyErr_Format(PyExc_TypeError, "the clear value must be an int");
            }
            return -1;
        }
        if (self->format.clear_type == 'f') {
            clear_value.clear_floats[0] = (float)PyFloat_AsDouble(value);
        } else if (self->format.clear_type == 'i') {
            clear_value.clear_ints[0] = PyLong_AsLong(value);
        } else if (self->format.clear_type == 'u') {
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

    if (size != self->format.components) {
        Py_DECREF(values);
        PyErr_Format(PyExc_ValueError, "invalid clear value size");
        return -1;
    }

    if (self->format.clear_type == 'f') {
        for (int i = 0; i < self->format.components; ++i) {
            clear_value.clear_floats[i] = (float)PyFloat_AsDouble(seq[i]);
        }
    } else if (self->format.clear_type == 'i') {
        for (int i = 0; i < self->format.components; ++i) {
            clear_value.clear_ints[i] = PyLong_AsLong(seq[i]);
        }
    } else if (self->format.clear_type == 'u') {
        for (int i = 0; i < self->format.components; ++i) {
            clear_value.clear_uints[i] = PyLong_AsUnsignedLong(seq[i]);
        }
    } else if (self->format.clear_type == 'x') {
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

PyObject * Renderer_meth_render(Renderer * self) {
    if (self->instance->mapped_buffers) {
        PyErr_Format(PyExc_RuntimeError, "rendering with mapped buffers");
        return NULL;
    }
    const GLMethods & gl = self->instance->gl;
    if (self->viewport.viewport != self->instance->viewport.viewport) {
        gl.Viewport(self->viewport.x, self->viewport.y, self->viewport.width, self->viewport.height);
    }
    bind_global_settings(self->instance, self->global_settings);
    bind_framebuffer(self->instance, self->framebuffer->obj);
    bind_program(self->instance, self->program->obj);
    bind_vertex_array(self->instance, self->vertex_array->obj);
    bind_descriptor_set_buffers(self->instance, self->descriptor_set_buffers);
    bind_descriptor_set_images(self->instance, self->descriptor_set_images);
    if (self->index_type) {
        long long offset = self->first_vertex * self->index_size;
        gl.DrawElementsInstanced(self->topology, self->vertex_count, self->index_type, (void *)offset, self->instance_count);
    } else {
        gl.DrawArraysInstanced(self->topology, self->first_vertex, self->vertex_count, self->instance_count);
    }
    Py_RETURN_NONE;
}

PyObject * Renderer_get_viewport(Renderer * self) {
    return Py_BuildValue("iiii", self->viewport.x, self->viewport.y, self->viewport.width, self->viewport.height);
}

int Renderer_set_viewport(Renderer * self, PyObject * viewport) {
    if (!is_viewport(viewport)) {
        PyErr_Format(PyExc_TypeError, "the viewport must be a tuple of 4 ints");
        return -1;
    }
    self->viewport = to_viewport(viewport);
    return 0;
}

struct vec3 {
    double x, y, z;
};

vec3 operator - (const vec3 & a, const vec3 & b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 normalize(const vec3 & a) {
    const double l = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    return {a.x / l, a.y / l, a.z / l};
}

vec3 cross(const vec3 & a, const vec3 & b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double dot(const vec3 & a, const vec3 & b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

PyObject * meth_camera(PyObject * self, PyObject * args, PyObject * kwargs) {
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
            0.0f, 0.0f, (float)(r3 * t.z - r4), 1.0f,
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

PyObject * meth_rgba(PyObject * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "format", NULL};

    PyObject * data;
    PyObject * format;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "OO!",
        keywords,
        &data,
        &PyUnicode_Type,
        &format
    );

    if (!args_ok) {
        return NULL;
    }

    Py_buffer view = {};
    if (PyObject_GetBuffer(data, &view, PyBUF_SIMPLE)) {
        return NULL;
    }

    PyObject * res = NULL;

    if (!PyUnicode_CompareWithASCIIString(format, "rgba")) {
        res = PyBytes_FromStringAndSize((char *)view.buf, view.len);
    }

    if (!PyUnicode_CompareWithASCIIString(format, "bgr")) {
        uint32_t pixel_count = (uint32_t)view.len / 3;
        res = PyBytes_FromStringAndSize(NULL, pixel_count * 4);
        uint8_t * data = (uint8_t *)PyBytes_AsString(res);
        uint8_t * src = (uint8_t *)view.buf;
        while (pixel_count--) {
            data[0] = src[2];
            data[1] = src[1];
            data[2] = src[0];
            data[3] = 255;
            data += 4;
            src += 3;
        }
    }

    if (!PyUnicode_CompareWithASCIIString(format, "rgb")) {
        uint32_t pixel_count = (uint32_t)view.len / 3;
        res = PyBytes_FromStringAndSize(NULL, pixel_count * 4);
        uint8_t * data = (uint8_t *)PyBytes_AsString(res);
        uint8_t * src = (uint8_t *)view.buf;
        while (pixel_count--) {
            data[0] = src[0];
            data[1] = src[1];
            data[2] = src[2];
            data[3] = 255;
            data += 4;
            src += 3;
        }
    }

    if (!PyUnicode_CompareWithASCIIString(format, "bgra")) {
        uint32_t pixel_count = (uint32_t)view.len / 4;
        res = PyBytes_FromStringAndSize(NULL, view.len);
        uint8_t * data = (uint8_t *)PyBytes_AsString(res);
        uint8_t * src = (uint8_t *)view.buf;
        while (pixel_count--) {
            data[0] = src[2];
            data[1] = src[1];
            data[2] = src[0];
            data[3] = src[3];
            data += 4;
            src += 4;
        }
    }

    if (!PyUnicode_CompareWithASCIIString(format, "lum")) {
        uint32_t pixel_count = (uint32_t)view.len;
        res = PyBytes_FromStringAndSize(NULL, pixel_count * 4);
        uint8_t * data = (uint8_t *)PyBytes_AsString(res);
        uint8_t * src = (uint8_t *)view.buf;
        while (pixel_count--) {
            data[0] = src[0];
            data[1] = src[0];
            data[2] = src[0];
            data[3] = 255;
            data += 4;
            src += 1;
        }
    }

    if (!res) {
        PyBuffer_Release(&view);
        PyErr_Format(PyExc_ValueError, "invalid format");
        return NULL;
    }

    PyBuffer_Release(&view);
    return res;
}

PyObject * meth_pack(PyObject * self, PyObject ** args, Py_ssize_t nargs) {
    if (!nargs) {
        return NULL;
    }
    PyObject * res = PyBytes_FromStringAndSize(NULL, nargs * 4);
    union {
        int * iptr;
        float * fptr;
        void * ptr;
    };
    ptr = PyBytes_AsString(res);
    for (int i = 0; i < nargs; ++i) {
        PyTypeObject * type = Py_TYPE(args[i]);
        if (type == &PyFloat_Type) {
            *fptr++ = (float)PyFloat_AsDouble(args[i]);
        } else if (type == &PyLong_Type) {
            *iptr++ = (int)PyLong_AsLong(args[i]);
        } else {
            Py_DECREF(res);
            PyErr_Format(PyExc_TypeError, "packing invalid type %s", type->tp_name);
            return NULL;
        }
    }
    return res;
}

void Instance_dealloc(Instance * self) {
    Py_DECREF(self->descriptor_set_buffers_cache);
    Py_DECREF(self->descriptor_set_images_cache);
    Py_DECREF(self->global_settings_cache);
    Py_DECREF(self->sampler_cache);
    Py_DECREF(self->vertex_array_cache);
    Py_DECREF(self->framebuffer_cache);
    Py_DECREF(self->program_cache);
    Py_DECREF(self->shader_cache);
    Py_DECREF(self->files);
    Py_DECREF(self->info);
    Py_TYPE(self)->tp_free(self);
}

void Buffer_dealloc(Buffer * self) {
    Py_DECREF(self->instance);
    Py_TYPE(self)->tp_free(self);
}

void Image_dealloc(Image * self) {
    Py_DECREF(self->instance);
    Py_DECREF(self->framebuffer);
    Py_DECREF(self->size);
    Py_TYPE(self)->tp_free(self);
}

void Renderer_dealloc(Renderer * self) {
    Py_DECREF(self->instance);
    Py_DECREF(self->descriptor_set_buffers);
    Py_DECREF(self->descriptor_set_images);
    Py_DECREF(self->global_settings);
    Py_DECREF(self->framebuffer);
    Py_DECREF(self->program);
    Py_DECREF(self->vertex_array);
    Py_TYPE(self)->tp_free(self);
}

void DescriptorSetBuffers_dealloc(DescriptorSetBuffers * self) {
    Py_TYPE(self)->tp_free(self);
}

void DescriptorSetImages_dealloc(DescriptorSetImages * self) {
    Py_TYPE(self)->tp_free(self);
}

void GlobalSettings_dealloc(GlobalSettings * self) {
    Py_TYPE(self)->tp_free(self);
}

void GLObject_dealloc(GLObject * self) {
    Py_TYPE(self)->tp_free(self);
}

PyMethodDef Instance_methods[] = {
    {"buffer", (PyCFunction)Instance_meth_buffer, METH_VARARGS | METH_KEYWORDS, NULL},
    {"image", (PyCFunction)Instance_meth_image, METH_VARARGS | METH_KEYWORDS, NULL},
    {"renderer", (PyCFunction)Instance_meth_renderer, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clear_shader_cache", (PyCFunction)Instance_meth_clear_shader_cache, METH_NOARGS, NULL},
    {"release", (PyCFunction)Instance_meth_release, METH_O, NULL},
    {},
};

PyMemberDef Instance_members[] = {
    {"files", T_OBJECT_EX, offsetof(Instance, files), READONLY, NULL},
    {"info", T_OBJECT_EX, offsetof(Instance, info), READONLY, NULL},
    {},
};

PyMethodDef Buffer_methods[] = {
    {"write", (PyCFunction)Buffer_meth_write, METH_VARARGS | METH_KEYWORDS, NULL},
    {"map", (PyCFunction)Buffer_meth_map, METH_VARARGS | METH_KEYWORDS, NULL},
    {"unmap", (PyCFunction)Buffer_meth_unmap, METH_NOARGS, NULL},
    {},
};

PyMemberDef Buffer_members[] = {
    {"size", T_INT, offsetof(Buffer, size), READONLY, NULL},
    {},
};

PyMethodDef Image_methods[] = {
    {"clear", (PyCFunction)Image_meth_clear, METH_NOARGS, NULL},
    {"write", (PyCFunction)Image_meth_write, METH_VARARGS | METH_KEYWORDS, NULL},
    {"read", (PyCFunction)Image_meth_read, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mipmaps", (PyCFunction)Image_meth_mipmaps, METH_VARARGS | METH_KEYWORDS, NULL},
    {"blit", (PyCFunction)Image_meth_blit, METH_VARARGS | METH_KEYWORDS, NULL},
    {},
};

PyGetSetDef Image_getset[] = {
    {"clear_value", (getter)Image_get_clear_value, (setter)Image_set_clear_value, NULL, NULL},
    {},
};

PyMemberDef Image_members[] = {
    {"size", T_OBJECT_EX, offsetof(Image, size), READONLY, NULL},
    {"samples", T_INT, offsetof(Image, samples), READONLY, NULL},
    {"color", T_BOOL, offsetof(Image, format.color), READONLY, NULL},
    {},
};

PyMethodDef Renderer_methods[] = {
    {"render", (PyCFunction)Renderer_meth_render, METH_NOARGS, NULL},
    {},
};

PyGetSetDef Renderer_getset[] = {
    {"viewport", (getter)Renderer_get_viewport, (setter)Renderer_set_viewport, NULL, NULL},
    {},
};

PyMemberDef Renderer_members[] = {
    {"vertex_count", T_OBJECT_EX, offsetof(Renderer, vertex_count), 0, NULL},
    {"instance_count", T_OBJECT_EX, offsetof(Renderer, instance_count), 0, NULL},
    {"first_vertex", T_OBJECT_EX, offsetof(Renderer, first_vertex), 0, NULL},
    {},
};

PyType_Slot Instance_slots[] = {
    {Py_tp_methods, Instance_methods},
    {Py_tp_members, Instance_members},
    {Py_tp_dealloc, Instance_dealloc},
    {},
};

PyType_Slot Buffer_slots[] = {
    {Py_tp_methods, Buffer_methods},
    {Py_tp_members, Buffer_members},
    {Py_tp_dealloc, Buffer_dealloc},
    {},
};

PyType_Slot Image_slots[] = {
    {Py_tp_methods, Image_methods},
    {Py_tp_getset, Image_getset},
    {Py_tp_members, Image_members},
    {Py_tp_dealloc, Image_dealloc},
    {},
};

PyType_Slot Renderer_slots[] = {
    {Py_tp_methods, Renderer_methods},
    {Py_tp_getset, Renderer_getset},
    {Py_tp_members, Renderer_members},
    {Py_tp_dealloc, Renderer_dealloc},
    {},
};

PyType_Slot DescriptorSetBuffers_slots[] = {
    {Py_tp_dealloc, DescriptorSetBuffers_dealloc},
    {},
};

PyType_Slot DescriptorSetImages_slots[] = {
    {Py_tp_dealloc, DescriptorSetImages_dealloc},
    {},
};

PyType_Slot GlobalSettings_slots[] = {
    {Py_tp_dealloc, GlobalSettings_dealloc},
    {},
};

PyType_Slot GLObject_slots[] = {
    {Py_tp_dealloc, GLObject_dealloc},
    {},
};

PyType_Spec Instance_spec = {"zengl.Instance", sizeof(Instance), 0, Py_TPFLAGS_DEFAULT, Instance_slots};
PyType_Spec Buffer_spec = {"zengl.Buffer", sizeof(Buffer), 0, Py_TPFLAGS_DEFAULT, Buffer_slots};
PyType_Spec Image_spec = {"zengl.Image", sizeof(Image), 0, Py_TPFLAGS_DEFAULT, Image_slots};
PyType_Spec Renderer_spec = {"zengl.Renderer", sizeof(Renderer), 0, Py_TPFLAGS_DEFAULT, Renderer_slots};
PyType_Spec DescriptorSetBuffers_spec = {"zengl.DescriptorSetBuffers", sizeof(DescriptorSetBuffers), 0, Py_TPFLAGS_DEFAULT, DescriptorSetBuffers_slots};
PyType_Spec DescriptorSetImages_spec = {"zengl.DescriptorSetImages", sizeof(DescriptorSetImages), 0, Py_TPFLAGS_DEFAULT, DescriptorSetImages_slots};
PyType_Spec GlobalSettings_spec = {"zengl.GlobalSettings", sizeof(GlobalSettings), 0, Py_TPFLAGS_DEFAULT, GlobalSettings_slots};
PyType_Spec GLObject_spec = {"zengl.GLObject", sizeof(GLObject), 0, Py_TPFLAGS_DEFAULT, GLObject_slots};

int module_exec(PyObject * self) {
    ModuleState * state = (ModuleState *)PyModule_GetState(self);

    state->helper = PyImport_ImportModule("_zengl");
    if (!state->helper) {
        return -1;
    }

    state->empty_tuple = PyTuple_New(0);
    state->str_none = PyUnicode_FromString("none");
    state->str_ccw = PyUnicode_FromString("ccw");
    state->float_one = PyFloat_FromDouble(1.0);
    state->default_color_mask = PyLong_FromUnsignedLongLong(0xffffffffffffffffull);
    state->Instance_type = (PyTypeObject *)PyType_FromSpec(&Instance_spec);
    state->Buffer_type = (PyTypeObject *)PyType_FromSpec(&Buffer_spec);
    state->Image_type = (PyTypeObject *)PyType_FromSpec(&Image_spec);
    state->Renderer_type = (PyTypeObject *)PyType_FromSpec(&Renderer_spec);
    state->DescriptorSetBuffers_type = (PyTypeObject *)PyType_FromSpec(&DescriptorSetBuffers_spec);
    state->DescriptorSetImages_type = (PyTypeObject *)PyType_FromSpec(&DescriptorSetImages_spec);
    state->GlobalSettings_type = (PyTypeObject *)PyType_FromSpec(&GlobalSettings_spec);
    state->GLObject_type = (PyTypeObject *)PyType_FromSpec(&GLObject_spec);

    PyModule_AddObject(self, "Instance", (PyObject *)state->Instance_type);
    PyModule_AddObject(self, "Buffer", (PyObject *)state->Buffer_type);
    PyModule_AddObject(self, "Image", (PyObject *)state->Image_type);
    PyModule_AddObject(self, "Renderer", (PyObject *)state->Renderer_type);

    PyModule_AddObject(self, "context", (PyObject *)new_ref(PyObject_GetAttrString(state->helper, "context")));
    PyModule_AddObject(self, "calcsize", (PyObject *)new_ref(PyObject_GetAttrString(state->helper, "calcsize")));
    PyModule_AddObject(self, "bind", (PyObject *)new_ref(PyObject_GetAttrString(state->helper, "bind")));

    return 0;
}

PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, module_exec},
    {},
};

PyMethodDef module_methods[] = {
    {"instance", (PyCFunction)meth_instance, METH_VARARGS | METH_KEYWORDS, NULL},
    {"camera", (PyCFunction)meth_camera, METH_VARARGS | METH_KEYWORDS, NULL},
    {"rgba", (PyCFunction)meth_rgba, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pack", (PyCFunction)meth_pack, METH_FASTCALL, NULL},
    {},
};

void module_free(PyObject * self) {
    ModuleState * state = (ModuleState *)PyModule_GetState(self);
    if (!state) {
        return;
    }
    Py_DECREF(state->empty_tuple);
    Py_DECREF(state->str_none);
    Py_DECREF(state->str_ccw);
    Py_DECREF(state->float_one);
    Py_DECREF(state->default_color_mask);
    Py_DECREF(state->Instance_type);
    Py_DECREF(state->Buffer_type);
    Py_DECREF(state->Image_type);
    Py_DECREF(state->Renderer_type);
    Py_DECREF(state->DescriptorSetBuffers_type);
    Py_DECREF(state->DescriptorSetImages_type);
    Py_DECREF(state->GlobalSettings_type);
    Py_DECREF(state->GLObject_type);
}

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "zengl", NULL, sizeof(ModuleState), module_methods, module_slots, NULL, NULL, (freefunc)module_free,
};

extern "C" PyObject * PyInit_zengl() {
    return PyModuleDef_Init(&module_def);
}
