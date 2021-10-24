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
};

struct GlobalSettings {
    PyObject_HEAD
    unsigned long long color_mask;
    int primitive_restart;
    int stencil_test;
    int depth_test;
    int depth_write;
    int depth_func;
    int cull_face;
    int front_face;
    float line_width;
    float point_size;
    int polygon_offset;
    float polygon_offset_factor;
    float polygon_offset_units;
    int blend_enable;
    int blend_src_color;
    int blend_dst_color;
    int blend_src_alpha;
    int blend_dst_alpha;
    StencilSettings stencil_front;
    StencilSettings stencil_back;
    int attachments;
};

struct Instance {
    PyObject_HEAD
    ModuleState * module_state;
    PyObject * descriptor_set_buffers_cache;
    PyObject * descriptor_set_images_cache;
    PyObject * settings_cache;
    PyObject * sampler_cache;
    PyObject * vertex_array_cache;
    PyObject * framebuffer_cache;
    PyObject * shader_cache;
    PyObject * files;
    PyObject * info;
    DescriptorSetBuffers * current_buffers;
    DescriptorSetImages * current_images;
    GlobalSettings * current_global_settings;
    int current_framebuffer;
    int current_program;
    int current_vertex_array;
    int viewport_width;
    int viewport_height;
    int default_texture_unit;
    GLMethods gl;
};

struct Buffer {
    PyObject_HEAD
    Instance * instance;
    int buffer;
    int size;
};

struct Image {
    PyObject_HEAD
    Instance * instance;
    PyObject * size;
    ImageFormat format;
    int image;
    int framebuffer;
    int width;
    int height;
    int array;
    int cubemap;
    int target;
    int samples;
    int renderbuffer;
};

struct Renderer {
    PyObject_HEAD
    Instance * instance;
    int framebuffer;
    int vertex_array;
    int program;
    int topology;
    int vertex_count;
    int instance_count;
    int index_type;
    int framebuffer_width;
    int framebuffer_height;
    DescriptorSetBuffers * descriptor_set_buffers;
    DescriptorSetImages * descriptor_set_images;
    GlobalSettings * global_settings;
};

void bind_descriptor_set_buffers(Instance * self, DescriptorSetBuffers * set) {
    const GLMethods & gl = self->gl;
    if (self->current_buffers != set) {
        self->current_buffers = set;
        for (int i = 0; i < set->buffers; ++i) {
            gl.BindBufferRange(
                GL_UNIFORM_BUFFER,
                i,
                set->buffer[i].buffer,
                set->buffer[i].offset,
                set->buffer[i].size
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
            gl.BindTexture(set->sampler[i].target, set->sampler[i].image);
            gl.BindSampler(i, set->sampler[i].sampler);
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
    gl.PointSize(settings->point_size);
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

int build_framebuffer(Instance * self, PyObject * attachments) {
    if (PyObject * cache = PyDict_GetItem(self->framebuffer_cache, attachments)) {
        return PyLong_AsLong(cache);
    }

    const GLMethods & gl = self->gl;

    int framebuffer = 0;
    gl.GenFramebuffers(1, (unsigned *)&framebuffer);
    bind_framebuffer(self, framebuffer);
    int attachment_count = (int)PyTuple_Size(attachments);
    for (int i = 0; i < attachment_count; ++i) {
        Image * image = (Image *)PyTuple_GetItem(attachments, i);
        if (image->renderbuffer) {
            if (image->format.attachment == GL_COLOR_ATTACHMENT0) {
                gl.FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_RENDERBUFFER, image->image);
            } else {
                gl.FramebufferRenderbuffer(GL_FRAMEBUFFER, image->format.attachment, GL_RENDERBUFFER, image->image);
            }
        } else {
            if (image->format.attachment == GL_COLOR_ATTACHMENT0) {
                gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, image->image, 0);
            } else {
                gl.FramebufferTexture2D(GL_FRAMEBUFFER, image->format.attachment, GL_TEXTURE_2D, image->image, 0);
            }
        }
    }

    unsigned int draw_buffers[MAX_ATTACHMENTS];
    for (int i = 0; i < attachment_count; ++i) {
        draw_buffers[i] = GL_COLOR_ATTACHMENT0 + i;
    }

    gl.DrawBuffers(attachment_count, draw_buffers);
    gl.ReadBuffer(GL_COLOR_ATTACHMENT0);

    PyDict_SetItem(self->framebuffer_cache, attachments, PyLong_FromLong(framebuffer));
    return framebuffer;
}

int build_vertex_array(Instance * self, PyObject * bindings) {
    if (PyObject * cache = PyDict_GetItem(self->framebuffer_cache, bindings)) {
        return PyLong_AsLong(cache);
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

    PyDict_SetItem(self->framebuffer_cache, bindings, PyLong_FromLong(vertex_array));
    return vertex_array;
}

int build_sampler(Instance * self, PyObject * params) {
    if (PyObject * cache = PyDict_GetItem(self->sampler_cache, params)) {
        return PyLong_AsLong(cache);
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

    PyDict_SetItem(self->sampler_cache, params, PyLong_FromLong(sampler));
    return sampler;
}

DescriptorSetBuffers * build_descriptor_set_buffers(Instance * self, PyObject * bindings) {
    if (PyObject * cache = PyDict_GetItem(self->descriptor_set_buffers_cache, bindings)) {
        return (DescriptorSetBuffers *)cache;
    }

    const GLMethods & gl = self->gl;

    int length = (int)PyTuple_Size(bindings);
    PyObject ** seq = PySequence_Fast_ITEMS(bindings);

    DescriptorSetBuffers * res = PyObject_New(DescriptorSetBuffers, self->module_state->DescriptorSetBuffers_type);
    memset(res->buffer, 0, sizeof(res->buffer));
    res->buffers = 0;

    for (int i = 0; i < length; i += 2) {
        int binding = PyLong_AsLong(seq[i + 0]);
        Buffer * buffer = (Buffer *)seq[i + 1];
        res->buffer[binding] = {buffer->buffer, 0, buffer->size};
        res->buffers = res->buffers > (binding + 1) ? res->buffers : (binding + 1);
    }

    PyDict_SetItem(self->framebuffer_cache, bindings, (PyObject *)res);
    return res;
}

DescriptorSetImages * build_descriptor_set_images(Instance * self, PyObject * bindings) {
    if (PyObject * cache = PyDict_GetItem(self->descriptor_set_images_cache, bindings)) {
        return (DescriptorSetImages *)cache;
    }

    const GLMethods & gl = self->gl;

    int length = (int)PyTuple_Size(bindings);
    PyObject ** seq = PySequence_Fast_ITEMS(bindings);

    DescriptorSetImages * res = PyObject_New(DescriptorSetImages, self->module_state->DescriptorSetImages_type);
    memset(res->sampler, 0, sizeof(res->sampler));
    res->samplers = 0;

    for (int i = 0; i < length; i += 3) {
        int binding = PyLong_AsLong(seq[i + 0]);
        Image * image = (Image *)seq[i + 1];
        int sampler = build_sampler(self, seq[i + 2]);
        res->sampler[binding] = {sampler, image->target, image->image};
        res->samplers = res->samplers > (binding + 1) ? res->samplers : (binding + 1);
    }

    PyDict_SetItem(self->framebuffer_cache, bindings, (PyObject *)res);
    return res;
}

GlobalSettings * build_global_settings(Instance * self, PyObject * settings) {
    if (PyObject * cache = PyDict_GetItem(self->settings_cache, settings)) {
        return (GlobalSettings *)cache;
    }

    PyObject ** seq = PySequence_Fast_ITEMS(settings);

    GlobalSettings * res = PyObject_New(GlobalSettings, self->module_state->GlobalSettings_type);

    res->primitive_restart = PyObject_IsTrue(seq[0]);
    res->point_size = (float)PyFloat_AsDouble(seq[1]);
    res->line_width = (float)PyFloat_AsDouble(seq[2]);
    res->front_face = PyLong_AsLong(seq[3]);
    res->cull_face = PyLong_AsLong(seq[4]);
    res->color_mask = PyLong_AsUnsignedLongLong(seq[5]);
    res->depth_test = PyObject_IsTrue(seq[6]);
    res->depth_write = PyObject_IsTrue(seq[7]);
    res->depth_func = PyLong_AsLong(seq[8]);
    res->stencil_test = PyObject_IsTrue(seq[9]);
    res->stencil_front = {
        PyLong_AsLong(seq[10]),
        PyLong_AsLong(seq[11]),
        PyLong_AsLong(seq[12]),
        PyLong_AsLong(seq[13]),
        PyLong_AsLong(seq[14]),
        PyLong_AsLong(seq[15]),
        PyLong_AsLong(seq[16]),
    };
    res->stencil_back = {
        PyLong_AsLong(seq[17]),
        PyLong_AsLong(seq[18]),
        PyLong_AsLong(seq[19]),
        PyLong_AsLong(seq[20]),
        PyLong_AsLong(seq[21]),
        PyLong_AsLong(seq[22]),
        PyLong_AsLong(seq[23]),
    };
    res->blend_enable = PyLong_AsLong(seq[24]);
    res->blend_src_color = PyLong_AsLong(seq[25]);
    res->blend_dst_color = PyLong_AsLong(seq[26]);
    res->blend_src_alpha = PyLong_AsLong(seq[27]);
    res->blend_dst_alpha = PyLong_AsLong(seq[28]);
    res->polygon_offset = PyObject_IsTrue(seq[29]);
    res->polygon_offset_factor = (float)PyFloat_AsDouble(seq[30]);
    res->polygon_offset_units = (float)PyFloat_AsDouble(seq[31]);
    res->attachments = PyLong_AsLong(seq[32]);

    PyDict_SetItem(self->settings_cache, settings, (PyObject *)res);
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
    gl.PrimitiveRestartIndex(-1);

    PyObject * info = PyTuple_New(3);
    PyTuple_SetItem(info, 0, to_str(gl.GetString(GL_VENDOR)));
    PyTuple_SetItem(info, 1, to_str(gl.GetString(GL_RENDERER)));
    PyTuple_SetItem(info, 2, to_str(gl.GetString(GL_VERSION)));

    Instance * res = PyObject_New(Instance, module_state->Instance_type);
    res->module_state = module_state;
    res->default_texture_unit = GL_TEXTURE0 + max_texture_image_units - 1;
    res->descriptor_set_buffers_cache = PyDict_New();
    res->descriptor_set_images_cache = PyDict_New();
    res->settings_cache = PyDict_New();
    res->sampler_cache = PyDict_New();
    res->vertex_array_cache = PyDict_New();
    res->framebuffer_cache = PyDict_New();
    res->shader_cache = PyDict_New();
    res->files = PyDict_New();
    res->info = info;
    res->current_buffers = NULL;
    res->current_images = NULL;
    res->current_global_settings = NULL;
    res->current_framebuffer = 0;
    res->current_program = 0;
    res->current_vertex_array = 0;
    res->viewport_width = -1;
    res->viewport_height = -1;
    res->gl = gl;
    return res;
}

Buffer * Instance_meth_buffer(Instance * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "size", "dynamic", NULL};

    PyObject * data = Py_None;
    int size = -1;
    int dynamic = true;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|O$ip", keywords, &data, &size, &dynamic)) {
        return NULL;
    }

    const GLMethods & gl = self->gl;

    Py_buffer view = {};

    if (data != Py_None) {
        if (PyObject_GetBuffer(data, &view, PyBUF_SIMPLE)) {
            return NULL;
        }
        size = (int)view.len;
    }

    if (size < 0) {
        return NULL;
    }

    int buffer = 0;
    gl.GenBuffers(1, (unsigned *)&buffer);
    gl.BindBuffer(GL_ARRAY_BUFFER, buffer);
    gl.BufferData(GL_ARRAY_BUFFER, size, view.buf, dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);

    Buffer * res = PyObject_New(Buffer, self->module_state->Buffer_type);
    res->instance = self;
    res->buffer = buffer;
    res->size = size;

    if (data != Py_None) {
        PyBuffer_Release(&view);
    }

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

    if ((texture != Py_True && texture != Py_False && texture != Py_None) || (samples > 1 && texture == Py_True)) {
        return NULL;
    }

    if ((samples > 1 || texture == Py_False) && (array || cubemap)) {
        return NULL;
    }

    if (cubemap && array) {
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

    Image * res = PyObject_New(Image, self->module_state->Image_type);
    res->instance = self;
    res->size = Py_BuildValue("(ii)", width, height);
    res->format = format;
    res->image = image;
    res->width = width;
    res->height = height;
    res->array = array;
    res->cubemap = cubemap;
    res->target = target;
    res->renderbuffer = renderbuffer;
    res->samples = samples;

    res->framebuffer = 0;
    if (!cubemap && !array) {
        PyObject * attachments = PyTuple_Pack(1, res);
        res->framebuffer = build_framebuffer(self, attachments);
    }

    if (data != Py_None) {
        PyBuffer_Release(&view);
    }

    return res;
}

int compile_shader(Instance * self, PyObject * code, int type, const char * name) {
    if (PyObject * cache = PyDict_GetItem(self->shader_cache, code)) {
        return PyLong_AsLong(cache);
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
        PyErr_Format(PyExc_Exception, "%s Error\n\n%s", name, log_text);
        free(log_text);
        return 0;
    }

    return shader;
}

int compile_program(Instance * self, PyObject * vert, PyObject * frag) {
    const GLMethods & gl = self->gl;

    PyObject * pair = PyObject_CallMethod(self->module_state->helper, "normalize_shaders", "OOO", vert, frag, self->files);
    if (!pair) {
        return 0;
    }

    PyObject * vert_code = PyTuple_GetItem(pair, 0);
    PyObject * frag_code = PyTuple_GetItem(pair, 1);

    if (PyObject * cache = PyDict_GetItem(self->shader_cache, pair)) {
        int program = PyLong_AsLong(cache);
        PyDict_SetItem(self->shader_cache, pair, PyLong_FromLong(program));
        Py_DECREF(pair);
        return program;
    }

    int vertex_shader = compile_shader(self, vert_code, GL_VERTEX_SHADER, "Vertex Shader");
    if (!vertex_shader) {
        Py_DECREF(pair);
        return 0;
    }

    PyDict_SetItem(self->shader_cache, vert_code, PyLong_FromLong(vertex_shader));

    int fragment_shader = compile_shader(self, frag_code, GL_FRAGMENT_SHADER, "Fragment Shader");
    if (!fragment_shader) {
        Py_DECREF(pair);
        return 0;
    }

    PyDict_SetItem(self->shader_cache, frag_code, PyLong_FromLong(fragment_shader));

    int program = gl.CreateProgram();
    gl.AttachShader(program, vertex_shader);
    gl.AttachShader(program, fragment_shader);
    gl.LinkProgram(program);

    int linked = false;
    gl.GetProgramiv(program, GL_LINK_STATUS, &linked);

    if (!linked) {
        int log_size = 0;
        gl.GetProgramiv(program, GL_INFO_LOG_LENGTH, &log_size);
        char * log_text = (char *)malloc(log_size + 1);
        gl.GetProgramInfoLog(program, log_size, &log_size, log_text);
        log_text[log_size] = 0;
        PyErr_Format(PyExc_Exception, "Linker Error\n\n%s", log_text);
        free(log_text);
        return 0;
    }

    PyDict_SetItem(self->shader_cache, pair, PyLong_FromLong(program));
    Py_DECREF(pair);
    return program;
}

Renderer * Instance_meth_renderer(Instance * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {
        "vertex_shader",
        "fragment_shader",
        "framebuffer",
        "topology",
        "layout",
        "resources",
        "vertex_buffers",
        "index_buffer",
        "vertex_count",
        "instance_count",
        "short_index",
        "primitive_restart",
        "point_size",
        "line_width",
        "front_face",
        "cull_face",
        "color_mask",
        "depth",
        "stencil",
        "blending",
        "polygon_offset",
        NULL,
    };

    PyObject * vertex_shader = NULL;
    PyObject * fragment_shader = NULL;
    PyObject * framebuffer_images = NULL;
    const char * topology = "triangles";
    PyObject * layout = self->module_state->empty_tuple;
    PyObject * resources = self->module_state->empty_tuple;
    PyObject * vertex_buffers = self->module_state->empty_tuple;
    PyObject * index_buffer = Py_None;
    int vertex_count = 0;
    int instance_count = 1;
    int short_index = false;
    PyObject * primitive_restart = Py_False;
    PyObject * point_size = self->module_state->float_one;
    PyObject * line_width = self->module_state->float_one;
    PyObject * front_face = self->module_state->str_ccw;
    PyObject * cull_face = self->module_state->str_none;
    PyObject * color_mask = self->module_state->default_color_mask;
    PyObject * depth = Py_True;
    PyObject * stencil = Py_False;
    PyObject * blending = Py_False;
    PyObject * polygon_offset = Py_False;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "|$OOOsOOOOiipOOOOOOOOOO",
        keywords,
        &vertex_shader,
        &fragment_shader,
        &framebuffer_images,
        &topology,
        &layout,
        &resources,
        &vertex_buffers,
        &index_buffer,
        &vertex_count,
        &instance_count,
        &short_index,
        &primitive_restart,
        &point_size,
        &line_width,
        &front_face,
        &cull_face,
        &color_mask,
        &depth,
        &stencil,
        &blending,
        &polygon_offset
    );

    if (!args_ok) {
        return NULL;
    }

    const GLMethods & gl = self->gl;

    int program = compile_program(self, vertex_shader, fragment_shader);
    if (!program) {
        return NULL;
    }

    int attribs = 0;
    int uniforms = 0;
    int uniform_buffers = 0;
    gl.GetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &attribs);
    gl.GetProgramiv(program, GL_ACTIVE_UNIFORMS, &uniforms);
    gl.GetProgramiv(program, GL_ACTIVE_UNIFORM_BLOCKS, &uniform_buffers);

    PyObject * program_attributes = PyList_New(attribs);
    PyObject * program_uniforms = PyList_New(uniforms);
    PyObject * program_uniform_buffers = PyList_New(uniform_buffers);

    for (int i = 0; i < attribs; ++i) {
        int size = 0;
        int type = 0;
        int length = 0;
        char name[256] = {};
        gl.GetActiveAttrib(program, i, 256, &length, &size, (unsigned *)&type, name);
        int location = gl.GetAttribLocation(program, name);
        PyList_SET_ITEM(program_attributes, i, Py_BuildValue("{sssi}", "name", name, "location", location));
    }

    for (int i = 0; i < uniforms; ++i) {
        int size = 0;
        int type = 0;
        int length = 0;
        char name[256] = {};
        gl.GetActiveUniform(program, i, 256, &length, &size, (unsigned *)&type, name);
        int location = gl.GetUniformLocation(program, name);
        PyList_SET_ITEM(program_uniforms, i, Py_BuildValue("{sssi}", "name", name, "location", location));
    }

    for (int i = 0; i < uniform_buffers; ++i) {
        int size = 0;
        int length = 0;
        char name[256] = {};
        gl.GetActiveUniformBlockiv(program, i, GL_UNIFORM_BLOCK_DATA_SIZE, &size);
        gl.GetActiveUniformBlockName(program, i, 256, &length, name);
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

    bind_program(self, program);
    int layout_count = layout != Py_None ? (int)PyList_Size(layout) : 0;
    for (int i = 0; i < layout_count; ++i) {
        PyObject * obj = PyList_GetItem(layout, i);
        PyObject * name = PyDict_GetItemString(obj, "name");
        int binding = PyLong_AsLong(PyDict_GetItemString(obj, "binding"));
        int location = gl.GetUniformLocation(program, PyUnicode_AsUTF8(name));
        if (location >= 0) {
            gl.Uniform1i(location, binding);
        } else {
            int index = gl.GetUniformBlockIndex(program, PyUnicode_AsUTF8(name));
            gl.UniformBlockBinding(program, index, binding);
        }
    }

    PyObject * attachments = PySequence_Tuple(framebuffer_images);
    if (!attachments) {
        return NULL;
    }

    int framebuffer = build_framebuffer(self, attachments);

    PyObject * bindings = PyObject_CallMethod(self->module_state->helper, "vertex_array_bindings", "OO", vertex_buffers, index_buffer);
    if (!bindings) {
        return NULL;
    }

    int vertex_array = build_vertex_array(self, bindings);

    PyObject * buffer_bindings = PyObject_CallMethod(self->module_state->helper, "buffer_bindings", "(O)", resources);
    if (!buffer_bindings) {
        return NULL;
    }

    PyObject * sampler_bindings = PyObject_CallMethod(self->module_state->helper, "sampler_bindings", "(O)", resources);
    if (!sampler_bindings) {
        return NULL;
    }

    PyObject * settings = PyObject_CallMethod(
        self->module_state->helper,
        "settings",
        "OOOOOOOOOOO",
        primitive_restart,
        point_size,
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

    DescriptorSetBuffers * descriptor_set_buffers = build_descriptor_set_buffers(self, buffer_bindings);
    DescriptorSetImages * descriptor_set_images = build_descriptor_set_images(self, sampler_bindings);
    GlobalSettings * global_settings = build_global_settings(self, settings);

    Image * first_image = (Image *)PyTuple_GetItem(attachments, 0);

    Renderer * res = PyObject_New(Renderer, self->module_state->Renderer_type);
    res->instance = self;
    res->framebuffer = framebuffer;
    res->vertex_array = vertex_array;
    res->program = program;
    res->topology = get_topology(topology);
    res->vertex_count = vertex_count;
    res->instance_count = instance_count;
    res->descriptor_set_buffers = descriptor_set_buffers;
    res->descriptor_set_images = descriptor_set_images;
    res->global_settings = global_settings;
    res->index_type = index_buffer != Py_None ? (short_index ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT) : 0;
    res->framebuffer_width = first_image->width;
    res->framebuffer_height = first_image->height;
    return res;
}

PyObject * Buffer_meth_write(Buffer * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "offset", NULL};

    Py_buffer view;
    int offset = 0;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "y*|i", keywords, &view, &offset)) {
        return NULL;
    }

    if (offset < 0 || (int)view.len + offset > self->size) {
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

    int size = -1;
    int offset = 0;
    int discard = false;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|$iip", keywords, &size, &offset, &discard)) {
        return NULL;
    }

    if (size < 0) {
        size = self->size - offset;
    }

    if (offset < 0 || size <= 0 || size + offset > self->size) {
        return NULL;
    }

    const GLMethods & gl = self->instance->gl;

    const int access = discard ? GL_MAP_READ_BIT | GL_MAP_WRITE_BIT : GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_WRITE_BIT;
    gl.BindBuffer(GL_ARRAY_BUFFER, self->buffer);
    void * ptr = gl.MapBufferRange(GL_ARRAY_BUFFER, offset, size, access);
    return PyMemoryView_FromMemory((char *)ptr, size, PyBUF_WRITE);
}

PyObject * Buffer_meth_unmap(Buffer * self) {
    const GLMethods & gl = self->instance->gl;
    gl.BindBuffer(GL_ARRAY_BUFFER, self->buffer);
    gl.UnmapBuffer(GL_ARRAY_BUFFER);
    Py_RETURN_NONE;
}

PyObject * Image_meth_clear(Image * self, PyObject ** args, Py_ssize_t nargs) {
    float clear_color[4] = {};
    float clear_depth = 1.0f;
    int clear_stencil = 0;
    if (nargs) {
        if (nargs != self->format.components) {
            return NULL;
        }
        if (self->format.attachment == GL_COLOR_ATTACHMENT0) {
            for (int i = 0; i < self->format.components; ++i) {
                clear_color[i] = (float)PyFloat_AsDouble(args[i]);
            }
        } else {
            clear_depth = (float)PyFloat_AsDouble(args[0]);
            if (self->format.components == 2) {
                clear_stencil = PyLong_AsLong(args[1]);
            }
        }
    }

    const GLMethods & gl = self->instance->gl;
    bind_framebuffer(self->instance, self->framebuffer);
    if (self->format.attachment == GL_COLOR_ATTACHMENT0) {
        gl.ColorMaski(0, 1, 1, 1, 1);
        gl.ClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);
        gl.Clear(GL_COLOR_BUFFER_BIT);
        if (GlobalSettings * settings = self->instance->current_global_settings) {
            gl.ColorMaski(0, settings->color_mask & 1, settings->color_mask & 2, settings->color_mask & 4, settings->color_mask & 8);
        }
    } else if (self->format.attachment == GL_DEPTH_ATTACHMENT) {
        gl.DepthMask(1);
        gl.ClearDepth(clear_depth);
        gl.Clear(GL_DEPTH_BUFFER_BIT);
        if (GlobalSettings * settings = self->instance->current_global_settings) {
            gl.DepthMask(settings->depth_write);
        }
    } else if (self->format.attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
        gl.DepthMask(1);
        gl.StencilMaskSeparate(GL_FRONT, 0xff);
        gl.ClearDepth(clear_depth);
        gl.ClearStencil(clear_stencil);
        gl.Clear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        if (GlobalSettings * settings = self->instance->current_global_settings) {
            gl.StencilMaskSeparate(GL_FRONT, settings->stencil_front.write_mask);
            gl.DepthMask(settings->depth_write);
        }
    } else if (self->format.attachment == GL_STENCIL_ATTACHMENT) {
        gl.StencilMaskSeparate(GL_FRONT, 0xff);
        gl.ClearStencil(clear_stencil);
        gl.Clear(GL_STENCIL_BUFFER_BIT);
        if (GlobalSettings * settings = self->instance->current_global_settings) {
            gl.StencilMaskSeparate(GL_FRONT, settings->stencil_front.write_mask);
        }
    }
    Py_RETURN_NONE;
}

PyObject * Image_meth_write(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "size", "offset", "layer", NULL};

    Py_buffer view;
    int width = -1;
    int height = -1;
    int x = 0;
    int y = 0;
    int layer = 0;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "y*|(ii)$(ii)i", keywords, &view, &width, &height, &x, &y, &layer)) {
        return NULL;
    }

    if (width == -1 && height == -1) {
        width = self->width - x;
        height = self->height - y;
    }

    if (x < 0 || y < 0 || width <= 0 || height <= 0 || width + x > self->width || height + y > self->height) {
        return NULL;
    }

    if (layer < 0 || (self->cubemap && layer >= 6) || (self->array && layer >= self->array)) {
        return NULL;
    }

    if (!self->cubemap && !self->array && layer) {
        return NULL;
    }

    const GLMethods & gl = self->instance->gl;

    gl.ActiveTexture(self->instance->default_texture_unit);
    gl.BindTexture(self->target, self->image);
    if (self->cubemap) {
        int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer;
        gl.TexSubImage2D(face, 0, x, y, width, height, self->format.format, self->format.type, view.buf);
    } else if (self->array) {
        gl.TexSubImage3D(self->target, 0, x, y, layer, width, height, 1, self->format.format, self->format.type, view.buf);
    } else {
        gl.TexSubImage2D(self->target, 0, x, y, width, height, self->format.format, self->format.type, view.buf);
    }

    PyBuffer_Release(&view);
    Py_RETURN_NONE;
}

PyObject * Image_meth_mipmaps(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"base", "levels", NULL};

    int base = 0;
    int levels = -1;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|(ii)", keywords, &base, &levels)) {
        return NULL;
    }

    if (base < 0) {
        return NULL;
    }

    if (levels < 0) {
        levels = count_mipmaps(self->width, self->height);
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

    int width = -1;
    int height = -1;
    int x = 0;
    int y = 0;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "|(ii)$(ii)", keywords, &width, &height, &x, &y)) {
        return NULL;
    }

    if (width == -1 && height == -1) {
        width = self->width - x;
        height = self->height - y;
    }

    if (x < 0 || y < 0 || width < 0 || height < 0 || width + x > self->width || height + y > self->height) {
        return NULL;
    }

    if (self->cubemap || self->array) {
        return NULL;
    }

    if (self->samples != 1) {
        PyErr_Format(PyExc_ValueError, "Cannot read multisample images");
        return NULL;
    }

    const GLMethods & gl = self->instance->gl;

    PyObject * res = PyBytes_FromStringAndSize(NULL, width * height * self->format.pixel_size);
    bind_framebuffer(self->instance, self->framebuffer);
    gl.ReadPixels(x, y, width, height, self->format.format, self->format.type, PyBytes_AS_STRING(res));
    return res;
}

PyObject * Image_meth_blit(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"dst", "dst_size", "dst_offset", "src_size", "src_offset", "filter", NULL};

    PyObject * dst = Py_None;
    int dst_width = -1;
    int dst_height = -1;
    int dst_x = 0;
    int dst_y = 0;
    int src_width = -1;
    int src_height = -1;
    int src_x = 0;
    int src_y = 0;
    int filter = true;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "|O$(ii)(ii)(ii)(ii)p",
        keywords,
        &dst,
        &dst_width,
        &dst_height,
        &dst_x,
        &dst_y,
        &src_width,
        &src_height,
        &src_x,
        &src_y,
        &filter
    );

    if (!args_ok) {
        return NULL;
    }

    if (dst != Py_None && Py_TYPE(dst) != self->instance->module_state->Image_type) {
        return NULL;
    }

    if (self->cubemap || self->array) {
        return NULL;
    }

    Image * dst_image = dst != Py_None ? (Image *)dst : NULL;

    if (src_width == -1 && src_height == -1) {
        src_width = self->width - src_x;
        src_height = self->height - src_y;
    }

    if (dst_width == -1 && dst_height == -1) {
        if (dst_image) {
            dst_width = dst_image->width - dst_x;
            dst_height = dst_image->height - dst_y;
        } else {
            dst_width = src_width;
            dst_height = src_height;
        }
    }

    if (src_x < 0 || src_y < 0 || dst_x < 0 || dst_y < 0 || src_width < 0 || src_height < 0 || dst_width < 0 || dst_height < 0) {
        return NULL;
    }

    if (src_width + src_x > self->width || src_height + src_y > self->height) {
        return NULL;
    }

    if (dst_image && (dst_width + dst_x > dst_image->width || dst_height + dst_y > dst_image->height)) {
        return NULL;
    }

    const GLMethods & gl = self->instance->gl;

    gl.ColorMaski(0, 1, 1, 1, 1);
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, self->framebuffer);
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, dst_image ? dst_image->framebuffer : 0);
    gl.BlitFramebuffer(src_x, src_y, src_width, src_height, dst_x, dst_y, dst_width, dst_height, GL_COLOR_BUFFER_BIT, filter ? GL_LINEAR : GL_NEAREST);
    gl.BindFramebuffer(GL_FRAMEBUFFER, self->instance->current_framebuffer);
    if (GlobalSettings * settings = self->instance->current_global_settings) {
        gl.ColorMaski(0, settings->color_mask & 1, settings->color_mask & 2, settings->color_mask & 4, settings->color_mask & 8);
    }
    Py_RETURN_NONE;
}

PyObject * Renderer_meth_render(Renderer * self) {
    const GLMethods & gl = self->instance->gl;
    if (self->framebuffer_width != self->instance->viewport_width || self->framebuffer_height != self->instance->viewport_height) {
        gl.Viewport(0, 0, self->framebuffer_width, self->framebuffer_height);
    }
    bind_global_settings(self->instance, self->global_settings);
    bind_framebuffer(self->instance, self->framebuffer);
    bind_program(self->instance, self->program);
    bind_vertex_array(self->instance, self->vertex_array);
    bind_descriptor_set_buffers(self->instance, self->descriptor_set_buffers);
    bind_descriptor_set_images(self->instance, self->descriptor_set_images);
    if (self->index_type) {
        gl.DrawElementsInstanced(self->topology, self->vertex_count, self->index_type, NULL, self->instance_count);
    } else {
        gl.DrawArraysInstanced(self->topology, 0, self->vertex_count, self->instance_count);
    }
    Py_RETURN_NONE;
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
    if (PyFloat_CheckExact(args[0])) {
        float * ptr = (float *)PyBytes_AsString(res);
        for (int i = 0; i < nargs; ++i) {
            *ptr++ = (float)PyFloat_AsDouble(args[i]);
            if (PyErr_Occurred()) {
                return NULL;
            }
        }
    } else {
        int * ptr = (int *)PyBytes_AsString(res);
        for (int i = 0; i < nargs; ++i) {
            *ptr++ = (int)PyLong_AsLong(args[i]);
            if (PyErr_Occurred()) {
                return NULL;
            }
        }
    }
    return res;
}

void default_dealloc(PyObject * self) {
    Py_TYPE(self)->tp_free(self);
}

PyMethodDef Instance_methods[] = {
    {"buffer", (PyCFunction)Instance_meth_buffer, METH_VARARGS | METH_KEYWORDS, NULL},
    {"image", (PyCFunction)Instance_meth_image, METH_VARARGS | METH_KEYWORDS, NULL},
    {"renderer", (PyCFunction)Instance_meth_renderer, METH_VARARGS | METH_KEYWORDS, NULL},
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
    {"clear", (PyCFunction)Image_meth_clear, METH_FASTCALL, NULL},
    {"write", (PyCFunction)Image_meth_write, METH_VARARGS | METH_KEYWORDS, NULL},
    {"read", (PyCFunction)Image_meth_read, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mipmaps", (PyCFunction)Image_meth_mipmaps, METH_VARARGS | METH_KEYWORDS, NULL},
    {"blit", (PyCFunction)Image_meth_blit, METH_VARARGS | METH_KEYWORDS, NULL},
    {},
};

PyMemberDef Image_members[] = {
    {"size", T_OBJECT_EX, offsetof(Image, size), READONLY, NULL},
    {"samples", T_INT, offsetof(Image, samples), READONLY, NULL},
    {},
};

PyMethodDef Renderer_methods[] = {
    {"render", (PyCFunction)Renderer_meth_render, METH_NOARGS, NULL},
    {},
};

PyMemberDef Renderer_members[] = {
    {"vertex_count", T_OBJECT_EX, offsetof(Renderer, vertex_count), 0, NULL},
    {"instance_count", T_OBJECT_EX, offsetof(Renderer, instance_count), 0, NULL},
    {},
};

PyType_Slot Instance_slots[] = {
    {Py_tp_methods, Instance_methods},
    {Py_tp_members, Instance_members},
    {Py_tp_dealloc, default_dealloc},
    {},
};

PyType_Slot Buffer_slots[] = {
    {Py_tp_methods, Buffer_methods},
    {Py_tp_members, Buffer_members},
    {Py_tp_dealloc, default_dealloc},
    {},
};

PyType_Slot Image_slots[] = {
    {Py_tp_methods, Image_methods},
    {Py_tp_members, Image_members},
    {Py_tp_dealloc, default_dealloc},
    {},
};

PyType_Slot Renderer_slots[] = {
    {Py_tp_methods, Renderer_methods},
    {Py_tp_members, Renderer_members},
    {Py_tp_dealloc, default_dealloc},
    {},
};

PyType_Slot DescriptorSetBuffers_slots[] = {
    {Py_tp_dealloc, default_dealloc},
    {},
};

PyType_Slot DescriptorSetImages_slots[] = {
    {Py_tp_dealloc, default_dealloc},
    {},
};

PyType_Slot GlobalSettings_slots[] = {
    {Py_tp_dealloc, default_dealloc},
    {},
};

PyType_Spec Instance_spec = {"zengl.Instance", sizeof(Instance), 0, Py_TPFLAGS_DEFAULT, Instance_slots};
PyType_Spec Buffer_spec = {"zengl.Buffer", sizeof(Buffer), 0, Py_TPFLAGS_DEFAULT, Buffer_slots};
PyType_Spec Image_spec = {"zengl.Image", sizeof(Image), 0, Py_TPFLAGS_DEFAULT, Image_slots};
PyType_Spec Renderer_spec = {"zengl.Renderer", sizeof(Renderer), 0, Py_TPFLAGS_DEFAULT, Renderer_slots};
PyType_Spec DescriptorSetBuffers_spec = {"zengl.DescriptorSetBuffers", sizeof(DescriptorSetBuffers), 0, Py_TPFLAGS_DEFAULT, DescriptorSetBuffers_slots};
PyType_Spec DescriptorSetImages_spec = {"zengl.DescriptorSetImages", sizeof(DescriptorSetImages), 0, Py_TPFLAGS_DEFAULT, DescriptorSetImages_slots};
PyType_Spec GlobalSettings_spec = {"zengl.GlobalSettings", sizeof(GlobalSettings), 0, Py_TPFLAGS_DEFAULT, GlobalSettings_slots};

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

    PyModule_AddObject(self, "Instance", (PyObject *)state->Instance_type);
    PyModule_AddObject(self, "Buffer", (PyObject *)state->Buffer_type);
    PyModule_AddObject(self, "Image", (PyObject *)state->Image_type);
    PyModule_AddObject(self, "Renderer", (PyObject *)state->Renderer_type);

    PyModule_AddObject(self, "context", PyObject_GetAttrString(state->helper, "context"));
    PyModule_AddObject(self, "calcsize", PyObject_GetAttrString(state->helper, "calcsize"));
    PyModule_AddObject(self, "bind", PyObject_GetAttrString(state->helper, "bind"));

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

PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "zengl", NULL, sizeof(ModuleState), module_methods, module_slots};

extern "C" PyObject * PyInit_zengl() {
    return PyModuleDef_Init(&module_def);
}
