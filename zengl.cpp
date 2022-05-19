#include "zengl.hpp"

const StencilSettings default_stencil_settings = {
    0x1E00, 0x1E00, 0x1E00, 0x0207, 0xff, 0xff, 0,
};

struct ModuleState {
    PyObject * helper;
    PyObject * empty_tuple;
    PyObject * str_none;
    PyObject * str_ccw;
    PyObject * float_one;
    PyObject * default_color_mask;
    PyTypeObject * Context_type;
    PyTypeObject * Buffer_type;
    PyTypeObject * Image_type;
    PyTypeObject * Pipeline_type;
    PyTypeObject * ImageFace_type;
    PyTypeObject * DescriptorSetBuffers_type;
    PyTypeObject * DescriptorSetImages_type;
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
    int is_mask_default;
    int is_stencil_default;
    int is_blend_default;
};

struct Context {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    ModuleState * module_state;
    PyObject * descriptor_set_buffers_cache;
    PyObject * descriptor_set_images_cache;
    PyObject * global_settings_cache;
    PyObject * sampler_cache;
    PyObject * vertex_array_cache;
    PyObject * framebuffer_cache;
    PyObject * program_cache;
    PyObject * shader_cache;
    PyObject * includes;
    PyObject * limits;
    PyObject * info;
    DescriptorSetBuffers * current_buffers;
    DescriptorSetImages * current_images;
    GlobalSettings * current_global_settings;
    Viewport viewport;
    int is_mask_default;
    int is_stencil_default;
    int is_blend_default;
    int current_attachments;
    int current_framebuffer;
    int current_program;
    int current_vertex_array;
    int current_clear_mask;
    int default_texture_unit;
    int max_samples;
    int mapped_buffers;
    int screen;
    GLMethods gl;
};

struct Buffer {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    Context * ctx;
    int buffer;
    int size;
    int mapped;
};

struct Image {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    Context * ctx;
    PyObject * size;
    GLObject * framebuffer;
    PyObject * faces;
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
    int max_level;
};

struct Pipeline {
    PyObject_HEAD
    GCHeader * gc_prev;
    GCHeader * gc_next;
    Context * ctx;
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
    int color;
};

void bind_descriptor_set_buffers(Context * self, DescriptorSetBuffers * set) {
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

void bind_descriptor_set_images(Context * self, DescriptorSetImages * set) {
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

void bind_global_settings(Context * self, GlobalSettings * settings) {
    const GLMethods & gl = self->gl;
    if (self->current_global_settings == settings) {
        return;
    }
    if (settings->primitive_restart) {
        gl.Enable(GL_PRIMITIVE_RESTART);
    } else {
        gl.Disable(GL_PRIMITIVE_RESTART);
    }
    if (settings->polygon_offset) {
        gl.Enable(GL_POLYGON_OFFSET_FILL);
        gl.PolygonOffset(settings->polygon_offset_factor, settings->polygon_offset_units);
    } else {
        gl.Disable(GL_POLYGON_OFFSET_FILL);
    }
    if (settings->cull_face) {
        gl.Enable(GL_CULL_FACE);
        gl.CullFace(settings->cull_face);
    } else {
        gl.Disable(GL_CULL_FACE);
    }
    if (settings->depth_test) {
        gl.Enable(GL_DEPTH_TEST);
    } else {
        gl.Disable(GL_DEPTH_TEST);
    }
    if (!self->is_stencil_default || !settings->is_stencil_default) {
        if (settings->stencil_test) {
            gl.Enable(GL_STENCIL_TEST);
        } else {
            gl.Disable(GL_STENCIL_TEST);
        }
        gl.StencilMaskSeparate(GL_FRONT, settings->stencil_front.write_mask);
        gl.StencilMaskSeparate(GL_BACK, settings->stencil_back.write_mask);
        gl.StencilFuncSeparate(GL_FRONT, settings->stencil_front.compare_op, settings->stencil_front.reference, settings->stencil_front.compare_mask);
        gl.StencilFuncSeparate(GL_BACK, settings->stencil_back.compare_op, settings->stencil_back.reference, settings->stencil_back.compare_mask);
        gl.StencilOpSeparate(GL_FRONT, settings->stencil_front.fail_op, settings->stencil_front.pass_op, settings->stencil_front.depth_fail_op);
        gl.StencilOpSeparate(GL_BACK, settings->stencil_back.fail_op, settings->stencil_back.pass_op, settings->stencil_back.depth_fail_op);
        self->is_stencil_default = settings->is_stencil_default;
    }
    if (!self->is_mask_default || !settings->is_mask_default || self->current_attachments != settings->attachments) {
        gl.DepthMask(settings->depth_write);
        for (int i = 0; i < settings->attachments; ++i) {
            gl.ColorMaski(
                i,
                settings->color_mask >> (i * 4 + 0) & 1,
                settings->color_mask >> (i * 4 + 1) & 1,
                settings->color_mask >> (i * 4 + 2) & 1,
                settings->color_mask >> (i * 4 + 3) & 1
            );
        }
        self->is_mask_default = settings->is_mask_default;
    }
    if (!self->is_blend_default || !settings->is_blend_default || self->current_attachments != settings->attachments) {
        gl.BlendFuncSeparate(settings->blend_src_color, settings->blend_dst_color, settings->blend_src_alpha, settings->blend_dst_alpha);
        for (int i = 0; i < settings->attachments; ++i) {
            if (settings->blend_enable >> i & 1) {
                gl.Enablei(GL_BLEND, i);
            } else {
                gl.Disablei(GL_BLEND, i);
            }
        }
        self->is_blend_default = settings->is_blend_default;
    }
    self->current_global_settings = settings;
    self->current_attachments = settings->attachments;
    if (settings->attachments > 0) {
        self->current_clear_mask = settings->color_mask & 0xf | settings->depth_write << 8 | settings->stencil_front.write_mask << 16;
    } else {
        self->current_clear_mask = self->current_clear_mask & 0xf | settings->depth_write << 8 | settings->stencil_front.write_mask << 16;
    }
}

void bind_framebuffer(Context * self, int framebuffer) {
    if (self->current_framebuffer != framebuffer) {
        self->current_framebuffer = framebuffer;
        self->gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    }
}

void bind_program(Context * self, int program) {
    if (self->current_program != program) {
        self->current_program = program;
        self->gl.UseProgram(program);
    }
}

void bind_vertex_array(Context * self, int vertex_array) {
    if (self->current_vertex_array != vertex_array) {
        self->current_vertex_array = vertex_array;
        self->gl.BindVertexArray(vertex_array);
    }
}

GLObject * build_framebuffer(Context * self, PyObject * attachments) {
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
            gl.FramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,face->image->image, face->level, face->layer);
        } else {
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, face->image->image, face->level);
        }
    }

    if (depth_stencil_attachment != Py_None) {
        ImageFace * face = (ImageFace *)depth_stencil_attachment;
        int buffer = face->image->format.buffer;
        int attachment = buffer == GL_DEPTH ? GL_DEPTH_ATTACHMENT : buffer == GL_STENCIL ? GL_STENCIL_ATTACHMENT : GL_DEPTH_STENCIL_ATTACHMENT;
        if (face->image->renderbuffer) {
            gl.FramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, face->image->image);
        } else if (face->image->cubemap) {
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face->layer, face->image->image, face->level);
        } else if (face->image->array) {
            gl.FramebufferTextureLayer(GL_FRAMEBUFFER, attachment,face->image->image, face->level, face->layer);
        } else {
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, face->image->image, face->level);
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

GLObject * build_vertex_array(Context * self, PyObject * bindings) {
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

GLObject * build_sampler(Context * self, PyObject * params) {
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

    float color[] = {
        (float)PyFloat_AsDouble(seq[11]),
        (float)PyFloat_AsDouble(seq[12]),
        (float)PyFloat_AsDouble(seq[13]),
        (float)PyFloat_AsDouble(seq[14]),
    };
    gl.SamplerParameterfv(sampler, GL_TEXTURE_BORDER_COLOR, color);

    GLObject * res = PyObject_New(GLObject, self->module_state->GLObject_type);
    res->obj = sampler;
    res->uses = 1;

    PyDict_SetItem(self->sampler_cache, params, (PyObject *)res);
    return res;
}

DescriptorSetBuffers * build_descriptor_set_buffers(Context * self, PyObject * bindings) {
    if (DescriptorSetBuffers * cache = (DescriptorSetBuffers *)PyDict_GetItem(self->descriptor_set_buffers_cache, bindings)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

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
        res->binding[binding] = {buffer->buffer, offset, size};
        res->buffers = res->buffers > (binding + 1) ? res->buffers : (binding + 1);
    }

    PyDict_SetItem(self->descriptor_set_buffers_cache, bindings, (PyObject *)res);
    return res;
}

DescriptorSetImages * build_descriptor_set_images(Context * self, PyObject * bindings) {
    if (DescriptorSetImages * cache = (DescriptorSetImages *)PyDict_GetItem(self->descriptor_set_images_cache, bindings)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

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

GlobalSettings * build_global_settings(Context * self, PyObject * settings) {
    if (GlobalSettings * cache = (GlobalSettings *)PyDict_GetItem(self->global_settings_cache, settings)) {
        cache->uses += 1;
        Py_INCREF(cache);
        return cache;
    }

    PyObject ** seq = PySequence_Fast_ITEMS(settings);

    GlobalSettings * res = PyObject_New(GlobalSettings, self->module_state->GlobalSettings_type);
    res->uses = 1;

    res->primitive_restart = PyObject_IsTrue(seq[0]);
    res->cull_face = PyLong_AsLong(seq[1]);
    res->color_mask = PyLong_AsUnsignedLongLong(seq[2]);
    res->depth_test = PyObject_IsTrue(seq[3]);
    res->depth_write = PyObject_IsTrue(seq[4]);
    res->depth_func = PyLong_AsLong(seq[5]);
    res->stencil_test = PyObject_IsTrue(seq[6]);
    res->stencil_front = {
        PyLong_AsLong(seq[7]),
        PyLong_AsLong(seq[8]),
        PyLong_AsLong(seq[9]),
        PyLong_AsLong(seq[10]),
        PyLong_AsLong(seq[11]),
        PyLong_AsLong(seq[12]),
        PyLong_AsLong(seq[13]),
    };
    res->stencil_back = {
        PyLong_AsLong(seq[14]),
        PyLong_AsLong(seq[15]),
        PyLong_AsLong(seq[16]),
        PyLong_AsLong(seq[17]),
        PyLong_AsLong(seq[18]),
        PyLong_AsLong(seq[19]),
        PyLong_AsLong(seq[20]),
    };
    res->blend_enable = PyLong_AsLong(seq[21]);
    res->blend_src_color = PyLong_AsLong(seq[22]);
    res->blend_dst_color = PyLong_AsLong(seq[23]);
    res->blend_src_alpha = PyLong_AsLong(seq[24]);
    res->blend_dst_alpha = PyLong_AsLong(seq[25]);
    res->polygon_offset = PyObject_IsTrue(seq[26]);
    res->polygon_offset_factor = (float)PyFloat_AsDouble(seq[27]);
    res->polygon_offset_units = (float)PyFloat_AsDouble(seq[28]);
    res->attachments = PyLong_AsLong(seq[29]);

    res->is_mask_default = res->color_mask == 0xffffffffffffffffull && res->depth_write;
    res->is_stencil_default = !res->stencil_test;
    if (memcmp(&res->stencil_front, &default_stencil_settings, sizeof(StencilSettings))) {
        res->is_stencil_default = false;
    }
    if (memcmp(&res->stencil_back, &default_stencil_settings, sizeof(StencilSettings))) {
        res->is_stencil_default = false;
    }
    res->is_blend_default = !res->blend_enable;
    if (res->blend_src_color != 1 || res->blend_dst_color != 0 || res->blend_src_alpha != 1 || res->blend_dst_alpha != 0) {
        res->is_blend_default = false;
    }

    PyDict_SetItem(self->global_settings_cache, settings, (PyObject *)res);
    return res;
}

GLObject * compile_shader(Context * self, PyObject * code, int type, const char * name) {
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

GLObject * compile_program(Context * self, PyObject * vert, PyObject * frag, PyObject * layout) {
    const GLMethods & gl = self->gl;

    PyObject * pair = PyObject_CallMethod(self->module_state->helper, "program", "OOOO", vert, frag, layout, self->includes);
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

Context * meth_context(PyObject * self, PyObject * vargs, PyObject * kwargs) {
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

    int max_uniform_buffer_bindings = 0;
    gl.GetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &max_uniform_buffer_bindings);

    int max_uniform_block_size = 0;
    gl.GetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &max_uniform_block_size);

    int max_combined_uniform_blocks = 0;
    gl.GetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS, &max_combined_uniform_blocks);

    int max_combined_texture_image_units = 0;
    gl.GetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_combined_texture_image_units);

    int max_vertex_attribs = 0;
    gl.GetIntegerv(GL_MAX_VERTEX_ATTRIBS, &max_vertex_attribs);

    int max_draw_buffers = 0;
    gl.GetIntegerv(GL_MAX_DRAW_BUFFERS, &max_draw_buffers);

    int max_samples = 0;
    gl.GetIntegerv(GL_MAX_SAMPLES, &max_samples);

    int max_texture_image_units = 0;
    gl.GetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &max_texture_image_units);
    int default_texture_unit = GL_TEXTURE0 + max_texture_image_units - 1;

    gl.PrimitiveRestartIndex(-1);
    gl.Enable(GL_PROGRAM_POINT_SIZE);
    gl.Enable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    gl.Enable(GL_FRAMEBUFFER_SRGB);

    PyObject * limits = Py_BuildValue(
        "{sisisisisisisi}",
        "max_uniform_buffer_bindings", max_uniform_buffer_bindings,
        "max_uniform_block_size", max_uniform_block_size,
        "max_combined_uniform_blocks", max_combined_uniform_blocks,
        "max_combined_texture_image_units", max_combined_texture_image_units,
        "max_vertex_attribs", max_vertex_attribs,
        "max_draw_buffers", max_draw_buffers,
        "max_samples", max_samples
    );

    PyObject * info = PyTuple_New(3);
    PyTuple_SetItem(info, 0, to_str(gl.GetString(GL_VENDOR)));
    PyTuple_SetItem(info, 1, to_str(gl.GetString(GL_RENDERER)));
    PyTuple_SetItem(info, 2, to_str(gl.GetString(GL_VERSION)));

    Context * res = PyObject_New(Context, module_state->Context_type);
    res->gc_prev = (GCHeader *)res;
    res->gc_next = (GCHeader *)res;
    res->module_state = module_state;
    res->descriptor_set_buffers_cache = PyDict_New();
    res->descriptor_set_images_cache = PyDict_New();
    res->global_settings_cache = PyDict_New();
    res->sampler_cache = PyDict_New();
    res->vertex_array_cache = PyDict_New();
    res->framebuffer_cache = PyDict_New();
    res->program_cache = PyDict_New();
    res->shader_cache = PyDict_New();
    res->includes = PyDict_New();
    res->limits = limits;
    res->info = info;
    res->current_buffers = NULL;
    res->current_images = NULL;
    res->current_global_settings = NULL;
    res->is_mask_default = false;
    res->is_stencil_default = false;
    res->is_blend_default = false;
    res->current_attachments = -1;
    res->current_framebuffer = -1;
    res->current_program = -1;
    res->current_vertex_array = -1;
    res->current_clear_mask = 0;
    res->viewport = {};
    res->default_texture_unit = default_texture_unit;
    res->max_samples = max_samples;
    res->mapped_buffers = 0;
    res->screen = 0;
    res->gl = gl;
    return res;
}

Buffer * Context_meth_buffer(Context * self, PyObject * vargs, PyObject * kwargs) {
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
    res->gc_prev = self->gc_prev;
    res->gc_next = (GCHeader *)self;
    res->gc_prev->gc_next = (GCHeader *)res;
    res->gc_next->gc_prev = (GCHeader *)res;
    res->ctx = (Context *)new_ref(self);
    res->buffer = buffer;
    res->size = size;
    res->mapped = false;

    if (data != Py_None) {
        PyBuffer_Release(&view);
    }

    Py_INCREF(res);
    return res;
}

Image * Context_meth_image(Context * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"size", "format", "data", "samples", "array", "texture", "cubemap", NULL};

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

    if (samples > self->max_samples) {
        samples = self->max_samples;
    }

    ImageFormat format = get_image_format(format_str);

    if (!format.type) {
        PyErr_Format(PyExc_ValueError, "invalid image format");
        return NULL;
    }

    Py_buffer view = {};

    if (data != Py_None) {
        if (PyObject_GetBuffer(data, &view, PyBUF_SIMPLE)) {
            return NULL;
        }
        int padded_row = (width * format.pixel_size + 3) & ~3;
        int expected_size = padded_row * height * (array ? array : 1) * (cubemap ? 6 : 1);
        if ((int)view.len != expected_size) {
            PyBuffer_Release(&view);
            PyErr_Format(PyExc_ValueError, "invalid data size, expected %d, got %d", expected_size, (int)view.len);
            return NULL;
        }
    }

    int image = 0;
    if (renderbuffer) {
        gl.GenRenderbuffers(1, (unsigned *)&image);
        gl.BindRenderbuffer(GL_RENDERBUFFER, image);
        gl.RenderbufferStorageMultisample(GL_RENDERBUFFER, samples > 1 ? samples : 0, format.internal_format, width, height);
    } else {
        gl.GenTextures(1, (unsigned *)&image);
        gl.ActiveTexture(self->default_texture_unit);
        gl.BindTexture(target, image);
        if (cubemap) {
            int padded_row = (width * format.pixel_size + 3) & ~3;
            int stride = padded_row * height;
            for (int i = 0; i < 6; ++i) {
                int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i;
                char * ptr = view.buf ? (char *)view.buf + stride * i : NULL;
                gl.TexImage2D(face, 0, format.internal_format, width, height, 0, format.format, format.type, ptr);
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
    res->gc_prev = self->gc_prev;
    res->gc_next = (GCHeader *)self;
    res->gc_prev->gc_next = (GCHeader *)res;
    res->gc_next->gc_prev = (GCHeader *)res;
    res->ctx = (Context *)new_ref(self);
    res->size = Py_BuildValue("(ii)", width, height);
    res->faces = PyDict_New();
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
    res->max_level = 0;

    res->framebuffer = NULL;
    if (!cubemap && !array) {
        if (format.color) {
            PyObject * face = PyObject_CallMethod((PyObject *)res, "face", "(ii)", 0, 0);
            PyObject * attachments = Py_BuildValue("((ii)(N)O)", width, height, face, Py_None);
            res->framebuffer = build_framebuffer(self, attachments);
            Py_DECREF(attachments);
        } else {
            PyObject * face = PyObject_CallMethod((PyObject *)res, "face", "(ii)", 0, 0);
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

Pipeline * Context_meth_pipeline(Context * self, PyObject * vargs, PyObject * kwargs) {
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
        "cull_face",
        "topology",
        "vertex_count",
        "instance_count",
        "first_vertex",
        "viewport",
        "skip_validation",
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
    PyObject * cull_face = self->module_state->str_none;
    const char * topology = "triangles";
    int vertex_count = 0;
    int instance_count = 1;
    int first_vertex = 0;
    PyObject * viewport = Py_None;
    int skip_validation = false;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "|$O!O!OOOOOOOOOOpOOsiiiOp",
        keywords,
        &PyUnicode_Type,
        &vertex_shader,
        &PyUnicode_Type,
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
        &cull_face,
        &topology,
        &vertex_count,
        &instance_count,
        &first_vertex,
        &viewport,
        &skip_validation
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
        PyList_SET_ITEM(program_attributes, i, Py_BuildValue("{sssisi}", "name", name, "location", location, "size", size));
    }

    for (int i = 0; i < uniforms; ++i) {
        int size = 0;
        int type = 0;
        int length = 0;
        char name[256] = {};
        gl.GetActiveUniform(program->obj, i, 256, &length, &size, (unsigned *)&type, name);
        int location = gl.GetUniformLocation(program->obj, name);
        PyList_SET_ITEM(program_uniforms, i, Py_BuildValue("{sssisi}", "name", name, "location", location, "size", size));
    }

    for (int i = 0; i < uniform_buffers; ++i) {
        int size = 0;
        int length = 0;
        char name[256] = {};
        gl.GetActiveUniformBlockiv(program->obj, i, GL_UNIFORM_BLOCK_DATA_SIZE, &size);
        gl.GetActiveUniformBlockName(program->obj, i, 256, &length, name);
        PyList_SET_ITEM(program_uniform_buffers, i, Py_BuildValue("{sssi}", "name", name, "size", size));
    }

    if (!skip_validation) {
        PyObject * validate = PyObject_CallMethod(
            self->module_state->helper,
            "validate",
            "NNNOOOO",
            program_attributes,
            program_uniforms,
            program_uniform_buffers,
            vertex_buffers,
            layout,
            resources,
            self->limits
        );

        if (!validate) {
            return NULL;
        }
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

    Viewport viewport_value = {};
    if (viewport != Py_None) {
        viewport_value = to_viewport(viewport);
    } else {
        PyObject * size = PyTuple_GetItem(attachments, 0);
        viewport_value.width = (short)PyLong_AsLong(PyTuple_GetItem(size, 0));
        viewport_value.height = (short)PyLong_AsLong(PyTuple_GetItem(size, 1));
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
        "OOOOOOON",
        primitive_restart,
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

    Pipeline * res = PyObject_New(Pipeline, self->module_state->Pipeline_type);
    res->gc_prev = self->gc_prev;
    res->gc_next = (GCHeader *)self;
    res->gc_prev->gc_next = (GCHeader *)res;
    res->gc_next->gc_prev = (GCHeader *)res;
    res->ctx = (Context *)new_ref(self);
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

PyObject * Context_meth_release(Context * self, PyObject * arg) {
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
            image->framebuffer->uses -= 1;
            if (!image->framebuffer->uses) {
                remove_dict_value(self->framebuffer_cache, (PyObject *)image->framebuffer);
                if (self->current_framebuffer == image->framebuffer->obj) {
                    self->current_framebuffer = 0;
                }
                gl.DeleteFramebuffers(1, (unsigned int *)&image->framebuffer->obj);
            }
        }
        if (image->faces) {
            PyObject * key = NULL;
            PyObject * value = NULL;
            Py_ssize_t pos = 0;
            while (PyDict_Next(image->faces, &pos, &key, &value)) {
                GLObject * framebuffer = (GLObject *)value;
                if (self->current_framebuffer == framebuffer->obj) {
                    self->current_framebuffer = 0;
                }
                gl.DeleteFramebuffers(1, (unsigned int *)&framebuffer->obj);
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
        pipeline->descriptor_set_buffers->uses -= 1;
        if (!pipeline->descriptor_set_buffers->uses) {
            remove_dict_value(self->descriptor_set_buffers_cache, (PyObject *)pipeline->descriptor_set_buffers);
            if (self->current_buffers == pipeline->descriptor_set_buffers) {
                self->current_buffers = NULL;
            }
        }
        pipeline->descriptor_set_images->uses -= 1;
        if (!pipeline->descriptor_set_images->uses) {
            for (int i = 0; i < pipeline->descriptor_set_images->samplers; ++i) {
                GLObject * sampler = pipeline->descriptor_set_images->sampler[i];
                sampler->uses -= 1;
                if (!sampler->uses) {
                    remove_dict_value(self->sampler_cache, (PyObject *)sampler);
                    gl.DeleteSamplers(1, (unsigned int *)&sampler->obj);
                }
            }
            remove_dict_value(self->descriptor_set_images_cache, (PyObject *)pipeline->descriptor_set_images);
            if (self->current_images == pipeline->descriptor_set_images) {
                self->current_images = NULL;
            }
        }
        pipeline->global_settings->uses -= 1;
        if (!pipeline->global_settings->uses) {
            remove_dict_value(self->global_settings_cache, (PyObject *)pipeline->global_settings);
            if (self->current_global_settings == pipeline->global_settings) {
                self->current_global_settings = NULL;
            }
        }
        pipeline->framebuffer->uses -= 1;
        if (!pipeline->framebuffer->uses) {
            remove_dict_value(self->framebuffer_cache, (PyObject *)pipeline->framebuffer);
            if (self->current_framebuffer == pipeline->framebuffer->obj) {
                self->current_framebuffer = 0;
            }
            gl.DeleteFramebuffers(1, (unsigned int *)&pipeline->framebuffer->obj);
        }
        pipeline->program->uses -= 1;
        if (!pipeline->program->uses) {
            remove_dict_value(self->program_cache, (PyObject *)pipeline->program);
            if (self->current_program == pipeline->program->obj) {
                self->current_program = 0;
            }
            gl.DeleteProgram(pipeline->program->obj);
        }
        pipeline->vertex_array->uses -= 1;
        if (!pipeline->vertex_array->uses) {
            remove_dict_value(self->vertex_array_cache, (PyObject *)pipeline->vertex_array);
            if (self->current_vertex_array == pipeline->vertex_array->obj) {
                self->current_vertex_array = 0;
            }
            gl.DeleteVertexArrays(1, (unsigned int *)&pipeline->vertex_array->obj);
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
            if (Py_TYPE(it) == self->module_state->Pipeline_type) {
                Py_DECREF(Context_meth_release(self, (PyObject *)it));
            }
            it = it->gc_next;
        }
        it = self->gc_next;
        while (it != (GCHeader *)self) {
            if (Py_TYPE(it) == self->module_state->Buffer_type) {
                Py_DECREF(Context_meth_release(self, (PyObject *)it));
            }
            if (Py_TYPE(it) == self->module_state->Image_type) {
                Py_DECREF(Context_meth_release(self, (PyObject *)it));
            }
            it = it->gc_next;
        }
    }
    Py_RETURN_NONE;
}

PyObject * Context_meth_reset(Context * self) {
    self->current_buffers = NULL;
    self->current_images = NULL;
    self->current_global_settings = NULL;
    self->viewport.viewport = 0xffffffffffffffffull;
    self->is_stencil_default = false;
    self->is_mask_default = false;
    self->is_blend_default = false;
    self->current_attachments = -1;
    self->current_framebuffer = -1;
    self->current_program = -1;
    self->current_vertex_array = -1;
    self->current_clear_mask = 0;
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

    const GLMethods & gl = self->ctx->gl;

    self->mapped = true;
    self->ctx->mapped_buffers += 1;
    const int access = discard ? GL_MAP_READ_BIT | GL_MAP_WRITE_BIT : GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_WRITE_BIT;
    gl.BindBuffer(GL_ARRAY_BUFFER, self->buffer);
    void * ptr = gl.MapBufferRange(GL_ARRAY_BUFFER, offset, size, access);
    return PyMemoryView_FromMemory((char *)ptr, size, PyBUF_WRITE);
}

PyObject * Buffer_meth_unmap(Buffer * self) {
    const GLMethods & gl = self->ctx->gl;
    if (self->mapped) {
        self->mapped = false;
        self->ctx->mapped_buffers -= 1;
        gl.BindBuffer(GL_ARRAY_BUFFER, self->buffer);
        gl.UnmapBuffer(GL_ARRAY_BUFFER);
    }
    Py_RETURN_NONE;
}

void clear_bound_image(Image * self) {
    const GLMethods & gl = self->ctx->gl;
    switch (self->format.buffer) {
        case GL_COLOR: {
            if ((self->ctx->current_clear_mask & 0xf) != 0xf) {
                self->ctx->current_clear_mask |= 0xf;
                self->ctx->current_global_settings = NULL;
                gl.ColorMaski(0, 1, 1, 1, 1);
            }
            break;
        }
        case GL_DEPTH: {
            if ((self->ctx->current_clear_mask & 0x100) != 0x100) {
                self->ctx->current_clear_mask |= 0x100;
                self->ctx->current_global_settings = NULL;
                gl.DepthMask(1);
            }
            break;
        }
        case GL_STENCIL: {
            if ((self->ctx->current_clear_mask & 0xff0000) != 0xff0000) {
                self->ctx->current_clear_mask |= 0xff0000;
                self->ctx->current_global_settings = NULL;
                gl.StencilMaskSeparate(GL_FRONT, 0xff);
            }
            break;
        }
        case GL_DEPTH_STENCIL: {
            if ((self->ctx->current_clear_mask & 0xff0100) != 0xff0100) {
                self->ctx->current_clear_mask |= 0xff0100;
                self->ctx->current_global_settings = NULL;
                gl.StencilMaskSeparate(GL_FRONT, 0xff);
                gl.DepthMask(1);
            }
            break;
        }
    }
    if (self->format.clear_type == 'f') {
        gl.ClearBufferfv(self->format.buffer, 0, self->clear_value.clear_floats);
    } else if (self->format.clear_type == 'i') {
        gl.ClearBufferiv(self->format.buffer, 0, self->clear_value.clear_ints);
    } else if (self->format.clear_type == 'u') {
        gl.ClearBufferuiv(self->format.buffer, 0, self->clear_value.clear_uints);
    } else if (self->format.clear_type == 'x') {
        gl.ClearBufferfi(self->format.buffer, 0, self->clear_value.clear_floats[0], self->clear_value.clear_ints[1]);
    }
}

PyObject * Image_meth_clear(Image * self) {
    if (!self->framebuffer) {
        PyErr_Format(PyExc_TypeError, "cannot clear cubemap or array textures");
        return NULL;
    }
    bind_framebuffer(self->ctx, self->framebuffer->obj);
    clear_bound_image(self);
    Py_RETURN_NONE;
}

PyObject * Image_meth_write(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"data", "size", "offset", "layer", "level", NULL};

    Py_buffer view;
    PyObject * size_arg = Py_None;
    PyObject * offset_arg = Py_None;
    PyObject * layer_arg = Py_None;
    int level = 0;

    if (!PyArg_ParseTupleAndKeywords(vargs, kwargs, "y*|O$OOi", keywords, &view, &size_arg, &offset_arg, &layer_arg, &level)) {
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
    const bool invalid_type = !self->format.color || self->samples != 1;

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
        } else if (!self->format.color) {
            PyErr_Format(PyExc_TypeError, "cannot write to depth or stencil images");
        } else if (self->samples != 1) {
            PyErr_Format(PyExc_TypeError, "cannot write to multisampled images");
        }
        return NULL;
    }

    int padded_row = (size.x * self->format.pixel_size + 3) & ~3;
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
        int padded_row = (size.x * self->format.pixel_size + 3) & ~3;
        int stride = padded_row * size.y;
        if (layer_arg != Py_None) {
            int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer;
            gl.TexSubImage2D(face, level, offset.x, offset.y, size.x, size.y, self->format.format, self->format.type, view.buf);
        } else {
            for (int i = 0; i < 6; ++i) {
                int face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i;
                gl.TexSubImage2D(face, level, offset.x, offset.y, size.x, size.y, self->format.format, self->format.type, (char *)view.buf + stride * i);
            }
        }
    } else if (self->array) {
        if (layer_arg != Py_None) {
            gl.TexSubImage3D(self->target, level, offset.x, offset.y, layer, size.x, size.y, 1, self->format.format, self->format.type, view.buf);
        } else {
            gl.TexSubImage3D(self->target, level, offset.x, offset.y, 0, size.x, size.y, self->array, self->format.format, self->format.type, view.buf);
        }
    } else {
        gl.TexSubImage2D(self->target, level, offset.x, offset.y, size.x, size.y, self->format.format, self->format.type, view.buf);
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

    const GLMethods & gl = self->ctx->gl;

    PyObject * res = PyBytes_FromStringAndSize(NULL, size.x * size.y * self->format.pixel_size);
    bind_framebuffer(self->ctx, self->framebuffer->obj);
    gl.ReadPixels(offset.x, offset.y, size.x, size.y, self->format.format, self->format.type, PyBytes_AS_STRING(res));
    return res;
}

PyObject * Image_meth_blit(Image * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"target", "target_viewport", "source_viewport", "filter", "srgb", "flush", NULL};

    PyObject * target_arg = Py_None;
    PyObject * target_viewport_arg = Py_None;
    PyObject * source_viewport_arg = Py_None;
    int filter = true;
    PyObject * srgb_arg = Py_None;
    PyObject * flush_arg = Py_None;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "|OO$OpOO",
        keywords,
        &target_arg,
        &target_viewport_arg,
        &source_viewport_arg,
        &filter,
        &srgb_arg,
        &flush_arg
    );

    if (!args_ok) {
        return NULL;
    }

    const bool invalid_target_type = target_arg != Py_None && Py_TYPE(target_arg) != self->ctx->module_state->Image_type;
    const bool invalid_srgb_parameter = srgb_arg != Py_True && srgb_arg != Py_False && srgb_arg != Py_None;
    const bool invalid_flush_parameter = flush_arg != Py_True && flush_arg != Py_False && flush_arg != Py_None;

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

    const bool srgb = (srgb_arg == Py_None && self->format.internal_format == GL_SRGB8_ALPHA8) || srgb_arg == Py_True;
    const bool flush = (flush_arg == Py_None && target_arg == Py_None) || flush_arg == Py_True;

    const bool invalid_target_viewport = invalid_target_viewport_type || (
        target_viewport.x < 0 || target_viewport.y < 0 || target_viewport.width <= 0 || target_viewport.height <= 0 ||
        (target && (target_viewport.x + target_viewport.width > target->width || target_viewport.y + target_viewport.height > target->height))
    );

    const bool invalid_source_viewport = invalid_source_viewport_type || (
        source_viewport.x < 0 || source_viewport.y < 0 || source_viewport.width <= 0 || source_viewport.height <= 0 ||
        source_viewport.x + source_viewport.width > self->width || source_viewport.y + source_viewport.height > self->height
    );

    const bool invalid_target = target && (target->cubemap || target->array || !target->format.color || target->samples > 1);
    const bool invalid_source = self->cubemap || self->array || !self->format.color;

    const bool error = (
        invalid_target_type || invalid_srgb_parameter || invalid_flush_parameter ||
        invalid_target_viewport_type || invalid_source_viewport_type || invalid_target_viewport ||
        invalid_source_viewport || invalid_target || invalid_source
    );

    if (error) {
        if (invalid_target_type) {
            PyErr_Format(PyExc_TypeError, "target must be an Image or None");
        } else if (invalid_srgb_parameter) {
            PyErr_Format(PyExc_TypeError, "invalid srgb parameter");
        } else if (invalid_flush_parameter) {
            PyErr_Format(PyExc_TypeError, "invalid flush parameter");
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
        } else if (target && !target->format.color) {
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
    if ((self->ctx->current_clear_mask & 0xf) != 0xf) {
        self->ctx->current_clear_mask |= 0xf;
        self->ctx->current_global_settings = NULL;
        gl.ColorMaski(0, 1, 1, 1, 1);
    }
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, self->framebuffer->obj);
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, target ? target->framebuffer->obj : self->ctx->screen);
    gl.BlitFramebuffer(
        source_viewport.x, source_viewport.y, source_viewport.x + source_viewport.width, source_viewport.y + source_viewport.height,
        target_viewport.x, target_viewport.y, target_viewport.x + target_viewport.width, target_viewport.y + target_viewport.height,
        GL_COLOR_BUFFER_BIT, filter ? GL_LINEAR : GL_NEAREST
    );
    if (!target) {
        self->ctx->current_framebuffer = self->ctx->screen;
    }
    gl.BindFramebuffer(GL_FRAMEBUFFER, self->ctx->current_framebuffer);
    if (!srgb) {
        gl.Enable(GL_FRAMEBUFFER_SRGB);
    }
    if (flush) {
        gl.Flush();
    }
    Py_RETURN_NONE;
}

ImageFace * Image_meth_face(Image * self, PyObject * vargs, PyObject * kwargs) {
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
    res->color = self->format.color;

    res->framebuffer = NULL;
    if (self->format.color) {
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

PyObject * Image_get_clear_value(Image * self) {
    if (self->format.clear_type == 'x') {
        return Py_BuildValue("fi", self->clear_value.clear_floats[0], self->clear_value.clear_ints[1]);
    }
    if (self->format.components == 1) {
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

PyObject * Pipeline_meth_render(Pipeline * self) {
    if (self->ctx->mapped_buffers) {
        PyErr_Format(PyExc_RuntimeError, "rendering with mapped buffers");
        return NULL;
    }
    const GLMethods & gl = self->ctx->gl;
    if (self->viewport.viewport != self->ctx->viewport.viewport) {
        gl.Viewport(self->viewport.x, self->viewport.y, self->viewport.width, self->viewport.height);
        self->ctx->viewport.viewport = self->viewport.viewport;
    }
    bind_global_settings(self->ctx, self->global_settings);
    bind_framebuffer(self->ctx, self->framebuffer->obj);
    bind_program(self->ctx, self->program->obj);
    bind_vertex_array(self->ctx, self->vertex_array->obj);
    bind_descriptor_set_buffers(self->ctx, self->descriptor_set_buffers);
    bind_descriptor_set_images(self->ctx, self->descriptor_set_images);
    if (self->index_type) {
        long long offset = self->first_vertex * self->index_size;
        gl.DrawElementsInstanced(self->topology, self->vertex_count, self->index_type, (void *)offset, self->instance_count);
    } else {
        gl.DrawArraysInstanced(self->topology, self->first_vertex, self->vertex_count, self->instance_count);
    }
    Py_RETURN_NONE;
}

PyObject * Pipeline_get_viewport(Pipeline * self) {
    return Py_BuildValue("iiii", self->viewport.x, self->viewport.y, self->viewport.width, self->viewport.height);
}

int Pipeline_set_viewport(Pipeline * self, PyObject * viewport) {
    if (!is_viewport(viewport)) {
        PyErr_Format(PyExc_TypeError, "the viewport must be a tuple of 4 ints");
        return -1;
    }
    self->viewport = to_viewport(viewport);
    return 0;
}

PyObject * Pipeline_get_framebuffer(Pipeline * self) {
    return PyLong_FromLong(self->framebuffer->obj);
}

int Pipeline_set_framebuffer(Pipeline * self, PyObject * framebuffer) {
    if (!PyLong_CheckExact(framebuffer)) {
        PyErr_Format(PyExc_TypeError, "the framebuffer must be an int");
        return -1;
    }
    self->framebuffer = PyObject_New(GLObject, self->ctx->module_state->GLObject_type);
    self->framebuffer->uses = -1;
    self->framebuffer->obj = PyLong_AsLong(framebuffer);
    return 0;
}

PyObject * meth_inspect(PyObject * self, PyObject * arg) {
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
        PyObject * buffers = PyList_New(pipeline->descriptor_set_buffers->buffers);
        for (int i = 0; i < pipeline->descriptor_set_buffers->buffers; ++i) {
            int buffer = pipeline->descriptor_set_buffers->binding[i].buffer;
            PyObject * obj = Py_BuildValue("{sssi}", "type", "buffer", "buffer", buffer);
            PyList_SetItem(buffers, i, obj);
        }
        PyObject * samplers = PyList_New(pipeline->descriptor_set_images->samplers);
        for (int i = 0; i < pipeline->descriptor_set_images->samplers; ++i) {
            int sampler = pipeline->descriptor_set_images->binding[i].sampler;
            int image = pipeline->descriptor_set_images->binding[i].image;
            PyObject * obj = Py_BuildValue("{sssisi}", "type", "sampler", "sampler", sampler, "texture", image);
            PyList_SetItem(samplers, i, obj);
        }
        return Py_BuildValue(
            "{sssNsNsisisi}",
            "type", "pipeline",
            "buffers", buffers,
            "samplers", samplers,
            "framebuffer", pipeline->framebuffer->obj,
            "vertex_array", pipeline->vertex_array->obj,
            "program", pipeline->program->obj
        );
    }
    Py_RETURN_NONE;
}

PyObject * ImageFace_meth_clear(ImageFace * self) {
    bind_framebuffer(self->ctx, self->framebuffer->obj);
    clear_bound_image(self->image);
    Py_RETURN_NONE;
}

PyObject * ImageFace_meth_blit(ImageFace * self, PyObject * vargs, PyObject * kwargs) {
    static char * keywords[] = {"target", "target_viewport", "source_viewport", "filter", "srgb", NULL};

    ImageFace * target;
    PyObject * target_viewport_arg = Py_None;
    PyObject * source_viewport_arg = Py_None;
    int filter = true;
    PyObject * srgb_arg = Py_None;

    int args_ok = PyArg_ParseTupleAndKeywords(
        vargs,
        kwargs,
        "|O!O$OpO",
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

    const bool srgb = (srgb_arg == Py_None && self->image->format.internal_format == GL_SRGB8_ALPHA8) || srgb_arg == Py_True;

    const bool invalid_target_viewport = invalid_target_viewport_type || (
        target_viewport.x < 0 || target_viewport.y < 0 || target_viewport.width <= 0 || target_viewport.height <= 0 ||
        (target && (target_viewport.x + target_viewport.width > target->width || target_viewport.y + target_viewport.height > target->height))
    );

    const bool invalid_source_viewport = invalid_source_viewport_type || (
        source_viewport.x < 0 || source_viewport.y < 0 || source_viewport.width <= 0 || source_viewport.height <= 0 ||
        source_viewport.x + source_viewport.width > self->width || source_viewport.y + source_viewport.height > self->height
    );

    const bool invalid_target = target->samples > 1 || !target->color;
    const bool invalid_source = !self->color;

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
    if ((self->ctx->current_clear_mask & 0xf) != 0xf) {
        self->ctx->current_clear_mask |= 0xf;
        self->ctx->current_global_settings = NULL;
        gl.ColorMaski(0, 1, 1, 1, 1);
    }
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, self->framebuffer->obj);
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, target ? target->framebuffer->obj : self->ctx->screen);
    gl.BlitFramebuffer(
        source_viewport.x, source_viewport.y, source_viewport.x + source_viewport.width, source_viewport.y + source_viewport.height,
        target_viewport.x, target_viewport.y, target_viewport.x + target_viewport.width, target_viewport.y + target_viewport.height,
        GL_COLOR_BUFFER_BIT, filter ? GL_LINEAR : GL_NEAREST
    );
    if (!target) {
        self->ctx->current_framebuffer = self->ctx->screen;
    }
    gl.BindFramebuffer(GL_FRAMEBUFFER, self->ctx->current_framebuffer);
    if (!srgb) {
        gl.Enable(GL_FRAMEBUFFER_SRGB);
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

void Context_dealloc(Context * self) {
    Py_DECREF(self->descriptor_set_buffers_cache);
    Py_DECREF(self->descriptor_set_images_cache);
    Py_DECREF(self->global_settings_cache);
    Py_DECREF(self->sampler_cache);
    Py_DECREF(self->vertex_array_cache);
    Py_DECREF(self->framebuffer_cache);
    Py_DECREF(self->program_cache);
    Py_DECREF(self->shader_cache);
    Py_DECREF(self->includes);
    Py_DECREF(self->limits);
    Py_DECREF(self->info);
    Py_TYPE(self)->tp_free(self);
}

void Buffer_dealloc(Buffer * self) {
    Py_DECREF(self->ctx);
    Py_TYPE(self)->tp_free(self);
}

void Image_dealloc(Image * self) {
    Py_DECREF(self->ctx);
    Py_DECREF(self->framebuffer);
    Py_DECREF(self->faces);
    Py_DECREF(self->size);
    Py_TYPE(self)->tp_free(self);
}

void Pipeline_dealloc(Pipeline * self) {
    Py_DECREF(self->ctx);
    Py_DECREF(self->descriptor_set_buffers);
    Py_DECREF(self->descriptor_set_images);
    Py_DECREF(self->global_settings);
    Py_DECREF(self->framebuffer);
    Py_DECREF(self->program);
    Py_DECREF(self->vertex_array);
    Py_TYPE(self)->tp_free(self);
}

void ImageFace_dealloc(ImageFace * self) {
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

PyMethodDef Context_methods[] = {
    {"buffer", (PyCFunction)Context_meth_buffer, METH_VARARGS | METH_KEYWORDS, NULL},
    {"image", (PyCFunction)Context_meth_image, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pipeline", (PyCFunction)Context_meth_pipeline, METH_VARARGS | METH_KEYWORDS, NULL},
    {"release", (PyCFunction)Context_meth_release, METH_O, NULL},
    {"reset", (PyCFunction)Context_meth_reset, METH_NOARGS, NULL},
    {},
};

PyMemberDef Context_members[] = {
    {"includes", T_OBJECT_EX, offsetof(Context, includes), READONLY, NULL},
    {"limits", T_OBJECT_EX, offsetof(Context, limits), READONLY, NULL},
    {"info", T_OBJECT_EX, offsetof(Context, info), READONLY, NULL},
    {"screen", T_INT, offsetof(Context, screen), 0, NULL},
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
    {"face", (PyCFunction)Image_meth_face, METH_VARARGS | METH_KEYWORDS, NULL},
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

PyMethodDef Pipeline_methods[] = {
    {"render", (PyCFunction)Pipeline_meth_render, METH_NOARGS, NULL},
    {},
};

PyGetSetDef Pipeline_getset[] = {
    {"viewport", (getter)Pipeline_get_viewport, (setter)Pipeline_set_viewport, NULL, NULL},
    {"_framebuffer", (getter)Pipeline_get_framebuffer, (setter)Pipeline_set_framebuffer, NULL, NULL},
    {},
};

PyMemberDef Pipeline_members[] = {
    {"vertex_count", T_INT, offsetof(Pipeline, vertex_count), 0, NULL},
    {"instance_count", T_INT, offsetof(Pipeline, instance_count), 0, NULL},
    {"first_vertex", T_INT, offsetof(Pipeline, first_vertex), 0, NULL},
    {},
};

PyMethodDef ImageFace_methods[] = {
    {"clear", (PyCFunction)ImageFace_meth_clear, METH_NOARGS, NULL},
    {"blit", (PyCFunction)ImageFace_meth_blit, METH_NOARGS, NULL},
    {},
};

PyMemberDef ImageFace_members[] = {
    {"image", T_OBJECT_EX, offsetof(ImageFace, image), READONLY, NULL},
    {"size", T_OBJECT_EX, offsetof(ImageFace, size), READONLY, NULL},
    {"layer", T_INT, offsetof(ImageFace, layer), READONLY, NULL},
    {"level", T_INT, offsetof(ImageFace, level), READONLY, NULL},
    {"samples", T_INT, offsetof(ImageFace, samples), READONLY, NULL},
    {"color", T_BOOL, offsetof(ImageFace, color), READONLY, NULL},
    {},
};

PyType_Slot Context_slots[] = {
    {Py_tp_methods, Context_methods},
    {Py_tp_members, Context_members},
    {Py_tp_dealloc, (void *)Context_dealloc},
    {},
};

PyType_Slot Buffer_slots[] = {
    {Py_tp_methods, Buffer_methods},
    {Py_tp_members, Buffer_members},
    {Py_tp_dealloc, (void *)Buffer_dealloc},
    {},
};

PyType_Slot Image_slots[] = {
    {Py_tp_methods, Image_methods},
    {Py_tp_getset, Image_getset},
    {Py_tp_members, Image_members},
    {Py_tp_dealloc, (void *)Image_dealloc},
    {},
};

PyType_Slot Pipeline_slots[] = {
    {Py_tp_methods, Pipeline_methods},
    {Py_tp_getset, Pipeline_getset},
    {Py_tp_members, Pipeline_members},
    {Py_tp_dealloc, (void *)Pipeline_dealloc},
    {},
};

PyType_Slot ImageFace_slots[] = {
    {Py_tp_methods, ImageFace_methods},
    {Py_tp_members, ImageFace_members},
    {Py_tp_dealloc, (void *)ImageFace_dealloc},
    {},
};

PyType_Slot DescriptorSetBuffers_slots[] = {
    {Py_tp_dealloc, (void *)DescriptorSetBuffers_dealloc},
    {},
};

PyType_Slot DescriptorSetImages_slots[] = {
    {Py_tp_dealloc, (void *)DescriptorSetImages_dealloc},
    {},
};

PyType_Slot GlobalSettings_slots[] = {
    {Py_tp_dealloc, (void *)GlobalSettings_dealloc},
    {},
};

PyType_Slot GLObject_slots[] = {
    {Py_tp_dealloc, (void *)GLObject_dealloc},
    {},
};

PyType_Spec Context_spec = {"zengl.Context", sizeof(Context), 0, Py_TPFLAGS_DEFAULT, Context_slots};
PyType_Spec Buffer_spec = {"zengl.Buffer", sizeof(Buffer), 0, Py_TPFLAGS_DEFAULT, Buffer_slots};
PyType_Spec Image_spec = {"zengl.Image", sizeof(Image), 0, Py_TPFLAGS_DEFAULT, Image_slots};
PyType_Spec Pipeline_spec = {"zengl.Pipeline", sizeof(Pipeline), 0, Py_TPFLAGS_DEFAULT, Pipeline_slots};
PyType_Spec ImageFace_spec = {"zengl.ImageFace", sizeof(ImageFace), 0, Py_TPFLAGS_DEFAULT, ImageFace_slots};
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
    state->Context_type = (PyTypeObject *)PyType_FromSpec(&Context_spec);
    state->Buffer_type = (PyTypeObject *)PyType_FromSpec(&Buffer_spec);
    state->Image_type = (PyTypeObject *)PyType_FromSpec(&Image_spec);
    state->Pipeline_type = (PyTypeObject *)PyType_FromSpec(&Pipeline_spec);
    state->ImageFace_type = (PyTypeObject *)PyType_FromSpec(&ImageFace_spec);
    state->DescriptorSetBuffers_type = (PyTypeObject *)PyType_FromSpec(&DescriptorSetBuffers_spec);
    state->DescriptorSetImages_type = (PyTypeObject *)PyType_FromSpec(&DescriptorSetImages_spec);
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

PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, (void *)module_exec},
    {},
};

PyMethodDef module_methods[] = {
    {"context", (PyCFunction)meth_context, METH_VARARGS | METH_KEYWORDS, NULL},
    {"inspect", (PyCFunction)meth_inspect, METH_O, NULL},
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
    Py_DECREF(state->Context_type);
    Py_DECREF(state->Buffer_type);
    Py_DECREF(state->Image_type);
    Py_DECREF(state->Pipeline_type);
    Py_DECREF(state->ImageFace_type);
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
