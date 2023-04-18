#include <Python.h>
#include <structmember.h>

extern int zengl_get_frame();
extern double zengl_get_time();
extern void zengl_get_mouse(int * mouse);
extern void zengl_get_mouse_delta(int * mouse);
extern void zengl_get_size(int * size);
extern int zengl_get_key(const char * key);

typedef struct Window {
    PyObject_HEAD
} Window;

static PyTypeObject * Window_type;

static PyObject * Window_meth_key_pressed(Window * self, PyObject * arg) {
    if (!PyUnicode_CheckExact(arg)) {
        return NULL;
    }
    int flags = zengl_get_key(PyUnicode_AsUTF8(arg));
    if ((flags & 1) && (~flags & 2)) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject * Window_meth_key_down(Window * self, PyObject * arg) {
    if (!PyUnicode_CheckExact(arg)) {
        return NULL;
    }
    int flags = zengl_get_key(PyUnicode_AsUTF8(arg));
    if (flags & 1) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject * Window_meth_key_release(Window * self, PyObject * arg) {
    if (!PyUnicode_CheckExact(arg)) {
        return NULL;
    }
    int flags = zengl_get_key(PyUnicode_AsUTF8(arg));
    if ((~flags & 1) & (flags & 2)) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject * Window_get_size(Window * self, void * closure) {
    int size[2];
    zengl_get_size(size);
    return Py_BuildValue("(ii)", size[0], size[1]);
}

static PyObject * Window_get_aspect(Window * self, void * closure) {
    int size[2];
    zengl_get_size(size);
    return PyFloat_FromDouble((double)size[0] / (double)size[1]);
}

static PyObject * Window_get_mouse(Window * self, void * closure) {
    int mouse[2];
    zengl_get_mouse(mouse);
    return Py_BuildValue("(ii)", mouse[0], mouse[1]);
}

static PyObject * Window_get_time(Window * self, void * closure) {
    return PyFloat_FromDouble(zengl_get_time());
}

static PyObject * Window_get_frame(Window * self, void * closure) {
    return PyLong_FromLong(zengl_get_frame());
}

static void default_dealloc(PyObject * self) {
    Py_TYPE(self)->tp_free(self);
}

static PyMethodDef Window_methods[] = {
    {"key_pressed", (PyCFunction)Window_meth_key_pressed, METH_O},
    {"key_down", (PyCFunction)Window_meth_key_down, METH_O},
    {"key_release", (PyCFunction)Window_meth_key_release, METH_O},
    {NULL},
};

static PyGetSetDef Window_getset[] = {
    {"size", (getter)Window_get_size, NULL},
    {"aspect", (getter)Window_get_aspect, NULL},
    {"mouse", (getter)Window_get_mouse, NULL},
    {"time", (getter)Window_get_time, NULL},
    {"frame", (getter)Window_get_frame, NULL},
    {NULL},
};

static PyType_Slot Window_slots[] = {
    {Py_tp_methods, Window_methods},
    {Py_tp_getset, Window_getset},
    {Py_tp_dealloc, default_dealloc},
    {0},
};

static PyType_Spec Window_spec = {"zengl_canvas.Window", sizeof(Window), 0, Py_TPFLAGS_DEFAULT, Window_slots};

static PyMethodDef module_methods[] = {
    {NULL},
};

static PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "zengl_canvas", NULL, -1, module_methods};

extern PyObject * PyInit_zengl_canvas() {
    PyObject * module = PyModule_Create(&module_def);
    Window_type = (PyTypeObject *)PyType_FromSpec(&Window_spec);

    Window * window = PyObject_New(Window, Window_type);
    PyModule_AddObject(module, "window", (PyObject *)window);

    return module;
}
