import os
import textwrap

from OpenGL import GL

ERROR_CODES = {
    0x0500: {
        'name': 'GL_INVALID_ENUM',
        'description': textwrap.dedent('''
            Given when an enumeration parameter is not a legal enumeration for that function. This is given only for
            local problems; if the spec allows the enumeration in certain circumstances, where other parameters or
            state dictate those circumstances, then GL_INVALID_OPERATION is the result instead.
        '''),
    },
    0x0501: {
        'name': 'GL_INVALID_VALUE',
        'description': textwrap.dedent('''
            Given when a value parameter is not a legal value for that function. This is only given for local problems;
            if the spec allows the value in certain circumstances, where other parameters or state dictate those
            circumstances, then GL_INVALID_OPERATION is the result instead.
        '''),
    },
    0x0502: {
        'name': 'GL_INVALID_OPERATION',
        'description': textwrap.dedent('''
            Given when the set of state for a command is not legal for the parameters given to that command.
            It is also given for commands where combinations of parameters define what the legal parameters are.
        '''),
    },
    0x0503: {
        'name': 'GL_STACK_OVERFLOW',
        'description': textwrap.dedent('''
            Given when a stack pushing operation cannot be done because it would overflow
            the limit of that stack's size.
        '''),
    },
    0x0504: {
        'name': 'GL_STACK_UNDERFLOW',
        'description': textwrap.dedent('''
            Given when a stack popping operation cannot be done because the stack is already at its lowest point.
        '''),
    },
    0x0505: {
        'name': 'GL_OUT_OF_MEMORY',
        'description': textwrap.dedent('''
            Given when performing an operation that can allocate memory, and the memory cannot be allocated.
            The results of OpenGL functions that return this error are undefined; it is allowable for partial
            execution of an operation to happen in this circumstance.
        '''),
    },
    0x0506: {
        'name': 'GL_INVALID_FRAMEBUFFER_OPERATION',
        'description': textwrap.dedent('''
            Given when doing anything that would attempt to read from or
            write/render to a framebuffer that is not complete.
        '''),
    },
    0x0507: {
        'name': 'GL_CONTEXT_LOST',
        'description': textwrap.dedent('''
            Given if the OpenGL context has been lost, due to a graphics card reset.
        '''),
    }
}


def clear_gl_error():
    GL.glGetError()


def assert_gl_error(hint=''):
    error = GL.glGetError()
    if not error:
        return

    info = ERROR_CODES.get(error)
    if not info:
        assert False, f'{hint}\nglGetError() = {error}'

    name = info['name']
    description = info['description']
    assert False, f'{hint}\nglGetError() = {error} | {name}\n{description}'
