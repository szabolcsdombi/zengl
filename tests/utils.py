from OpenGL import GL

ERROR_CODES = {
    0x0500: 'GL_INVALID_ENUM',
    0x0501: 'GL_INVALID_VALUE',
    0x0502: 'GL_INVALID_OPERATION',
    0x0503: 'GL_STACK_OVERFLOW',
    0x0504: 'GL_STACK_UNDERFLOW',
    0x0505: 'GL_OUT_OF_MEMORY',
    0x0506: 'GL_INVALID_FRAMEBUFFER_OPERATION',
    0x0507: 'GL_CONTEXT_LOST',
}


def clear_gl_error():
    GL.glGetError()


def assert_gl_error(hint=''):
    error = GL.glGetError()
    if not error:
        return

    name = ERROR_CODES.get(error, '')
    raise AssertionError(f'{hint}\nglGetError() = {error} | {name}')
