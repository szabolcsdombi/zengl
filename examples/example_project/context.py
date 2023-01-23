import zengl


class Context:
    context = None
    main_uniform_buffer = None
    main_uniform_buffer_data = bytearray(b'\x00' * 64)

    @classmethod
    def initialize(cls):
        ctx = zengl.context()
        cls.context = ctx
        cls.main_uniform_buffer = ctx.buffer(size=64)
        ctx.includes['main_uniform_buffer'] = '''
            layout (std140, binding = 0) uniform MainUniformBuffer {
                mat4 mvp;
            };
        '''

    @classmethod
    def update_camera(cls, eye, target, aspect, fov):
        cls.main_uniform_buffer_data[0:64] = zengl.camera(eye, target, aspect=aspect, fov=fov)

    @classmethod
    def flush_uniform_buffer(cls):
        cls.main_uniform_buffer.write(cls.main_uniform_buffer_data)
