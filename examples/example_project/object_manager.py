import struct
from context import Context


class Instance:
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    @property
    def position(self):
        return struct.unpack('fff', self.parent.data[self.index * 16 : self.index * 16 + 12])

    @position.setter
    def position(self, value):
        self.parent.data[self.index * 16 : self.index * 16 + 12] = struct.pack('fff', *value)


class ObjectManager:
    def __init__(self, framebuffer):
        ctx = Context.context
        self.framebuffer = framebuffer.framebuffer
        self.buffer = ctx.buffer(size=1024)
        self.data = bytearray(b'\x00' * 1024)
        self.pipelines = []
        self.index = 0

    def model(self, model):
        index = self.index
        resource = {
            'type': 'uniform_buffer',
            'binding': 1,
            'buffer': self.buffer,
            'offset': self.index * 16,
            'size': self.index * 16 + 16,
        }
        self.index += 1
        self.pipelines.append(model.pipeline(resource, self.framebuffer))
        return Instance(self, index)

    def render(self):
        self.buffer.write(self.data)
        for pipeline in self.pipelines:
            pipeline.render()
