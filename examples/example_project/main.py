import assets
from blur import Blur
from context import Context
from object_manager import ObjectManager
from simple_framebuffer import SimpleFramebuffer
from simple_model import SimpleModel
from window import Window

wnd = Window(1280, 720)
Context.initialize()

framebuffer = SimpleFramebuffer(wnd.size)
blur = Blur(framebuffer.image)

object_manager = ObjectManager(framebuffer)

monkey_model = SimpleModel(assets.get('monkey.obj'))
blob_model = SimpleModel(assets.get('blob.obj'))
box_model = SimpleModel(assets.get('box.obj'))

monkey = object_manager.model(monkey_model)
blob1 = object_manager.model(blob_model)
blob2 = object_manager.model(blob_model)
blob3 = object_manager.model(blob_model)
box = object_manager.model(box_model)

while wnd.update():
    Context.update_camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.5), aspect=wnd.aspect, fov=45.0)

    box.position = (0.0, -3.0, 0.5)
    blob1.position = (-4.0, 0.0, 0.0)
    blob2.position = (-4.0, -1.5, 0.0)
    blob3.position = (-4.0, -3.0, 0.0)

    Context.flush_uniform_buffer()

    framebuffer.clear(0.2, 0.2, 0.2)
    object_manager.render()

    # framebuffer.image.blit()

    blur.render()
    blur.output_image.blit()
