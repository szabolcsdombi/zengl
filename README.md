# ZenGL

[![ZenGL](https://repository-images.githubusercontent.com/420309094/f7c17e13-4d5b-4a38-8b52-ab2dfdacd5a0)](#zengl)

```
pip install zengl
```

- [Documentation](https://zengl.readthedocs.io/)
- [zengl on Github](https://github.com/szabolcsdombi/zengl/)
- [zengl on PyPI](https://pypi.org/project/zengl/)

## Concept

ZenGL provides a simple way to render from Python. We aim to support headless rendering first,
rendering to a window is done by blitting the final image to the screen. By doing this we have full control of
what we render. The window does not have to be multisample, and it requires no depth buffer at all.

[read more...](https://zengl.readthedocs.io/en/latest/#concept)

## Examples

```
pip install zengl[examples]
```

#### [grass.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/grass.py)

[![grass](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/grass.png)](#grasspy)

#### [envmap.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/envmap.py)

[![envmap](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/envmap.png)](#envmappy)

#### [normal_mapping.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/normal_mapping.py)

[![normal_mapping](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/normal_mapping.jpg)](#normal_mappingpy)

#### [rigged_objects.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/rigged_objects.py)

[![rigged_objects](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/rigged_objects.png)](#rigged_objectspy)

#### [instanced_crates.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/instanced_crates.py)

[![instanced_crates](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/instanced_crates.jpg)](#instanced_cratespy)

#### [julia_fractal.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/julia_fractal.py)

[![julia_fractal](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/julia_fractal.png)](#julia_fractalpy)

#### [blending.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/blending.py)

[![blending](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/blending.png)](#blendingpy)

#### [render_to_texture.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/render_to_texture.py)

[![render_to_texture](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/render_to_texture.png)](#render_to_texturepy)

#### [pybullet_box_pile.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/pybullet_box_pile.py)

[![pybullet_box_pile](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/pybullet_box_pile.png)](#pybullet_box_pilepy)

#### [pygmsh_shape.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/pygmsh_shape.py)

[![pygmsh_shape](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/pygmsh_shape.png)](#pygmsh_shapepy)

#### [texture_array.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/texture_array.py)

[![texture_array](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/texture_array.png)](#texture_arraypy)

#### [monkey.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/monkey.py)

[![monkey](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/monkey.png)](#monkeypy)

#### [reflection.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/reflection.py)

[![reflection](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/reflection.png)](#reflectionpy)

#### [polygon_offset.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/polygon_offset.py)

[![polygon_offset](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/polygon_offset.png)](#polygon_offsetpy)

#### [blur.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/blur.py)

[![blur](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/blur.png)](#blurpy)

#### [hello_triangle.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/hello_triangle.py)

[![hello_triangle](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/hello_triangle.png)](#hello_trianglepy)

#### [hello_triangle_srgb.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/hello_triangle_srgb.py)

[![hello_triangle_srgb](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/hello_triangle_srgb.png)](#hello_triangle_srgbpy)

#### [viewports.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/viewports.py)

[![viewports](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/viewports.png)](#viewportspy)

#### [points.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/points.py)

[![points](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/points.png)](#pointspy)

#### [wireframe_terrain.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/wireframe_terrain.py)

[![wireframe_terrain](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/wireframe_terrain.png)](#wireframe_terrainpy)

#### [crate.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/crate.py)

[![crate](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/crate.png)](#cratepy)

#### [sdf_example.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/sdf_example.py)

[![sdf_example](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/sdf_example.png)](#sdf_examplepy)

#### [sdf_tree.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/sdf_tree.py)

[![sdf_tree](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/sdf_tree.png)](#sdf_treepy)

#### [mipmaps.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/mipmaps.py)

[![mipmaps](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/mipmaps.png)](#mipmapspy)

#### [conways_game_of_life.py](https://github.com/szabolcsdombi/zengl/blob/main/examples/conways_game_of_life.py)

[![conways_game_of_life](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/conways_game_of_life.png)](#conways_game_of_lifepy)

#### Headless

```py
import zengl
from PIL import Image

ctx = zengl.context(zengl.loader(headless=True))

size = (1280, 720)
image = ctx.image(size, 'rgba8unorm', samples=1)

triangle = ctx.pipeline(
    vertex_shader='''
        #version 330

        out vec3 v_color;

        vec2 positions[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.6, -0.8),
            vec2(0.6, -0.8)
        );

        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            v_color = colors[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 330

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

image.clear_value = (1.0, 1.0, 1.0, 1.0)
image.clear()
triangle.render()

Image.frombuffer('RGBA', size, image.read(), 'raw', 'RGBA', 0, -1).save('hello.png')
```

### Type Hints

[![linting_01](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/linting/linting_01.png)](#typehints)

[![linting_02](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/linting/linting_02.png)](#typehints)

[![linting_03](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/linting/linting_03.png)](#typehints)

[![linting_04](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/linting/linting_04.png)](#typehints)

[![linting_05](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/linting/linting_05.png)](#typehints)

[![linting_06](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/linting/linting_06.png)](#typehints)

[![linting_07](https://github.com/szabolcsdombi/zengl-example-images/raw/examples/linting/linting_07.png)](#typehints)
