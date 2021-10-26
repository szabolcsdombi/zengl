from typing import Any, Dict, Iterable, List, Literal, Tuple, TypedDict, Union

Topology = Literal['points', 'lines', 'line_loop', 'line_strip', 'triangles', 'triangle_strip', 'triangle_fan']
FrontFace = Literal['cw', 'ccw']
CullFace = Literal['front', 'back', 'front_and_back', 'none']
MinFilter = Literal['nearest', 'linear', 'nearest_mipmap_nearest', 'linear_mipmap_nearest', 'nearest_mipmap_linear', 'linear_mipmap_linear']
MagFilter = Literal['nearest', 'linear']
TextureWrap = Literal['repeat', 'clamp_to_edge', 'mirrored_repeat']
CompareMode = Literal['ref_to_texture', 'none']
CompareFunc = Literal['never', 'less', 'equal', 'lequal', 'greater', 'notequal', 'gequal', 'always']
StencilOp = Literal['zero', 'keep', 'replace', 'incr', 'decr', 'invert', 'incr_wrap', 'decr_wrap']
Step = Literal['vertex', 'instance']

BlendConstant = Literal[
    'zero', 'one', 'src_color', 'one_minus_src_color', 'src_alpha', 'one_minus_src_alpha', 'dst_alpha', 'one_minus_dst_alpha',
    'dst_color', 'one_minus_dst_color', 'src_alpha_saturate', 'constant_color', 'one_minus_constant_color', 'constant_alpha',
    'one_minus_constant_alpha', 'src1_alpha', 'src1_color', 'one_minus_src1_color', 'one_minus_src1_alpha',
]

VertexFormatShort = Literal[
    '2u1', '4u1', '2i1', '4i1', '2nu1', '4nu1', '2ni1', '4ni1', '2u2', '4u2', '2i2', '4i2', '2nu2', '4nu2', '2ni2', '4ni2',
    '2h', '4h', '1f', '2f', '3f', '4f', '1u', '2u', '3u', '4u', '1i', '2i', '3i', '4i',
]

VertexFormat = Literal[
    'uint8x2', 'uint8x4', 'sint8x2', 'sint8x4', 'unorm8x2', 'unorm8x4', 'snorm8x2', 'snorm8x4', 'uint16x2', 'uint16x4', 'sint16x2',
    'sint16x4', 'unorm16x2', 'unorm16x4', 'snorm16x2', 'snorm16x4', 'float16x2', 'float16x4', 'float32', 'float32x2', 'float32x3',
    'float32x4', 'uint32', 'uint32x2', 'uint32x3', 'uint32x4', 'sint32', 'sint32x2', 'sint32x3', 'sint32x4',
]

ImageFormat = Literal[
    'uint8x2', 'uint8x4', 'sint8x2', 'sint8x4', 'unorm8x2', 'unorm8x4', 'snorm8x2', 'snorm8x4', 'uint16x2', 'uint16x4', 'sint16x2',
    'sint16x4', 'unorm16x2', 'unorm16x4', 'snorm16x2', 'snorm16x4', 'float16x2', 'float16x4', 'float32', 'float32x2', 'float32x3',
    'float32x4', 'uint32', 'uint32x2', 'uint32x3', 'uint32x4', 'sint32', 'sint32x2', 'sint32x3', 'sint32x4', 'r8unorm', 'rg8unorm',
    'rgba8unorm', 'bgra8unorm', 'r8snorm', 'rg8snorm', 'rgba8snorm', 'r8uint', 'rg8uint', 'rgba8uint', 'r16uint', 'rg16uint',
    'rgba16uint', 'r32uint', 'rg32uint', 'rgba32uint', 'r8sint', 'rg8sint', 'rgba8sint', 'r16sint', 'rg16sint', 'rgba16sint',
    'r32sint', 'rg32sint', 'rgba32sint', 'r16float', 'rg16float', 'rgba16float', 'r32float', 'rg32float', 'rgba32float',
    'rgba8unorm-srgb', 'bgra8unorm-srgb', 'stencil8', 'depth16unorm', 'depth24plus', 'depth24plus-stencil8', 'depth32float',
]


class PolygonOffsetSettings(TypedDict, total=False):
    factor: float
    units: float


class LayoutBinding(TypedDict, total=False):
    name: str
    binding: int


class BufferResourceBinding(TypedDict, total=False):
    type: Literal['uniform_buffer']
    binding: int
    buffer: 'Buffer'
    offset: int
    size: int


class ImageResourceBinding(TypedDict, total=False):
    type: Literal['sampler']
    binding: int
    image: 'Image'
    min_filter: MinFilter
    mag_filter: MagFilter
    min_lod: float
    max_lod: float
    wrap_x: TextureWrap
    wrap_y: TextureWrap
    wrap_z: TextureWrap
    compare_mode: CompareMode
    compare_func: CompareFunc
    border_color: Tuple[float, float, float, float]


ResourceBinding = BufferResourceBinding | ImageResourceBinding


class VertexBufferBinding(TypedDict, total=False):
    buffer: 'Buffer'
    format: VertexFormat
    location: int
    offset: int
    stride: int
    step: Step


class DepthSettings(TypedDict, total=False):
    test: bool
    write: bool
    func: CompareFunc


class StencilFaceSettings(TypedDict, total=False):
    fail_op: StencilOp
    pass_op: StencilOp
    depth_fail_op: StencilOp
    compare_op: CompareFunc
    compare_mask: int
    write_mask: int
    reference: int


class StencilSettings(TypedDict, total=False):
    test: bool
    front: StencilFaceSettings
    back: StencilFaceSettings
    both: StencilFaceSettings


class BlendingSettings(TypedDict, total=False):
    enable: bool | int
    src_color: BlendConstant
    dst_color: BlendConstant
    src_alpha: BlendConstant
    dst_alpha: BlendConstant


Viewport = Tuple[int, int, int, int] | Tuple[int, int] | None


class Context:
    def load(name: str) -> int: ...


class Buffer:
    size: int
    def write(self, data: bytes, /, offset: int = 0): ...
    def map(self, /, size: int = -1, *, offset: int = 0, discard: bool = False) -> memoryview: ...
    def unmap(self) -> None: ...


class Image:
    size: Tuple[int, int]
    samples: int
    color: bool
    clear_value: Iterable[int | float] | int | float
    def clear(self): ...
    def write(self, data: bytes, /, size: Tuple[int, int] = ..., offset: Tuple[int, int] = (0, 0), layer: int = 0): ...
    def mipmaps(self, /, *, base: int = 0, levels: int = -1): ...
    def read(self, /, size: Tuple[int, int] = ..., *, offset: Tuple[int, int] = (0, 0)): ...
    def blit(
        self, /, dst: 'Image' | None = None, *, dst_size: Tuple[int, int] = ..., dst_offset: Tuple[int, int] = (0, 0),
        src_size: Tuple[int, int] = ..., src_offset: Tuple[int, int] = (0, 0), filter: bool = True): ...


class Renderer:
    vertex_count: int
    instance_count: int
    viewport: Viewport
    def render(self) -> None: ...


class Instance:
    files: Dict[str, str]
    def buffer(self, /, data: bytes | None = None, *, size: int = 0, dynamic: bool = False) -> Buffer: ...
    def image(self, size: Tuple[int, int], format: str, /, data: bytes | None = None, *, samples: int = 1, texture: bool | None = None) -> Image: ...
    def renderer(
        self, /, *, vertex_shader: str = ..., fragment_shader: str = ..., framebuffer: Iterable[Image] = (), topology: Topology = 'triangles',
        layout: Iterable[LayoutBinding] = (), resources: Iterable[ResourceBinding] = (), vertex_buffers: Iterable[VertexBufferBinding] = (),
        index_buffer: Buffer | None = None, vertex_count: int = 0, instance_count: int = 0, short_index: bool = False,
        primitive_restart: bool = False, point_size: float = 1.0, line_width: float = 1.0, front_face: str = 'ccw', cull_face: str = 'none',
        color_mask: int = 0xffffffffffffffff, depth: DepthSettings | bool | None = None, stencil: StencilSettings | None = None,
        blending: BlendingSettings | None = None, polygon_offset: PolygonOffsetSettings | None = None, viewport: Viewport = None) -> Renderer: ...


def instance(context: Context | Any) -> Instance: ...
def camera(
    eye: Tuple[float, float, float], target: Tuple[float, float, float], /, up: Tuple[float, float, float] = (0.0, 0.0, 1.0), *,
    fov: float = 45.0, aspect: float = 1.0, near: float = 0.1, far: float = 1000.0, size: float = 1.0, clip: bool = False) -> bytes: ...
def rgba(data: bytes, format: str) -> bytes: ...
def pack(*arg: Iterable[Union[float, int]]) -> bytes: ...
def bind(buffer: Buffer, layout: str, *attributes: Iterable[int]) -> List[VertexBufferBinding]: ...
def calcsize(layout: str) -> int: ...
def context(headless: bool = False) -> Context: ...
