from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple, TypedDict

CullFace = Literal['front', 'back', 'front_and_back', 'none']
Topology = Literal['points', 'lines', 'line_loop', 'line_strip', 'triangles', 'triangle_strip', 'triangle_fan']

MinFilter = Literal[
    'nearest', 'linear', 'nearest_mipmap_nearest', 'linear_mipmap_nearest', 'nearest_mipmap_linear',
    'linear_mipmap_linear',
]

MagFilter = Literal['nearest', 'linear']
TextureWrap = Literal['repeat', 'clamp_to_edge', 'mirrored_repeat']
CompareMode = Literal['ref_to_texture', 'none']
CompareFunc = Literal['never', 'less', 'equal', 'lequal', 'greater', 'notequal', 'gequal', 'always']
StencilOp = Literal['zero', 'keep', 'replace', 'incr', 'decr', 'invert', 'incr_wrap', 'decr_wrap']
Step = Literal['vertex', 'instance']

BlendConstant = Literal[
    'zero', 'one', 'src_color', 'one_minus_src_color', 'src_alpha', 'one_minus_src_alpha', 'dst_alpha',
    'one_minus_dst_alpha', 'dst_color', 'one_minus_dst_color', 'src_alpha_saturate', 'constant_color',
    'one_minus_constant_color', 'constant_alpha', 'one_minus_constant_alpha', 'src1_alpha', 'src1_color',
    'one_minus_src1_color', 'one_minus_src1_alpha',
]

VertexFormatShort = Literal[
    '2u1', '4u1', '2i1', '4i1', '2nu1', '4nu1', '2ni1', '4ni1', '2u2', '4u2', '2i2', '4i2', '2nu2', '4nu2', '2ni2',
    '4ni2', '2h', '4h', '1f', '2f', '3f', '4f', '1u', '2u', '3u', '4u', '1i', '2i', '3i', '4i',
]

VertexFormat = Literal[
    'uint8x2', 'uint8x4', 'sint8x2', 'sint8x4', 'unorm8x2', 'unorm8x4', 'snorm8x2', 'snorm8x4', 'uint16x2', 'uint16x4',
    'sint16x2', 'sint16x4', 'unorm16x2', 'unorm16x4', 'snorm16x2', 'snorm16x4', 'float16x2', 'float16x4', 'float32',
    'float32x2', 'float32x3', 'float32x4', 'uint32', 'uint32x2', 'uint32x3', 'uint32x4', 'sint32', 'sint32x2',
    'sint32x3', 'sint32x4',
]

ImageFormat = Literal[
    'r8unorm', 'rg8unorm', 'rgba8unorm', 'r8snorm', 'rg8snorm', 'rgba8snorm', 'r8uint', 'rg8uint',
    'rgba8uint', 'r16uint', 'rg16uint', 'rgba16uint', 'r32uint', 'rg32uint', 'rgba32uint', 'r8sint', 'rg8sint',
    'rgba8sint', 'r16sint', 'rg16sint', 'rgba16sint', 'r32sint', 'rg32sint', 'rgba32sint', 'r16float', 'rg16float',
    'rgba16float', 'r32float', 'rg32float', 'rgba32float', 'rgba8unorm-srgb', 'stencil8',
    'depth16unorm', 'depth24plus', 'depth24plus-stencil8', 'depth32float',
]

Vec3 = Tuple[float, float, float]
Viewport = Tuple[int, int, int, int]
Bytes = bytes | Any


class UniformBufferResource(TypedDict, total=False):
    type: Literal['uniform_buffer']
    binding: int
    buffer: 'Buffer'
    offset: int
    size: int


class StorageBufferResource(TypedDict, total=False):
    type: Literal['storage_buffer']
    binding: int
    buffer: 'Buffer'
    offset: int
    size: int


class SamplerResource(TypedDict, total=False):
    type: Literal['sampler']
    binding: int
    image: 'Image'
    min_filter: MinFilter
    mag_filter: MagFilter
    min_lod: float
    max_lod: float
    lod_bias: float
    wrap_x: TextureWrap
    wrap_y: TextureWrap
    wrap_z: TextureWrap
    compare_mode: CompareMode
    compare_func: CompareFunc
    max_anisotropy: float


class ImageResource(TypedDict, total=False):
    type: Literal['image']
    binding: int
    image: 'Image'


Resources = UniformBufferResource | StorageBufferResource | SamplerResource | ImageResource


class VertexBufferBinding(TypedDict, total=False):
    buffer: 'Buffer'
    format: VertexFormat
    location: int
    offset: int
    stride: int
    step: Step


class DepthSettings(TypedDict, total=False):
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
    front: StencilFaceSettings
    back: StencilFaceSettings
    both: StencilFaceSettings


class BlendingSettings(TypedDict, total=False):
    enable: bool | int
    src_color: BlendConstant
    dst_color: BlendConstant
    src_alpha: BlendConstant
    dst_alpha: BlendConstant


class Info(TypedDict):
    vendor: str
    renderer: str
    version: str
    glsl: str


class Limits(TypedDict):
    max_uniform_buffer_bindings: int
    max_uniform_block_size: int
    max_combined_uniform_blocks: int
    max_shader_storage_buffer_bindings: int
    max_shader_storage_block_size: int
    max_combined_shader_storage_blocks: int
    max_combined_texture_image_units: int
    max_image_units: int
    max_vertex_attribs: int
    max_draw_buffers: int
    max_samples: int


class ImageFace:
    image: 'Image'
    size: Tuple[int, int]
    samples: int
    color: bool
    def clear(self) -> None: ...
    def blit(
        self, target: 'ImageFace', target_viewport: Viewport | None = None,
        source_viewport: Viewport | None = None, filter: bool = True, srgb: bool | None = None) -> None: ...


class ContextLoader:
    def load_opengl_function(name: str) -> int: ...
    def load(name: str) -> int: ...


class Buffer:
    size: int
    def write(self, data: Bytes, offset: int = 0) -> None: ...
    def read(self, size: int = ..., offset: int = 0) -> bytes: ...
    def map(self, size: int | None = None, offset: int | None = None, discard: bool = False) -> memoryview: ...
    def unmap(self) -> None: ...


class Image:
    size: Tuple[int, int]
    samples: int
    color: bool
    clear_value: Iterable[int | float] | int | float
    def face(self, layer: int = 0, level: int = 0) -> ImageFace: ...
    def clear(self) -> None: ...
    def write(
        self, data: Bytes, size: Tuple[int, int] | None = None,
        offset: Tuple[int, int] | None = None, layer: int | None = None, level: int = 0) -> None: ...
    def mipmaps(self, base: int = 0, levels: int | None = None) -> None: ...
    def read(self, size: Tuple[int, int] | None = None, offset: Tuple[int, int] | None = None) -> bytes: ...
    def blit(
        self, target: 'Image' | None = None, target_viewport: Viewport | None = None,
        source_viewport: Viewport | None = None, filter: bool = True, srgb: bool | None = None,
        flush: bool | None = None) -> None: ...


class Pipeline:
    vertex_count: int
    instance_count: int
    first_vertex: int
    viewport: Viewport
    uniforms: Dict[str, memoryview] | None
    def render(self) -> None: ...


class Compute:
    group_count: int
    uniforms: Dict[str, memoryview] | None
    def run(self) -> None: ...


class Context:
    info: Info
    limits: Limits
    includes: Dict[str, str]
    before_frame: Callable | None
    after_frame: Callable | None
    frame_time: int
    screen: int
    def buffer(
        self, data: Bytes | None = None, size: int | None = None, external: int = 0) -> Buffer: ...
    def image(
        self, size: Tuple[int, int], format: ImageFormat, data: Bytes | None = None,
        samples: int = 1, array: int = 0, levels: int = ..., texture: bool | None = None, cubemap: bool = False,
        external: int = 0) -> Image: ...
    def pipeline(
        self,
        vertex_shader: str = ...,
        fragment_shader: str = ...,
        resources: Iterable[Resources] = (),
        uniforms: Dict[str, Any] | None = None,
        depth: DepthSettings | None = None,
        stencil: StencilSettings | None = None,
        blend: Iterable[BlendingSettings] = (),
        framebuffer: Iterable[Image] = (),
        empty_framebuffer: Tuple[int, int] | None = None,
        vertex_buffers: Iterable[VertexBufferBinding] = (),
        index_buffer: Buffer | None = None,
        indirect_buffer: Buffer | None = None,
        short_index: bool = False,
        cull_face: CullFace = 'none',
        topology: Topology = 'triangles',
        vertex_count: int = 0,
        instance_count: int = 0,
        indirect_count: int = 0,
        first_vertex: int = 0,
        dynamic_state: memoryview | None = None,
        viewport: Viewport | None = None,
        includes: Dict[str, str] | None = None) -> Pipeline: ...
    def compute(
        self,
        compute_shader: str = ...,
        resources: Iterable[Resources] = (),
        uniforms: Dict[str, Any] | None = None,
        group_count: Tuple[int, int, int] = (0, 0, 0),
        includes: Dict[str, str] | None = None) -> Pipeline: ...
    def barrier(self) -> None: ...
    def new_frame(self, reset: bool = True, frame_time: bool = False) -> None: ...
    def end_frame(self, clean: bool = True, flush: bool = True, sync: bool = False) -> None: ...
    def release(self, obj: Buffer | Image | Pipeline | Compute | Literal['shader_cache'] | Literal['all']) -> None: ...


def context(loader: ContextLoader | Any | None = None) -> Context: ...
def inspect(self, obj: Buffer | Image | Pipeline): ...
def camera(
    eye: Vec3, target: Vec3, up: Vec3 = (0.0, 0.0, 1.0),
    fov: float = 45.0, aspect: float = 1.0, near: float = 0.1, far: float = 1000.0,
    size: float = 1.0, clip: bool = False) -> bytes: ...
def bind(buffer: Buffer, layout: str, *attributes: Iterable[int]) -> List[VertexBufferBinding]: ...
def calcsize(layout: str) -> int: ...
def loader(headless: bool = False) -> ContextLoader: ...
