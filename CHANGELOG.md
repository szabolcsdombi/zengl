# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

# [main](https://github.com/szabolcsdombi/zengl/compare/2.7.1...main)

- Fixed swapped paremeters `depth_fail_op`, `pass_op` in `glStencilOpSeparate`
- Added missing stubs for blend functions

# [2.7.1](https://github.com/szabolcsdombi/zengl/compare/2.7.0...2.7.1)

- Fixed the Pipeline index buffer binding changed by buffer create

# [2.7.0](https://github.com/szabolcsdombi/zengl/compare/2.6.1...2.7.0)

- Removed the `frame_time` parameter from `Context.new_frame` method
- Removed the `sync` parameter from `Context.end_frame` method

# [2.6.1](https://github.com/szabolcsdombi/zengl/compare/2.6.0...2.6.1)

- Changed `zengl.cleanup` to release objects
- Added the `zengl.cleanup` missing stubs

# [2.6.0](https://github.com/szabolcsdombi/zengl/compare/2.5.5...2.6.0)

- Implemented a builtin loader to avoid using ctypes
- Implemented `zengl.cleanup` to cleanup the objects stored at module level
- Implemented `Context.lost` flag
- Added the `Context.gc` missing stubs
- Moved the `zengl.default_loader` to `Context.loader`
- Fixed function pointer initialization flag type
- Fixed a one off memory leak in `zengl.init`
- Fixed the error message for the `Context.screen` setter

# [2.5.5](https://github.com/szabolcsdombi/zengl/compare/2.5.4...2.5.5)

- Fixed detection for python wasm sdk builds

# [2.5.4](https://github.com/szabolcsdombi/zengl/compare/2.5.3...2.5.4)

- Improved the web support
- Implemented `DefaultLoader.extra` to hold a reference to the objects created
- Fixed shader log decode errors when invalid unicode characters are present

# [2.5.3](https://github.com/szabolcsdombi/zengl/compare/2.5.2...2.5.3)

- Added validation for pipeline layout binding values

# [2.5.2](https://github.com/szabolcsdombi/zengl/compare/2.5.1...2.5.2)

- Fixed the documentation

# [2.5.1](https://github.com/szabolcsdombi/zengl/compare/2.5.0...2.5.1)

- Removed abi3 builds entirely

# [2.5.0](https://github.com/szabolcsdombi/zengl/compare/2.4.1...2.5.0)

- Implemented the `offset` parameter for `zengl.bind`
- Implemented the `instance` parameter for `zengl.bind`
- Implemented the `Image.renderbuffer` attribute
- Fixed the `Image.array` type hints
- Added the `zengl.default_loader` to hold a reference to the loader
- Added an error message when invalid images are attached as textures
- Added `rgba8unorm` as the default value for the `format` parameter in the `Context.image` method
- Removed the `Context.before_frame` callback
- Removed the `Context.after_frame` callback
- Removed abi3 builds for the web

# [2.4.1](https://github.com/szabolcsdombi/zengl/compare/2.4.0...2.4.1)

- Fixed detection for both egl and x11 context types

# [2.4.0](https://github.com/szabolcsdombi/zengl/compare/2.3.0...2.4.0)

- Improved the wayland support
- Added detection for both egl and x11 context types
- Fixed default argument value for reset in `Context.new_frame`
- Fixed the error message for multiple missing gl functions
- Changed the `Image.blit` parameters to `offset`, `size` and `crop`
- Removed the error messages from `Image.blit` for out of bounds regions
- Added support for flipping images with a negative size in `Image.blit`
- Added support to blit depth and stencil images
- Added a default value for the `target` parameter in the `zengl.camera` method
- Added support for angle brackets in shader includes
- Added support to hook the shader source right before compilation
- Added tuple length error checking for sizes and viewports

# [2.3.0](https://github.com/szabolcsdombi/zengl/compare/2.2.2...2.3.0)

- Removed the srgb image formats

# [2.2.2](https://github.com/szabolcsdombi/zengl/compare/2.2.1...2.2.2)

- Improved the typehints and documentation

# [2.2.1](https://github.com/szabolcsdombi/zengl/compare/2.2.0...2.2.1)

- Fixed the canvas detection in a web environment
- Fixed the `zengl.bind` typehints

# [2.2.0](https://github.com/szabolcsdombi/zengl/compare/2.1.0...2.2.0)

- Fixed `zengl.context` broken signature in the web build
- Fixed limits minimum and maximum values
- Added complete `Context.info` support in the web build
- Added pyscript and pygbag support alongside the pyodide support

# [2.1.0](https://github.com/szabolcsdombi/zengl/compare/1.18.0...2.1.0)

- Fixed custom glsl linker error callback
- Added missing `Image.format` typehints
- Added proper error message for a missing window in the default loader
- Removed `Buffer.map` and `Buffer.unmap`
- Implemented `Buffer.read` and `Buffer.view`
- Implemented `into` parameter for `Image.read` and `Buffer.read`
- Added support for Buffer sourced data
- Removed `dynamic` flag from `Context.buffer`
- Added `access` flag to `Context.buffer`
- Removed `Context.limits`
- Added limits to `Context.info`
- Changed framebuffer handling to only use the draw or read framebuffer binding
- Removed default framebuffer detection
- Fixed `zengl.max_combined_uniform_blocks` maximum value
- Fixed catching empty include statements

# [1.18.0](https://github.com/szabolcsdombi/zengl/compare/1.17.0...1.18.0)

- Added subinterpreter support
- Removed global reference for the default loader
- Changed `MAX_ATTACHMENTS` to 8
- Changed `MAX_BUFFER_BINDINGS` to 8
- Changed `MAX_SAMPLER_BINDINGS` to 16

# [1.17.0](https://github.com/szabolcsdombi/zengl/compare/1.16.0...1.17.0)

- Removed the loader parameter from `zengl.context`
- Implemented `zengl.init` with a loader parameter
- Changed `zengl.context` to return the first created context
- Fixed `Context.new_frame` type hints and documentation
- Fixed rendering to the default framebuffer without depth and stencil test
- Implemented a default loader for the web using the canvas by id
- Improved the documentation

# [1.16.0](https://github.com/szabolcsdombi/zengl/compare/1.15.0...1.16.0)

- Added `index`, and `uniform` parameters for `Context.buffer`
- Fixed clear in `Context.new_frame` for a nonzero default framebuffer
- Removed deprecated `load` method support from the loader

# [1.15.0](https://github.com/szabolcsdombi/zengl/compare/1.14.0...1.15.0)

- Added nativ web support without third party webgl binding
- Added `zengl.setup_gl` for the web build
- Fixed the default loader for osx

# [1.14.0](https://github.com/szabolcsdombi/zengl/compare/1.13.0...1.14.0)

- Improved web support
- Added Python 3.12 release builds
- Added support for customized builds

# [1.13.0](https://github.com/szabolcsdombi/zengl/compare/1.12.2...1.13.0)

- Implemented external pipeline render parameters
- Implemented external viewport data
- Implemented external uniform data
- Implemented uniform binding jump table
- Changed `Pipeline.uniforms` to be always present

# [1.12.2](https://github.com/szabolcsdombi/zengl/compare/1.12.1...1.12.2)

- Added web support

# [1.12.1](https://github.com/szabolcsdombi/zengl/compare/1.12.0...1.12.1)

- Fixed no detectable viewport size when rendering to the default framebuffer
- Fixed init time mipmap level allocation width and height

# [1.12.0](https://github.com/szabolcsdombi/zengl/compare/1.11.0...1.12.0)

- Changed `Image.blit()` to support `ImageFace` objects
- Changed `ImageFace.blit()` to support `Image` objects
- Implemented `Image.read()` and `ImageFace.read()` for cubemap and array images
- Implemented `Image.read()` and `ImageFace.read()` for multisampled images
- Implemented init time mipmap level allocation
- Removed the arguments of the `Image.mipmaps` method
- Added support for unbound vertex attributes

# [1.11.0](https://github.com/szabolcsdombi/zengl/compare/1.10.2...1.11.0)

- Implemented the `Context.new_frame()` and `Context.end_frame()` methods
- Removed the flush parameter from `Image.blit` method
- Changed the `Context.pipeline()` blending parameter to blend
- Removed the skip_validation, primitive_restart, color_mask parameters from the `Context.pipeline()` method
- Primitive restart default index is enabled by default
- Changed the `Context.info` tuple to a dictionary
- Removed `glcontext` dependency for windowed rendering
- Implemented `zengl.__version__` string

# [1.10.2](https://github.com/szabolcsdombi/zengl/compare/1.10.1...1.10.2)

- Fixed zengl camera utility with broken translation in ortho mode

# [1.10.1](https://github.com/szabolcsdombi/zengl/compare/1.10.0...1.10.1)

- Implemented default texture filter for improved external texture support

# [1.10.0](https://github.com/szabolcsdombi/zengl/compare/1.9.3...1.10.0)

- Implemented external buffers and textures
- Changed `zengl.context()` to prefer the `load_opengl_function()` over the `load()` method
- Deprecated `ContextLoader.load()` in favor of `ContextLoader.load_opengl_function()`
- Fixed vertex array caching
- Fixed read buffer for depth only framebuffers
- Fixed empty vertex and fragment shader cache collision
- Fixed use after free in `Context.release()`

# [1.9.3](https://github.com/szabolcsdombi/zengl/compare/1.9.2...1.9.3)

- Moved zengl.pyi to zengl-stubs
- Fixed missing type hinting

# [1.9.2](https://github.com/szabolcsdombi/zengl/compare/1.9.1...1.9.2)

- Implemented blending equations

# [1.9.1](https://github.com/szabolcsdombi/zengl/compare/1.9.0...1.9.1)

- Fixed uniform struct array naming

# [1.9.0](https://github.com/szabolcsdombi/zengl/compare/1.8.4...1.9.0)

- Implemented Pipeline uniforms
- Implemented rendering to cubemap faces and array layers
- Removed `zengl.pack()`
- Removed `zengl.rgba()`

# [1.8.4](https://github.com/szabolcsdombi/zengl/compare/1.8.3...1.8.4)

- Fixed unbound vertex shader builtin variables error reporting
- Fixed silent error for invalid topology
- Fixed redundant framebuffer bind when blitting images
- Replaced string formatting to f-strings in the `_zengl` helper module

# [1.8.3](https://github.com/szabolcsdombi/zengl/compare/1.8.2...1.8.3)

- Fixed uniform buffer array members received a valid location parameter
- Deprecated `zengl.pack()`
- Deprecated `zengl.rgba()`

# [1.8.2](https://github.com/szabolcsdombi/zengl/compare/1.8.1...1.8.2)

- Fixed attribute array and uniform array location
- Fixed broken linker error formatting
- Added glcontext as a dependency

# [1.8.1](https://github.com/szabolcsdombi/zengl/compare/1.8.0...1.8.1)

- Fixed setting depth function

# [1.8.0](https://github.com/szabolcsdombi/zengl/compare/1.7.0...1.8.0)

- Fixed pipeline required arguments to raise an excenption
- Fixed buffer write size check with non zero offset
- Implemented error for renderbuffer images with initial data
- Implemented error for negative image array count
- Implemented range check for writing to already generated mipmap leves
- Implemented `Context.release('all')`
- Fixed cubemap images failed to create without initial data
- Fixed fault when clearing or blitting layered images
- Fixed reading depth and stencil images
- Fixed refcount for types when deinitializing the module
- Fixed `Image.blit` srgb parameter defaulted to the flush parameter
- Removed deprecated `clear_shader_cache`

# [1.7.0](https://github.com/szabolcsdombi/zengl/compare/1.6.1...1.7.0)

- Implemented `Context.reset` to assume the context is dirty
- Fixed integer textures used the wrong base format
- Removed invalid texture format type hints
- Fixed broken bound object cache after releasing objects
- Improved invalid size error message for images
- Implemented write to image layer
- Fixed image write for all levels
- Fixed image create and image write stride for small cubemaps
- Implemented size check for image create and image write
- Implemented `Context.release('shader_cache')`
- Deprecated `clear_shader_cache`

# [1.6.1](https://github.com/szabolcsdombi/zengl/compare/1.6.0...1.6.1)

- Fixed reading single sample renderbuffers

# [1.6.0](https://github.com/szabolcsdombi/zengl/compare/1.5.0...1.6.0)

- Support texture max anisotropy
- Removed deprecated `line_width` and `front_face`

# [1.5.0](https://github.com/szabolcsdombi/zengl/compare/1.4.3...1.5.0)

- Implemented flush parameter for blitting
- Improved default for srgb parameter for blitting
- Fixed error checking for image blitting
- Fixed dependency list for examples
- Support binding uniform and attribute arrays
- Optimized binding global settings
- Deprecated `line_width` and `front_face`

# [1.4.3](https://github.com/szabolcsdombi/zengl/compare/1.4.2...1.4.3)

- Support pipeline creation without validation
- Improved package info

# [1.4.2](https://github.com/szabolcsdombi/zengl/compare/1.4.1...1.4.2)

- Fixed wrong stride used while initializing cubemap images
- Implemented size check for creating images

# [1.4.1](https://github.com/szabolcsdombi/zengl/compare/1.4.0...1.4.1)

- Optimized binding the same global settings
- Optimized binding the same viewport
- Optimized cache for write masks for clear operations
- Fixed missing default depth function for explicit depth settings
- Fixed stencil test was on when no stencil buffer was present

# [1.4.0](https://github.com/szabolcsdombi/zengl/compare/1.3.0...1.4.0)

- Improve interoperability for porting existing code
- Implemented `Context.screen` as the default framebuffer
- Implemented `zengl.inspect` for debugging

# [1.3.0](https://github.com/szabolcsdombi/zengl/compare/1.2.1...1.3.0)

- Support list of ints where tuple of ints are expected

# [1.2.2](https://github.com/szabolcsdombi/zengl/compare/1.2.1...1.2.2)

- Fixed broken `Pipeline.vertex_count`, `Pipeline.instance_count` and `Pipeline.first_vertex`

# [1.2.1](https://github.com/szabolcsdombi/zengl/compare/1.2.0...1.2.1)

- Support x86 architecture
- Support raspberry pi

# [1.2.0](https://github.com/szabolcsdombi/zengl/compare/1.1.0...1.2.0)

- Validate image format
- Removed `max_varying_components` limit
- Fixed blank screen for osx when double buffering is enabled
- Disabled shadow window configuration for osx
- Simplified examples main loop

# [1.1.0](https://github.com/szabolcsdombi/zengl/compare/1.0.1...1.1.0)

- Implemented soft limit for max number of samples
- Implemented `Context.limits`
- Implemented uniform buffer size validation
- Blit to screen flushes the command buffer

# [1.0.1](https://github.com/szabolcsdombi/zengl/compare/1.0.0...1.0.1)

- Fixed compile issues on multiple platforms
- Build wheels with cibuildwheel

# [1.0.0](https://github.com/szabolcsdombi/zengl/tree/1.0.0)

First stable version
