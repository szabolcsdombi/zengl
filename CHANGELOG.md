# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

# [main](https://github.com/szabolcsdombi/zengl/compare/1.10.1...main)

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
