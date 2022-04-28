# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

# [main](https://github.com/szabolcsdombi/zengl/compare/1.5.0...main)

# [1.5.0](https://github.com/szabolcsdombi/zengl/compare/1.4.3...1.5.0)

- Implemented flush parameter for blitting
- Improved default for srgb parameter for blitting
- Fixed error checking for image blitting
- Fixed dependency list for examples
- Support binding uniform and attribute arrays
- Optimized binding global settings
- Deprecated line_width and front_face

# [1.4.3](https://github.com/szabolcsdombi/zengl/compare/1.4.2...1.4.3)

- Support pipeline creation without validation
- Improve package info

# [1.4.2](https://github.com/szabolcsdombi/zengl/compare/1.4.1...1.4.2)

- Fixed wrong stride used while initializing cubemap images
- Implement size check for creating images

# [1.4.1](https://github.com/szabolcsdombi/zengl/compare/1.4.0...1.4.1)

- Optimized binding the same global settings
- Optimized binding the same viewport
- Optimized cache for write masks for clear operations
- Fixed missing default depth function for explicit depth settings
- Fixed stencil test was on when no stencil buffer was present

# [1.4.0](https://github.com/szabolcsdombi/zengl/compare/1.3.0...1.4.0)

- Improve interoperability for porting existing code
- Implement `Context.screen`
- Implement `zengl.inspect`

# [1.3.0](https://github.com/szabolcsdombi/zengl/compare/1.2.1...1.3.0)

- Support list of ints where tuple of ints are expected

# [1.2.2](https://github.com/szabolcsdombi/zengl/compare/1.2.1...1.2.2)

- Fix broken `Pipeline.vertex_count`, `Pipeline.instance_count` and `Pipeline.first_vertex`

# [1.2.1](https://github.com/szabolcsdombi/zengl/compare/1.2.0...1.2.1)

- Support x86 architecture
- Support raspberry pi

# [1.2.0](https://github.com/szabolcsdombi/zengl/compare/1.1.0...1.2.0)

- Invalid image format error
- Removed `max_varying_components` limit
- Fixed blank screen for osx when double buffering is enabled
- Disabled shadow window configuration for osx
- Simplified examples main loop

# [1.1.0](https://github.com/szabolcsdombi/zengl/compare/1.0.1...1.1.0)

- Implement soft limit for max number of samples
- Implement `Context.limits`
- Validate uniform buffer size
- Blit to screen flushes the command buffer

# [1.0.1](https://github.com/szabolcsdombi/zengl/compare/1.0.0...1.0.1)

- Fix compile issues on multiple platforms
- Build wheels with cibuildwheel

# [1.0.0](https://github.com/szabolcsdombi/zengl/tree/1.0.0)

First stable version
