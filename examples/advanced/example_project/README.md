# Example Project

This project intends to demonstrate:

- organized code into multiple files
- updating the buffers only once at the end of the updates
- rendering to a framebuffer
- post-processing

Architected code in this folder may not be a good fit for a different project.
The `Blur` class is somewhat reusable as it takes an image and outputs to another.
It depends on a globally importable context that may be a problem when moving this code elsewhere.
The `ObjectManager` allows creating different objects. Updating the positions is postponed to a single write call.
The `Context` contains the uniform buffer shared across the programs.
To ensure the layout is the same in the programs using the main uniform buffer,
the glsl definition is defined once and can be included in multiple places.
Despite multiple pipelines are created from the same source code,
the cache returns the built program and consecutive render calls do not re-bind the program object.
This can be seen below. The `glUseProgram(Program 109)` is called once before five renders.

The 10th frame of the rendering produces the following api calls:

```
1  Context Configuration()
2  glBindBuffer(GL_ARRAY_BUFFER, Buffer 86)
3  glBufferSubData(Buffer 86, (64 bytes))
4  glBindFramebuffer(GL_FRAMEBUFFER, Framebuffer 88)
5  glClearBufferfv(0, { 0.20, 0.20, 0.20, 1.00 })
6  glBindFramebuffer(GL_FRAMEBUFFER, Framebuffer 90)
7  glClearBufferfv(0, { 1.00 })
8  glBindBuffer(GL_ARRAY_BUFFER, Buffer 103)
9  glBufferSubData(Buffer 103, (1024 bytes))
10 glEnable(GL_PRIMITIVE_RESTART)
11 glDisable(GL_POLYGON_OFFSET_FILL)
12 glEnable(GL_CULL_FACE)
13 glCullFace(GL_BACK)
14 glEnable(GL_DEPTH_TEST)
15 glBindFramebuffer(GL_FRAMEBUFFER, Framebuffer 110)
16 glUseProgram(Program 109)
17 glBindVertexArray(Vertex Array 111)
18 glBindBufferRange(GL_UNIFORM_BUFFER, 0, Buffer 86)
19 glBindBufferRange(GL_UNIFORM_BUFFER, 1, Buffer 103)
20 glDrawArraysInstanced(3276, 1)
21 glBindVertexArray(Vertex Array 112)
22 glBindBufferRange(GL_UNIFORM_BUFFER, 0, Buffer 86)
23 glBindBufferRange(GL_UNIFORM_BUFFER, 1, Buffer 103)
24 glDrawArraysInstanced(1440, 1)
25 glBindVertexArray(Vertex Array 113)
26 glBindBufferRange(GL_UNIFORM_BUFFER, 0, Buffer 86)
27 glBindBufferRange(GL_UNIFORM_BUFFER, 1, Buffer 103)
28 glDrawArraysInstanced(1440, 1)
29 glBindVertexArray(Vertex Array 114)
30 glBindBufferRange(GL_UNIFORM_BUFFER, 0, Buffer 86)
31 glBindBufferRange(GL_UNIFORM_BUFFER, 1, Buffer 103)
32 glDrawArraysInstanced(1440, 1)
33 glBindVertexArray(Vertex Array 115)
34 glBindBufferRange(GL_UNIFORM_BUFFER, 0, Buffer 86)
35 glBindBufferRange(GL_UNIFORM_BUFFER, 1, Buffer 103)
36 glDrawArraysInstanced(36, 1)
37 glEnable(GL_PRIMITIVE_RESTART)
38 glDisable(GL_POLYGON_OFFSET_FILL)
39 glDisable(GL_CULL_FACE)
40 glEnable(GL_DEPTH_TEST)
41 glBindFramebuffer(GL_FRAMEBUFFER, Framebuffer 92)
42 glUseProgram(Program 97)
43 glBindVertexArray(Vertex Array 98)
44 glActiveTexture(GL_TEXTURE0)
45 glBindTexture(GL_TEXTURE_2D, Texture 87)
46 glBindSampler(0, Sampler 99)
47 glDrawArraysInstanced(3, 1)
48 glBindFramebuffer(GL_FRAMEBUFFER, Framebuffer 94)
49 glUseProgram(Program 101)
50 glBindVertexArray(Vertex Array 102)
51 glActiveTexture(GL_TEXTURE0)
52 glBindTexture(GL_TEXTURE_2D, Texture 91)
53 glBindSampler(0, Sampler 99)
54 glDrawArraysInstanced(3, 1)
55 glDisable(GL_FRAMEBUFFER_SRGB)
56 glBindFramebuffer(GL_READ_FRAMEBUFFER, Framebuffer 94)
57 glBindFramebuffer(GL_DRAW_FRAMEBUFFER, Backbuffer FBO)
58 glBlitFramebuffer(Framebuffer 94, Backbuffer FBO)
59 glBindFramebuffer(GL_FRAMEBUFFER, Backbuffer FBO)
60 glEnable(GL_FRAMEBUFFER_SRGB)
61 glFlush()
```
