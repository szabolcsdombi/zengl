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
12 glDisable(GL_POLYGON_OFFSET_LINE)
13 glDisable(GL_POLYGON_OFFSET_POINT)
14 glDisable(GL_STENCIL_TEST)
15 glEnable(GL_DEPTH_TEST)
16 glEnable(GL_CULL_FACE)
17 glCullFace(GL_BACK)
18 glLineWidth(1.00)
19 glFrontFace(GL_CCW)
20 glDepthMask(True)
21 glStencilMaskSeparate(GL_FRONT, 255)
22 glStencilMaskSeparate(GL_BACK, 255)
23 glStencilFuncSeparate(GL_FRONT, GL_ALWAYS, 0, 255)
24 glStencilFuncSeparate(GL_BACK, GL_ALWAYS, 0, 255)
25 glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_KEEP)
26 glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_KEEP)
27 glBlendFuncSeparate(GL_LINES, GL_NONE, GL_LINES, GL_NONE)
28 glPolygonOffset(0.00, 0.00)
29 glDisablei(GL_BLEND, 0)
30 glColorMaski(0, True, True, True, True)
31 glBindFramebuffer(GL_FRAMEBUFFER, Framebuffer 110)
32 glUseProgram(Program 109)
33 glBindVertexArray(Vertex Array 111)
34 glBindBufferRange(GL_UNIFORM_BUFFER, 0, Buffer 86)
35 glBindBufferRange(GL_UNIFORM_BUFFER, 1, Buffer 103)
36 glDrawArraysInstanced(3276, 1)
37 glBindVertexArray(Vertex Array 112)
38 glBindBufferRange(GL_UNIFORM_BUFFER, 0, Buffer 86)
39 glBindBufferRange(GL_UNIFORM_BUFFER, 1, Buffer 103)
40 glDrawArraysInstanced(1440, 1)
41 glBindVertexArray(Vertex Array 113)
42 glBindBufferRange(GL_UNIFORM_BUFFER, 0, Buffer 86)
43 glBindBufferRange(GL_UNIFORM_BUFFER, 1, Buffer 103)
44 glDrawArraysInstanced(1440, 1)
45 glBindVertexArray(Vertex Array 114)
46 glBindBufferRange(GL_UNIFORM_BUFFER, 0, Buffer 86)
47 glBindBufferRange(GL_UNIFORM_BUFFER, 1, Buffer 103)
48 glDrawArraysInstanced(1440, 1)
49 glBindVertexArray(Vertex Array 115)
50 glBindBufferRange(GL_UNIFORM_BUFFER, 0, Buffer 86)
51 glBindBufferRange(GL_UNIFORM_BUFFER, 1, Buffer 103)
52 glDrawArraysInstanced(36, 1)
53 glEnable(GL_PRIMITIVE_RESTART)
54 glDisable(GL_POLYGON_OFFSET_FILL)
55 glDisable(GL_POLYGON_OFFSET_LINE)
56 glDisable(GL_POLYGON_OFFSET_POINT)
57 glDisable(GL_STENCIL_TEST)
58 glEnable(GL_DEPTH_TEST)
59 glDisable(GL_CULL_FACE)
60 glLineWidth(1.00)
61 glFrontFace(GL_CCW)
62 glDepthMask(True)
63 glStencilMaskSeparate(GL_FRONT, 255)
64 glStencilMaskSeparate(GL_BACK, 255)
65 glStencilFuncSeparate(GL_FRONT, GL_ALWAYS, 0, 255)
66 glStencilFuncSeparate(GL_BACK, GL_ALWAYS, 0, 255)
67 glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_KEEP)
68 glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_KEEP)
69 glBlendFuncSeparate(GL_LINES, GL_NONE, GL_LINES, GL_NONE)
70 glPolygonOffset(0.00, 0.00)
71 glDisablei(GL_BLEND, 0)
72 glColorMaski(0, True, True, True, True)
73 glBindFramebuffer(GL_FRAMEBUFFER, Framebuffer 92)
74 glUseProgram(Program 97)
75 glBindVertexArray(Vertex Array 98)
76 glActiveTexture(GL_TEXTURE0)
77 glBindTexture(GL_TEXTURE_2D, Texture 87)
78 glBindSampler(0, Sampler 99)
79 glDrawArraysInstanced(3, 1)
80 glBindFramebuffer(GL_FRAMEBUFFER, Framebuffer 94)
81 glUseProgram(Program 101)
82 glBindVertexArray(Vertex Array 102)
83 glActiveTexture(GL_TEXTURE0)
84 glBindTexture(GL_TEXTURE_2D, Texture 91)
85 glBindSampler(0, Sampler 99)
86 glDrawArraysInstanced(3, 1)
87 glDisable(GL_FRAMEBUFFER_SRGB)
88 glBindFramebuffer(GL_READ_FRAMEBUFFER, Framebuffer 94)
89 glBindFramebuffer(GL_DRAW_FRAMEBUFFER, Backbuffer FBO)
90 glBlitFramebuffer(Framebuffer 94, Backbuffer FBO)
91 glBindFramebuffer(GL_FRAMEBUFFER, Backbuffer FBO)
92 glEnable(GL_FRAMEBUFFER_SRGB)
93 glFlush()
```
