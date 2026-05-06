"""
GLOverlay — GPU-accelerated camera overlay with elegant warm effects.

Renders the live camera feed with a subtle warm vignette, soft film grain,
and a gentle scanline texture. No cyberpunk effects — purely cinematic
and refined, like looking through a premium optical instrument.
"""

import time
import numpy as np
import ctypes
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Property, QPropertyAnimation, QEasingCurve, Slot
from PySide6.QtGui import QImage
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# ─── Shaders ──────────────────────────────────────────────────────────

VERTEX_SHADER = """
#version 330
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;
out vec2 v_texCoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_texCoord = texCoord;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_image;
uniform float u_time;
uniform float u_scanline_opacity;

void main() {
    vec2 uv = v_texCoord;

    // Sample the camera image
    fragColor = texture(u_image, uv);

    // Warm color grade — subtle amber/sepia push
    fragColor.r *= 1.04;
    fragColor.g *= 1.00;
    fragColor.b *= 0.94;

    // Elegant vignette — darker, warm edges
    float dist = distance(uv, vec2(0.5, 0.5));
    float vignette = smoothstep(0.85, 0.35, dist);
    fragColor.rgb *= mix(0.45, 1.0, vignette);

    // Very subtle scanlines (like a premium display)
    float scanline = sin(uv.y * 800.0) * 0.5 + 0.5;
    fragColor.rgb -= scanline * u_scanline_opacity * 0.06;

    // Film grain — extremely subtle organic texture
    float noise = fract(sin(dot(uv, vec2(12.9898, 78.233) * (u_time * 0.1 + 1.0))) * 43758.5453);
    fragColor.rgb += (noise - 0.5) * 0.015;

    // Slight contrast boost
    fragColor.rgb = mix(vec3(0.5), fragColor.rgb, 1.08);
}
"""


class GLOverlay(QOpenGLWidget):
    """
    GPU-accelerated overlay for elegant camera feed rendering.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None
        self._texture_id = None
        self._program = None
        self._vao = None
        self._vbo = None
        self._start_time = time.time()

        # Shader parameters
        self._scanline_opacity = 0.15  # Very subtle by default

    @Slot(float)
    def set_scanline_opacity(self, value):
        self._scanline_opacity = value
        self.update()

    def set_image(self, qimage):
        self._image = qimage
        self.update()

    # ─── OpenGL Lifecycle ─────────────────────────────────────────────

    def initializeGL(self):
        glClearColor(0.02, 0.02, 0.015, 1)

        try:
            self._program = compileProgram(
                compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(f"[GLOverlay] Shader compilation failed: {e}")
            return

        vertices = np.array([
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 0.0,
        ], dtype=np.float32)

        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)

        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

        self._texture_id = glGenTextures(1)

    def paintGL(self):
        if not self._program or self._image is None or self._image.isNull():
            return

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self._program)

        img = self._image.convertToFormat(QImage.Format.Format_RGBA8888)
        width, height = img.width(), img.height()
        ptr = img.bits()

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, ptr)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glUniform1i(glGetUniformLocation(self._program, "u_image"), 0)
        glUniform1f(glGetUniformLocation(self._program, "u_time"), time.time() - self._start_time)
        glUniform1f(glGetUniformLocation(self._program, "u_scanline_opacity"), self._scanline_opacity)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
