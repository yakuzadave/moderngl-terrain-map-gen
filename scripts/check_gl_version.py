import moderngl
import sys

try:
    ctx = moderngl.create_standalone_context()
    print(f"OpenGL Version: {ctx.version_code}")
    print(f"Vendor: {ctx.info['GL_VENDOR']}")
    print(f"Renderer: {ctx.info['GL_RENDERER']}")
    print(f"Version: {ctx.info['GL_VERSION']}")
except Exception as e:
    print(f"Error creating context: {e}")
