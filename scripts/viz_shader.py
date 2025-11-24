import sys
import argparse
import time
from pathlib import Path
import moderngl_window as mglw

# Global to store parsed args
SHADER_ARGS = None


class ShaderPreview(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "GLSL Preview"
    resource_dir = (Path(__file__).parent / 'src' / 'shaders').resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quad = mglw.geometry.quad_fs()
        self.program = None
        self.start_time = time.time()
        self.mouse_pos = (0, 0)

        # Use global args
        self.shader_path = Path(SHADER_ARGS.shader)
        self.texture_path = Path(
            SHADER_ARGS.texture) if SHADER_ARGS.texture else None
        self.texture = None

        if self.texture_path:
            self.load_texture()

        self.load_shader()

    def load_texture(self):
        try:
            self.texture = self.load_texture_2d(str(self.texture_path))
            # Repeat mode for seamless testing
            self.texture.repeat_x = True
            self.texture.repeat_y = True
            print(f"Loaded texture {self.texture_path}")
        except Exception as e:
            print(f"Error loading texture: {e}")

    def load_shader(self):
        try:
            with open(self.shader_path, 'r') as f:
                frag_source = f.read()

            # Basic vertex shader
            vert_source = """
            #version 330
            in vec3 in_position;
            in vec2 in_texcoord_0;
            out vec2 uv;
            void main() {
                gl_Position = vec4(in_position, 1.0);
                uv = in_texcoord_0;
            }
            """

            self.program = self.ctx.program(
                vertex_shader=vert_source,
                fragment_shader=frag_source
            )
            print(f"Loaded {self.shader_path}")
        except Exception as e:
            print(f"Error loading shader: {e}")

    def on_render(self, time_val, frame_time):
        self.ctx.clear()

        if self.program:
            # Bind texture if exists
            if self.texture:
                self.texture.use(location=0)
                if 'u_texture_0' in self.program:
                    self.program['u_texture_0'].value = 0

            # Set standard uniforms if they exist
            if 'u_time' in self.program:
                self.program['u_time'].value = time_val
            if 'u_resolution' in self.program:
                self.program['u_resolution'].value = self.wnd.buffer_size
            if 'u_mouse' in self.program:
                self.program['u_mouse'].value = self.mouse_pos

            self.quad.render(self.program)

    def mouse_position_event(self, x, y, dx, dy):
        self.mouse_pos = (x, self.wnd.height - y)

    def key_event(self, key, action, modifiers):
        if key == self.wnd.keys.R and action == self.wnd.keys.ACTION_PRESS:
            self.load_shader()


if __name__ == '__main__':
    # Parse our args first
    parser = argparse.ArgumentParser()
    parser.add_argument("shader", help="Path to fragment shader")
    parser.add_argument(
        "--texture", help="Path to input texture", default=None)

    # Parse known args and keep the rest for mglw
    SHADER_ARGS, remaining_args = parser.parse_known_args()

    # Update sys.argv so mglw only sees its own args
    sys.argv = [sys.argv[0]] + remaining_args

    mglw.run_window_config(ShaderPreview)
