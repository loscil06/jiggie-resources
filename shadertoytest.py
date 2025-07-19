import numpy as np

from wgpu_shadertoy import Shadertoy

from PIL import Image

shader_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));
    fragColor = vec4(col,1.0);
}
"""


shader = Shadertoy(shader_code, resolution=(800, 450), offscreen=True)

if __name__ == "__main__":
    frame0_data = shader.snapshot()
    frame600_data = shader.snapshot(time_float=10.0, frame=600)
    frame0_img = Image.fromarray(np.asarray(frame0_data))
    frame0_img.save("frame0.png")
    shader.show()
