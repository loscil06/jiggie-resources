import os
import sys
import json
import math
import random
import colorsys
from pathlib import Path

import click
import numpy as np
from jinja2 import Template
from PIL import Image, ImageDraw, ImageColor, ImageFilter
from wgpu_shadertoy import Shadertoy

# Supported styles
STYLES = ['gradient', 'liquid', 'voronoi', 'topographic', 'spiral', 'squiggle', 'mesh', 'scales', 'watercolor']

# Preset gradients separated by color count
GRADIENTS_2 = {
    "fireandice": ["#ff1b6b", "#45caff"],
    "cyberpunk": ["#40c9ff", "#e81cff"],
    "bubblegum": ["#df89b5", "#bfd9fe"],
    "pastelsky": ["#EEAECA", "#94BBE9"],
    "abyssalblue": ["#1CB5E0", "#000851"],
    "skymint": ["#00C9FF", "#94E29C"],
    "goldenlagoon": ["#FDBB2D", "#22C1C3"],
    "sunsetroyal": ["#FDBB2D", "#3A1C71"],
    "sunset": ["#fcef04", "#dc419b"],
    "paloalto": ["#16a085", "#f4d03f"],
    "spicy_sunset": ["#BD4B35", "#D2C400"],
    "ocean_and_sun": ["#1c90bf", "#eb8a3d"],
    "christmas": ["#AA3A38", "#2F7336"],
    "pizelex": ["#F29492", "#114357"],
    "virgin": ["#FFAFBD", "#C9FFBF"],
    "earthly": ["#DBD5A4", "#649173"],
    "misty_lagoon": ["#e4e4d9", "#215f00"],
}
GRADIENTS_3 = {
    "mermaid_dream": ["#84ffc9", "#aab2ff", "#d084ff"],
    "heatwave": ["#833AB4", "#FD1D1D", "#FCB045"],
    "mintberry": ["#5AFF15", "#AAFFE5", "#9D75CB"],
    "oceanbliss": ["#7B4B94", "#7D82B8", "#B7E3CC"],
    "sunsetsorbet": ["#79ADDC", "#FFC09F", "#FFEE93"],
    "candydust": ["#2E294E", "#EFBCD5", "#BE97C6"],
    "playland": ["#caefd7", "#f5bfd7", "#abc9e9"],
    "weddingdayblues": ["#FF0080", "#FF8C00", "#40E0D0"],
    "lunada": ["#A5FECB", "#20BDFF", "#5433FF"],
}

# Gradient orientation mapping
ORIENTATIONS = {
    "vertical": 90,
    "horizontal": 0,
    "diagonal": 45,
    "diagonal-reverse": 135
}


def hex_to_rgb(hex_color):
    return ImageColor.getrgb(hex_color)


def rgb_to_float(rgb):
    return tuple(c / 255.0 for c in rgb)


def rgb_to_hsl(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    return colorsys.rgb_to_hls(r, g, b)


def hsl_to_rgb(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def generate_vibrant_midcolor(rgb1, rgb2):
    h1, l1, s1 = rgb_to_hsl(rgb1)
    h2, l2, s2 = rgb_to_hsl(rgb2)

    # Hue interpolation (shortest distance around the color wheel)
    dh = h2 - h1
    if abs(dh) > 0.5:
        if dh > 0:
            h1 += 1
        else:
            h2 += 1
    h_mid = (h1 + h2) / 2.0 % 1.0

    l_mid = (l1 + l2) / 2.0 + 0.05  # slight brightness boost
    s_mid = (s1 + s2) / 2.0
    return hsl_to_rgb(h_mid, min(l_mid, 1.0), s_mid)


def derive_shader_colors(base_colors):
    base_rgb = [hex_to_rgb(c) for c in base_colors]

    if len(base_rgb) == 2:
        color1, color2 = base_rgb
        color3 = generate_vibrant_midcolor(color1, color2)
        return [color1, color2, color3]
    else:
        return base_rgb[:3]


def derive_topographic_colors(base_colors):
    """
    Derive 4-color palette for topographic maps from base gradient colors
    Uses different color derivation strategy than other shaders
    """
    base_rgb = [hex_to_rgb(c) for c in base_colors]

    if len(base_rgb) == 2:
        # Create 4 colors from 2-color gradient
        color1, color2 = base_rgb
        return [
            color1,
            mix_colors(color1, color2, 0.33),
            mix_colors(color1, color2, 0.67),
            color2
        ]
    elif len(base_rgb) == 3:
        # Create 4 colors from 3-color gradient
        color1, color2, color3 = base_rgb
        return [
            color1,
            color2,
            color3,
            adjust_lightness(color3, -15)  # Darken last color slightly
        ]
    else:
        # Use first 4 colors
        return base_rgb[:4]


def adjust_lightness(rgb, percent):
    """Adjust lightness of color in HSL space"""
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_l = max(0.0, min(1.0, l + percent / 100.0))
    r, g, b = colorsys.hls_to_rgb(h, new_l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def mix_colors(rgb1, rgb2, factor):
    """Mix two RGB colors with given factor (0.0-1.0)"""
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    r = int(r1 * (1 - factor) + r2 * factor)
    g = int(g1 * (1 - factor) + g2 * factor)
    b = int(b1 * (1 - factor) + b2 * factor)
    return r, g, b


def resource_path(*relative_parts):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent
    return base_path.joinpath(*relative_parts)


def rgb_to_vec3_glsl(rgb):
    """Format an RGB tuple as GLSL vec3 with three decimals."""
    f = rgb_to_float(rgb)
    return f"vec3({f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f})"


def render_liquid_gradient_shader_background(width, height, colors):
    if not isinstance(colors, (list, tuple)) or len(colors) != 2:
        raise ValueError("render_liquid_gradient_shader_background requires exactly two colors.")

    common_path = resource_path("shaders", "liquid_gradient", "common_liquid_gradient.glsl")
    with common_path.open("r") as f:
        common_code = f.read()
    template_path = resource_path("shaders", "liquid_gradient", "main_liquid_gradient.glsl.j2")
    with template_path.open("r") as f:
        template = Template(f.read())

    color1 = rgb_to_vec3_glsl(hex_to_rgb(colors[0]))
    color2 = rgb_to_vec3_glsl(hex_to_rgb(colors[1]))

    code = template.render(color1=color1, color2=color2, seed=random.uniform(1, 10000))
    shader = Shadertoy(code, common=common_code, resolution=(width, height), offscreen=True)
    return Image.fromarray(np.asarray(shader.snapshot(time_float=0.0)))


def render_voronoi_gradient_shader_background(width, height, colors):
    if not isinstance(colors, (list, tuple)) or len(colors) != 2:
        raise ValueError("render_liquid_gradient_shader_background requires exactly two colors.")

    template_path = resource_path("shaders", "voronoi_gradient", "main_voronoi_gradient.glsl.j2")
    with template_path.open("r") as f:
        template = Template(f.read())

    color1 = rgb_to_vec3_glsl(hex_to_rgb(colors[0]))
    color2 = rgb_to_vec3_glsl(hex_to_rgb(colors[1]))

    code = template.render(color1=color1, color2=color2, seed=random.uniform(1, 10))
    shader = Shadertoy(code, resolution=(width, height), offscreen=True)
    return Image.fromarray(np.asarray(shader.snapshot(time_float=0.0)))


def render_topographic_shader_background(width, height, colors, scale=1.0):
    """Render topographic shader background using wgpu-shadertoy"""
    shader_colors = derive_topographic_colors(colors)

    template_path = resource_path("shaders", "topographic", "main_topographic_shader.glsl.j2")
    with template_path.open("r") as f:
        template = Template(f.read())

    color_1 = rgb_to_vec3_glsl(shader_colors[0])
    color_2 = rgb_to_vec3_glsl(shader_colors[1])
    color_3 = rgb_to_vec3_glsl(shader_colors[2])
    color_4 = rgb_to_vec3_glsl(shader_colors[3])

    code = template.render(color_1=color_1, color_2=color_2, color_3=color_3, color_4=color_4, seed=random.uniform(1, 1200))
    shader = Shadertoy(code, resolution=(width, height), offscreen=True)
    return Image.fromarray(np.asarray(shader.snapshot(time_float=0.0)))


def render_spiral_shader_background(width, height, colors):
    if not isinstance(colors, (list, tuple)) or len(colors) != 2:
        raise ValueError("render_spiral_shader_background requires exactly two colors.")

    template_path = resource_path("shaders", "spiral", "main_spiral_shader.glsl.j2")
    with template_path.open("r") as f:
        template = Template(f.read())

    color_1 = rgb_to_vec3_glsl(hex_to_rgb(colors[0]))
    color_2 = rgb_to_vec3_glsl(hex_to_rgb(colors[1]))

    code = template.render(color_1=color_1, color_2=color_2)
    shader = Shadertoy(code, resolution=(width, height), offscreen=True)
    return Image.fromarray(np.asarray(shader.snapshot(time_float=1)))


def render_squiggle_shader_background(width, height, colors):
    shader_colors = derive_topographic_colors(colors)
    random.shuffle(shader_colors)

    template_path = resource_path("shaders", "squiggle", "main_squiggle_shader.glsl.j2")
    with template_path.open("r") as f:
        template = Template(f.read())

    randomized_colors = {
        f"color{n+1}": rgb_to_vec3_glsl(c) for n, c in enumerate(shader_colors)
    }

    code = template.render(
        **randomized_colors,
        random_color=random.choice(list(randomized_colors.values())),
        seed=random.uniform(1, 10)
    )
    shader = Shadertoy(code, resolution=(width, height), offscreen=True)
    return Image.fromarray(np.asarray(shader.snapshot(time_float=0.0)))


def render_mesh_gradient_shader_background(width, height, colors):
    # Use topographic color derivation for mesh gradient
    shader_colors = derive_topographic_colors(colors)

    template_path = resource_path("shaders", "mesh_gradient", "main_mesh_gradient_shader.glsl.j2")
    with template_path.open("r") as f:
        template = Template(f.read())

    color1 = rgb_to_vec3_glsl(shader_colors[0])
    color2 = rgb_to_vec3_glsl(shader_colors[1])
    color3 = rgb_to_vec3_glsl(shader_colors[2])
    color4 = rgb_to_vec3_glsl(shader_colors[3])

    code = template.render(
        color1=color1,
        color2=color2,
        color3=color3,
        color4=color4,
        seed=random.uniform(1, 400)
    )
    shader = Shadertoy(code, resolution=(width, height), offscreen=True)
    return Image.fromarray(np.asarray(shader.snapshot(time_float=0.0)))


def render_scales_shader_background(width, height, colors):
    template_path = resource_path("shaders", "scales", "main_scales_shader.glsl.j2")
    with template_path.open("r") as f:
        template = Template(f.read())

    randomized_colors = {
        f"color{n+1}": rgb_to_vec3_glsl(hex_to_rgb(c)) for n, c in enumerate(colors)
    }

    code = template.render(
        **randomized_colors,
    )
    with open("scaleshader.glsl", "w") as f:
        f.write(code)
    shader = Shadertoy(code, resolution=(width, height), offscreen=True)
    return Image.fromarray(np.asarray(shader.snapshot(time_float=0.0)))


def render_watercolor_shader_background(width, height, colors):
    rgb_colors = [hex_to_rgb(c) for c in colors]
    if len(rgb_colors) == 2:
        color3 = mix_colors(rgb_colors[0], rgb_colors[1], 0.5)
        rgb_colors.append(color3)
    color1 = rgb_to_vec3_glsl(rgb_colors[0])
    color2 = rgb_to_vec3_glsl(rgb_colors[1])
    color3 = rgb_to_vec3_glsl(rgb_colors[2])

    template_path = resource_path("shaders", "watercolor", "main_watercolor_shader.glsl.j2")
    with template_path.open("r") as f:
        template = Template(f.read())

    code = template.render(
        color1=color1,
        color2=color2,
        color3=color3,
        random1=random.uniform(1, 10000),
        random2=random.uniform(1, 10000)
    )
    shader = Shadertoy(code, resolution=(width, height), offscreen=True)
    return Image.fromarray(np.asarray(shader.snapshot(time_float=0.0)))


def create_linear_gradient(width, height, colors, angle=90):
    """Create linear gradient with customizable angle"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    # Convert angle to radians
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Calculate max projection distance
    corners = [
        (0, 0),
        (width, 0),
        (0, height),
        (width, height)
    ]
    projections = [x * cos_a + y * sin_a for (x, y) in corners]
    min_proj = min(projections)
    max_proj = max(projections)
    proj_range = max_proj - min_proj

    # Get gradient colors
    color1 = hex_to_rgb(colors[0])
    color2 = hex_to_rgb(colors[1])

    # Calculate gradient for each pixel
    for y in range(height):
        for x in range(width):
            # Calculate position along gradient axis
            pos = x * cos_a + y * sin_a
            t = (pos - min_proj) / proj_range if proj_range != 0 else 0.5

            # Interpolate color
            r = int(color1[0] * (1 - t) + color2[0] * t)
            g = int(color1[1] * (1 - t) + color2[1] * t)
            b = int(color1[2] * (1 - t) + color2[2] * t)

            draw.point((x, y), fill=(r, g, b))

    return img


def create_barycentric_gradient(width, height, colors):
    """Create barycentric gradient with 3 color points"""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    c0 = hex_to_rgb(colors[0])
    c1 = hex_to_rgb(colors[1])
    c2 = hex_to_rgb(colors[2])

    for y in range(height):
        for x in range(width):
            w0 = 1 - (x / width)
            w1 = (x / width) - (y / height)
            w2 = y / height

            w0 = max(0, w0)
            w1 = max(0, w1)
            w2 = max(0, w2)

            total = w0 + w1 + w2
            if total > 0:
                w0 /= total
                w1 /= total
                w2 /= total

            r = int(w0 * c0[0] + w1 * c1[0] + w2 * c2[0])
            g = int(w0 * c0[1] + w1 * c1[1] + w2 * c2[1])
            b = int(w0 * c0[2] + w1 * c1[2] + w2 * c2[2])

            pixels[x, y] = (r, g, b)

    return img


def refine_mask(alpha_channel, close_radius=2, median_radius=1):
    """
    Refine the alpha mask using morphological operations and median filtering
    - close_radius: Size for closing operation (to fill holes)
    - median_radius: Size for median filter (to smooth edges)
    """
    # Convert to binary mask for morphological operations
    binary_mask = alpha_channel.point(lambda x: 255 if x > 128 else 0)

    # Create kernel for morphological operations
    kernel_size = close_radius * 2 + 1

    # Closing operation (dilation followed by erosion)
    dilated = binary_mask.filter(ImageFilter.MaxFilter(kernel_size))
    eroded = dilated.filter(ImageFilter.MinFilter(kernel_size))

    # Median filter to smooth edges
    median_size = median_radius * 2 + 1
    return eroded.filter(ImageFilter.MedianFilter(median_size))


def remove_background(img, bg_color, fuzziness, refine=False, close_radius=2, median_radius=1, only_transparent=False):
    """
    Process image background based on mode:
    - only_transparent: Only replace fully transparent pixels (preserves all colors)
    - Normal mode: Remove specified background color with fuzziness
    """
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    alpha_data = []

    new_data = []
    if only_transparent:
        # Only process fully transparent pixels
        datas = img.getdata()
        for item in datas:
            r, g, b, a = item
            # Preserve all existing colors
            # Only fully transparent pixels will be replaced
            new_data.append((r, g, b, a))
            alpha_data.append(a)

    else:
        # Normal background removal process
        datas = img.getdata()
        bg_rgb = hex_to_rgb(bg_color)

        # Calculate fuzziness threshold
        threshold = (fuzziness / 100) * 255
        threshold_sq = threshold ** 2

        for item in datas:
            r, g, b, a_orig = item

            # Calculate distance from background color
            distance_sq = (bg_rgb[0] - r) ** 2 + (bg_rgb[1] - g) ** 2 + (bg_rgb[2] - b) ** 2

            # Preserve existing transparency by default
            new_alpha = a_orig

            # If pixel matches background (with fuzziness), make transparent
            if distance_sq <= threshold_sq:
                new_alpha = 0

            new_data.append((r, g, b, new_alpha))
            alpha_data.append(new_alpha)

    img.putdata(new_data)

    # Refinement only makes sense if we're modifying the alpha channel
    if refine:
        # Create alpha channel image
        alpha_img = Image.new('L', img.size)
        alpha_img.putdata(alpha_data)

        # Refine the mask
        refined_alpha = refine_mask(alpha_img, close_radius, median_radius)

        # Apply the refined alpha channel
        r, g, b, a = img.split()
        img.putalpha(refined_alpha)

    return img


def parse_orientation(value):
    """Convert orientation input to angle in degrees"""
    if isinstance(value, int):
        return value % 360

    if value.lower() in ORIENTATIONS:
        return ORIENTATIONS[value.lower()]

    try:
        # Try to convert string to integer
        angle = int(value)
        return angle % 360
    except ValueError:
        raise ValueError(f"Invalid orientation: {value}. Must be integer or one of: {list(ORIENTATIONS.keys())}")


@click.command()
@click.option("--fuzziness", default=1, type=click.IntRange(1, 100),
              help="Background removal sensitivity (1-100%%, default: 1) - Ignored in only-transparent mode")
@click.option("--gradient", default=None,
              help="Preset name or custom gradient (#hex,#hex or #hex,#hex,#hex)")
@click.option("--bgcolor", default="#ffffff",
              help="Background color to remove (hex format, default: #ffffff) - Ignored in only-transparent mode")
@click.option("--overwrite", is_flag=True,
              help="Overwrite existing output files")
@click.option("--refine-mask", "refine_mask_arg", is_flag=True,
              help="Apply intelligent mask refinement to clean edges and fill holes")
@click.option("--close-radius", default=2, type=click.IntRange(0, 10),
              help="Mask closing radius for filling holes (0-10, default: 2)")
@click.option("--median-radius", default=1, type=click.IntRange(0, 5),
              help="Median filter radius for smoothing edges (0-5, default: 1)")
@click.option("--only-transparent", "-ot", is_flag=True,
              help="ONLY replace transparent background, preserve all colored pixels (including white)")
@click.option("--orientation", "-o", default="vertical",
              help="Gradient orientation: 'vertical' (90°), 'horizontal' (0°), 'diagonal' (45°), 'diagonal-reverse' (135°), or custom angle (0-360)")
@click.option("--style", default="gradient",
              type=click.Choice(STYLES, case_sensitive=False),
              help=f"Background style: {', '.join(STYLES)} (default: gradient)")
@click.option("--shader-scale", default=0.8, type=float,
              help="Shader pattern scale (default: 0.8)")
@click.option("--user-gradients", type=click.Path(exists=True, dir_okay=False),
              help="Path to a JSON file with user-defined gradients (format: {name: [hex, ...], ...})")
@click.option("--combine-presets/--only-user-gradients", default=True,
              help="Combine script presets with user gradients (default: combine)")
@click.argument("imagefiles", nargs=-1, type=click.Path(exists=True))
def main(fuzziness, gradient, bgcolor, overwrite, refine_mask_arg, close_radius,
         median_radius, only_transparent, orientation, style,
         shader_scale, imagefiles, user_gradients, combine_presets):
    """
    🎨 GRADIENTIFY - Replace backgrounds with beautiful gradients 🎨

    Features:
    - Multiple background styles: gradient, shader, topographic, spiral, voronoi, squiggle, mesh, scales, watercolor
    - Two processing modes: normal background removal OR transparent-only replacement
    - Preserves all colored pixels in --only-transparent mode
    - Handles images with existing transparency
    - 2-color and 3-color preset gradients
    - Custom gradient support (e.g. --gradient "#ff0000,#00ff00")
    - User-defined gradients from JSON file (--user-gradients)
    - Option to combine user gradients with built-in presets (--combine-presets/--only-user-gradients)
    - Adjustable background detection sensitivity
    - Intelligent mask refinement
    - Batch processing with overwrite protection

    Style Options:
    --style gradient:     Linear or barycentric gradient (default)
    --style liquid:       Organic liquid-like gradient
    --style topographic:  Topographic map-like patterns
    --style spiral:       Spiral shader background
    --style voronoi:      Voronoi diagram shader background
    --style squiggle:     Squiggle shader background
    --style mesh:         Mesh gradient shader (2 or 3 colors, smooth mesh-like blend)
    --style scales:       Scales shader (2 or 3 colors, fish scale pattern)
    --style watercolor:   Watercolor shader (2 or 3 colors, soft blended effect)

    User-defined gradients:
    --user-gradients FILE    Path to a JSON file with gradients in the format:
                                {
                                    "mytwocolor": ["#123456", "#abcdef"],
                                    "mythreecolor": ["#ff0000", "#00ff00", "#0000ff"]
                                }
    --combine-presets           Combine user gradients with built-in presets (default)
    --only-user-gradients       Use only user gradients, ignore built-in presets

    Recommended to use transparent PNGs for best results.

    Examples:
    1. Use topographic background with sunset gradient:
       gradientify.py --style topographic --gradient -ot sunset image.png

    2. Gradient style with custom colors:
       gradientify.py --gradient #4159d0,#c84fc0,#ffcd70 -ot image.png

    3. Use only user gradients from a JSON file:
       gradientify.py --user-gradients mygrads.json --only-user-gradients --gradient mytwocolor image.png

    4. Combine user gradients with presets (default)(chosen randomly but will either user presets or user provided gradients):
       gradientify.py --user-gradients mygrads.json image.png

    5. Custom topographic parameters:
       gradientify.py --style topographic --gradient oceanbliss -ot photo.jpg

    6. Mesh style shader with 2 or 3 colors:
       gradientify.py --style mesh --gradient "#ff0000,#00ff00,#0000ff" image.png

    7. Scales style shader with 2 or 3 colors:
       gradientify.py --style scales --gradient "#ff0000,#00ff00,#0000ff" image.png
    """
    # Process files
    if not imagefiles:
        imagefiles = [f for f in os.listdir()
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                      and "_output_" not in f]

    # Validate gradient input
    colors = None
    preset_name = None

    # Load user gradients if provided
    user_gradients_2 = {}
    user_gradients_3 = {}
    if user_gradients:
        with open(user_gradients, "r") as f:
            user_gradients_dict = json.load(f)
        for name, icolors in user_gradients_dict.items():
            if len(icolors) == 2:
                user_gradients_2[name] = icolors
            elif len(icolors) == 3:
                user_gradients_3[name] = icolors

    # Combine or replace presets
    if user_gradients:
        if combine_presets:
            all_gradients_2 = {**GRADIENTS_2, **user_gradients_2}
            all_gradients_3 = {**GRADIENTS_3, **user_gradients_3}
        else:
            all_gradients_2 = user_gradients_2
            all_gradients_3 = user_gradients_3
    else:
        all_gradients_2 = GRADIENTS_2
        all_gradients_3 = GRADIENTS_3

    if gradient:
        if gradient.startswith("#"):
            colors = gradient.split(",")
            preset_name = "custom"
            if len(colors) not in [2, 3]:
                raise click.BadParameter("Custom gradient must have 2 or 3 colors")
        elif gradient in all_gradients_2:
            colors = all_gradients_2[gradient]
            preset_name = gradient
        elif gradient in all_gradients_3:
            colors = all_gradients_3[gradient]
            preset_name = gradient
        else:
            raise click.BadParameter(f"Unknown gradient: '{gradient}'")

    # Parse orientation
    try:
        angle = parse_orientation(orientation)
    except ValueError as e:
        raise click.BadParameter(str(e))

    for path in imagefiles:
        try:
            # Handle images with existing transparency
            with Image.open(path) as img:
                # Convert to RGBA if needed
                if img.mode == 'P':
                    img = img.convert('RGBA')
                elif img.mode == 'RGB':
                    img = img.convert('RGBA')

                # Get base name for output
                base = os.path.splitext(path)[0]

                # Select gradient if not specified
                current_preset = preset_name
                current_colors = colors

                if not current_colors and not gradient:
                    # Pick a random gradient for this image
                    if style in {"gradient", "topographic", "squiggle"}:
                        all_gradients = list(all_gradients_2.items()) + list(all_gradients_3.items())
                    else:
                        all_gradients = list(all_gradients_2.items())
                    current_preset, current_colors = random.choice(all_gradients)

                # Determine output filename
                output = f"{base}_output_{current_preset}.png"
                if style != "gradient":
                    output = f"{base}_output_{current_preset}_{style}.png"

                # Skip existing files unless overwrite is specified
                if os.path.exists(output) and not overwrite:
                    print(f"⏩ Skipping {path} (output exists: {output})")
                    continue

                mode = "TRANSPARENT-ONLY" if only_transparent else "NORMAL"
                print(f"🔧 Processing {path} [{mode}] with gradient: {current_preset} ({style} style)...")

                # Create background based on style
                match style:
                    case "liquid":
                        print("   └── Rendering liquid gradient background...")
                        if len(current_colors) != 2:
                            raise click.BadParameter("Liquid gradient style requires exactly 2 colors.")
                        gradient_img = render_liquid_gradient_shader_background(
                            img.width,
                            img.height,
                            current_colors,
                        )
                    case "spiral":
                        print("   └── Rendering spiral shader background...")
                        if len(current_colors) != 2:
                            raise click.BadParameter("Spiral shader style requires exactly 2 colors.")
                        gradient_img = render_spiral_shader_background(
                            img.width,
                            img.height,
                            current_colors,
                        )
                    case "voronoi":
                        print("   └── Rendering Voronoi gradient background...")
                        if len(current_colors) != 2:
                            raise click.BadParameter("Voronoi gradient style requires exactly 2 colors.")
                        gradient_img = render_voronoi_gradient_shader_background(
                            img.width,
                            img.height,
                            current_colors,
                        )
                    case "topographic":
                        print("   └── Rendering topographic background...")
                        if len(current_colors) > 4:
                            raise click.BadParameter("Topographic shader style requires 2, 3, or 4 colors.")
                        gradient_img = render_topographic_shader_background(
                            img.width,
                            img.height,
                            current_colors,
                            scale=shader_scale,
                        )
                    case "gradient" if len(current_colors) == 2:
                        gradient_img = create_linear_gradient(
                            img.width, img.height, current_colors, angle
                        )
                    case "gradient":
                        print("   └── Note: Orientation ignored for 3-color gradients")
                        gradient_img = create_barycentric_gradient(
                            img.width, img.height, current_colors
                        )
                    case "squiggle":
                        print("   └── Rendering squiggle shader background...")
                        # Use 4 colors for squiggle, derive if needed
                        if len(current_colors) > 4:
                            raise click.BadParameter("Squiggle shader style requires 2, 3, or 4 colors.")
                        gradient_img = render_squiggle_shader_background(
                            img.width,
                            img.height,
                            current_colors,
                        )
                    case "mesh":
                        print("   └── Rendering mesh gradient shader background...")
                        if len(current_colors) not in [2, 3]:
                            raise click.BadParameter("Mesh gradient style requires 2 or 3 colors.")
                        gradient_img = render_mesh_gradient_shader_background(
                            img.width,
                            img.height,
                            current_colors,
                        )
                    case "scales":
                        print("   └── Rendering scales shader background...")
                        if len(current_colors) not in [2, 3]:
                            raise click.BadParameter("Scales shader style requires exactly 2 or 3 colors.")
                        gradient_img = render_scales_shader_background(
                            img.width,
                            img.height,
                            current_colors,
                        )
                    case "watercolor":
                        print("   └── Rendering watercolor shader background...")
                        if len(current_colors) not in [2, 3]:
                            raise click.BadParameter("Watercolor shader style requires 2 or 3 colors.")
                        gradient_img = render_watercolor_shader_background(
                            img.width,
                            img.height,
                            current_colors,
                        )

                # Process background based on mode
                transparent_img = remove_background(
                    img,
                    bgcolor,
                    fuzziness,
                    refine=refine_mask_arg,
                    close_radius=close_radius,
                    median_radius=median_radius,
                    only_transparent=only_transparent
                )

                # Composite images
                gradient_img.paste(transparent_img, (0, 0), transparent_img)

                # Save result
                gradient_img.save(output, format='PNG')
                print(f"✅ Saved: {output}")

                # Show refinement info if used
                if refine_mask:
                    print(f"   └── Applied mask refinement (close: {close_radius}, median: {median_radius})")

        except Exception as e:
            print(f" Error processing {path}: {str(e)}")
            raise e


if __name__ == "__main__":
    main()
