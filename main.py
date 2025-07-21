import click
import os
import random
import math
import colorsys
import numpy as np
from jinja2 import Template
from PIL import Image, ImageDraw, ImageColor, ImageFilter
from wgpu_shadertoy import Shadertoy
import sys
from pathlib import Path

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


def render_shader_background(width, height, colors, scale=1.0, time_speed=0.5):
    pass


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
    breakpoint()
    shader = Shadertoy(code, resolution=(width, height), offscreen=True)
    return Image.fromarray(np.asarray(shader.snapshot(time_float=1)))


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
              type=click.Choice(['gradient', 'liquid', 'voronoi', 'topographic', 'spiral'], case_sensitive=False),
              help="Background style: gradient (linear/barycentric), liquid (liquid gradient), spiral, or topographic (map-like)")
@click.option("--shader-scale", default=0.8, type=float,
              help="Shader pattern scale (default: 0.8)")
@click.option("--shader-speed", default=0.3, type=float,
              help="Shader animation speed (default: 0.3)")
@click.argument("imagefiles", nargs=-1, type=click.Path(exists=True))
def main(fuzziness, gradient, bgcolor, overwrite, refine_mask_arg, close_radius,
         median_radius, only_transparent, orientation, style,
         shader_scale, shader_speed, imagefiles):
    """
    🎨 GRADIENTIFY - Replace backgrounds with beautiful gradients 🎨

    Features:
    - Multiple background styles: gradient, shader, topographic
    - Two processing modes: normal background removal OR transparent-only replacement
    - Preserves all colored pixels in --only-transparent mode
    - Handles images with existing transparency
    - 26 preset gradients with 2-3 colors
    - Custom gradient support
    - Adjustable background detection sensitivity
    - Intelligent mask refinement
    - Batch processing with overwrite protection

    Style Options:
    --style gradient:     Linear or barycentric gradient (default)
    --style shader:       Organic animated patterns
    --style topographic:  Topographic map-like patterns

    Shader Mode:
    --shader-scale:       Pattern scale (0.5-2.0, default: 0.8)
    --shader-speed:       Animation speed (0.0-1.0, default: 0.3)
    --debug-shader:       Open visible browser for debugging
    --save-shader-html:   Save generated shader HTML to file

    Examples:
    1. Use topographic background with sunset gradient:
       gradientify.py --style topographic --gradient sunset image.png

    2. Debug organic shader:
       gradientify.py --style shader --gradient heatwave --debug-shader logo.png

    3. Custom topographic parameters:
       gradientify.py --style topographic --gradient oceanbliss photo.jpg
    """
    # Process files
    if not imagefiles:
        imagefiles = [f for f in os.listdir()
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                      and "_output_" not in f]

    # Validate gradient input
    colors = None
    preset_name = None

    if gradient:
        if gradient.startswith("#"):
            colors = gradient.split(",")
            preset_name = "custom"
            if len(colors) not in [2, 3]:
                raise click.BadParameter("Custom gradient must have 2 or 3 colors")
        elif gradient in GRADIENTS_2:
            colors = GRADIENTS_2[gradient]
            preset_name = gradient
        elif gradient in GRADIENTS_3:
            colors = GRADIENTS_3[gradient]
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
                    if style in {"gradient", "topographic"}:
                        # Pick randomly from both 2 and 3 color gradients
                        all_gradients = list(GRADIENTS_2.items()) + list(GRADIENTS_3.items())
                    else:
                        # For certain shader styles, only use 2-color gradients
                        all_gradients = list(GRADIENTS_2.items())
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
            print(f"�� Error processing {path}: {str(e)}")
            raise e


if __name__ == "__main__":
    main()
