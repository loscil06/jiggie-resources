import click
import os
import random
import math
import base64
import colorsys
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageColor, ImageFilter
from wgpu_shadertoy import Shadertoy


# Preset gradients
GRADIENTS = {
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

ORGANIC_SHADER_TEMPLATE = """
// Organic animated glow shader
uniform vec3 iColor1;
uniform vec3 iColor2;
uniform vec3 iColor3;

vec2 hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise(vec2 p) {
    const float K1 = 0.366025404;
    const float K2 = 0.211324865;

    vec2 i = floor(p + (p.x + p.y) * K1);
    vec2 a = p - i + (i.x + i.y) * K2;
    vec2 o = (a.x > a.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec2 b = a - o + K2;
    vec2 c = a - 1.0 + 2.0 * K2;

    vec3 h = max(0.5 - vec3(dot(a,a), dot(b,b), dot(c,c)), 0.0);
    vec3 n = h * h * h * h * vec3(
        dot(a, hash(i + 0.0)),
        dot(b, hash(i + o)),
        dot(c, hash(i + 1.0))
    );

    return dot(n, vec3(70.0));
}

float fbm(vec2 p){
  float a = .5;
  float n = 0.;
  for(float i=0.; i<8.; i++){
    n += a * noise(p);
    p *= 2.;
    a *= .5;
  }
  return n;
}

mat2 rotate(float a){
  float s = sin(a);
  float c = cos(a);
  return mat2(c,-s,s,c);
}

vec3 glow(float v, float r, float ins, vec3 col){
  float dist = pow(r/v,ins);
  return 1.-exp(-dist*col);
}

void mainImage(out vec4 O, in vec2 I){
  vec2 R = iResolution.xy;
  vec2 uv = (I*2.-R)/R.y;
  O.rgb = vec3(0);
  O.a = 1.;

  uv *= 2.;
  vec2 p = uv;

  float l = length(uv)-iTime*0.3;
  p*=rotate(l);

  float n = noise(uv);
  p += n*.5;

  vec3 c1 = iColor1;
  vec3 c2 = iColor2;
  vec3 c3 = iColor3;

  n = fbm(p*0.4);
  O.rgb = glow(n, 0.2, 2., c1);

  n = fbm(p*0.2*rotate(.1));
  c2 = glow(n, 0.3, 2., c2);

  O.rgb *= c2;
}
"""

TOPOGRAPHIC_SHADER_CODE = """
precision highp float;

// Declare uniforms with binding points
layout(binding = 0) uniform vec3 u_color1;
layout(binding = 1) uniform vec3 u_color2;
layout(binding = 2) uniform vec3 u_color3;
layout(binding = 3) uniform vec3 u_color4;
layout(binding = 4) uniform float u_scale;

#define u_resolution iResolution.xy

#define M_PI 3.1415926535897932384626433832795

float dot2( in vec2 v ) { return dot(v,v); }
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }

vec2 random2(vec2 st){
    st = vec2(dot(st,vec2(127.1,311.7)), dot(st,vec2(269.5,183.3)));
    return -1.0 + 2.0 * fract(sin(st) * 43758.5453);
}

float noise2(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    vec2 u = f*f*(3.0-2.0*f);

    vec2 r00 = random2(i + vec2(0.0,0.0));
    vec2 r10 = random2(i + vec2(1.0,0.0));
    vec2 r01 = random2(i + vec2(0.0,1.0));
    vec2 r11 = random2(i + vec2(1.0,1.0));

    float a = mix( dot(r00, f - vec2(0.0,0.0)),
                   dot(r10, f - vec2(1.0,0.0)), u.x);
    float b = mix( dot(r01, f - vec2(0.0,1.0)),
                   dot(r11, f - vec2(1.0,1.0)), u.x);
    return mix(a, b, u.y);
}

float fbm2(vec2 coord) {
    float value = 0.0;
    float scale = 0.5;
    for (int i = 0; i < 4; i++) {
        value += noise2(coord) * scale;
        coord *= 2.0;
        scale *= 0.5;
    }
    return value + 0.2;
}

#define NUM_COLORS 4
const float stepSize = 0.02;
const float epsilon = 0.001;
// FIXED: Precomputed constant values
const float invIdxMult = 4.0 * stepSize * 3.0;  // NUM_COLORS is 4

float height(vec2 pixel) {
    return pow(fbm2(pixel * 0.0072), 1.248);
}

float plateau(float h) {
    return floor(h / stepSize);
}

float contour(vec2 p, float dx) {
    vec3 d = vec3(dx, 0.0, -dx);
    float h = plateau(height(p));

    float a = 0.0;
    a += plateau(height(p + d.xy));
    a += plateau(height(p + d.yx));
    a += plateau(height(p + d.zy));
    a += plateau(height(p + d.yz));
    a += plateau(height(p + d.xx));
    a += plateau(height(p + d.zz));
    a += plateau(height(p + d.xz));
    a += plateau(height(p + d.zx));

    return smoothstep(0.0, epsilon, abs(a / 8.0 - h));
}

vec3 landform(vec2 p) {
    float h = plateau(height(p));
    // FIXED: Avoid runtime constant expression
    float index = clamp(h * invIdxMult, 0.0, 3.0);  // NUM_COLORS-1 = 3
    int i = int(index);
    float m = fract(index);  // Equivalent to mod(index, 1.0)

    // Create color array directly
    vec3 color0 = u_color1;
    vec3 color1 = u_color2;
    vec3 color2 = u_color3;
    vec3 color3 = u_color4;

    if (i == 0) return mix(color0, color1, m);
    if (i == 1) return mix(color1, color2, m);
    if (i == 2) return mix(color2, color3, m);
    return color3;
}

const float scaler = 0.485;
const float contourStroke = 0.38;

#define CONTOUR 1
vec3 shade(vec2 p) {
    #if CONTOUR
    vec3 ctColor = vec3(0.1);
    float ct = contour(p, contourStroke * scaler);
    return mix(landform(p), ctColor, ct);
    #else
    return landform(p);
    #endif
}

#define AA 0

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 I = fragCoord;
    vec2 p = I * u_scale * scaler;
    vec3 accum = shade(p);
    fragColor = vec4(accum, 1.0);
}
"""

def hex_to_rgb(hex_color):
    return ImageColor.getrgb(hex_color)

def rgb_to_float(rgb):
    return tuple(c/255.0 for c in rgb)

def rgb_to_hsl(rgb):
    r, g, b = [x/255.0 for x in rgb]
    return colorsys.rgb_to_hls(r, g, b)

def hsl_to_rgb(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r*255), int(g*255), int(b*255))

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
    r, g, b = [x/255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_l = max(0.0, min(1.0, l + percent/100.0))
    r, g, b = colorsys.hls_to_rgb(h, new_l, s)
    return (int(r*255), int(g*255), int(b*255))

def mix_colors(rgb1, rgb2, factor):
    """Mix two RGB colors with given factor (0.0-1.0)"""
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    r = int(r1*(1-factor) + r2*factor)
    g = int(g1*(1-factor) + g2*factor)
    b = int(b1*(1-factor) + b2*factor)
    return (r, g, b)

def render_shader_background(width, height, colors, scale=1.0, time_speed=0.5):
    shader_colors = derive_shader_colors(colors)
    code = ORGANIC_SHADER_TEMPLATE
    for i, rgb in enumerate(shader_colors):
        code = code.replace(f"iColor{i+1}", f"vec3({rgb[0]/255.0:.4f}, {rgb[1]/255.0:.4f}, {rgb[2]/255.0:.4f})")
    shader = Shadertoy(code, resolution=(width, height), offscreen=True)
    return Image.fromarray(np.asarray(shader.snapshot(time_float=0.0)))

def render_topographic_background(width, height, colors, scale=1.0):
    """Render topographic shader background using wgpu-shadertoy"""
    shader_colors = derive_topographic_colors(colors)

    # Create shader instance
    shader = Shadertoy(
        TOPOGRAPHIC_SHADER_CODE,
        resolution=(width, height),
        offscreen=True
    )

    # Set uniforms directly on the shader object
    shader.uniforms = {
        "u_scale": scale
    }

    # Add color uniforms
    for i, color in enumerate(shader_colors[:4], 1):
        r, g, b = rgb_to_float(color)
        shader.uniforms[f"u_color{i}"] = [r, g, b]

    # Capture at time=0
    snapshot = shader.snapshot(time_float=0.0)
    return Image.fromarray(np.asarray(snapshot))

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
            distance_sq = (bg_rgb[0] - r)**2 + (bg_rgb[1] - g)**2 + (bg_rgb[2] - b)**2

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
              type=click.Choice(['gradient', 'shader', 'topographic'], case_sensitive=False),
              help="Background style: gradient (linear/barycentric), shader (organic patterns), or topographic (map-like)")
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
        elif gradient in GRADIENTS:
            colors = GRADIENTS[gradient]
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
                    current_preset = random.choice(list(GRADIENTS.keys()))
                    current_colors = GRADIENTS[current_preset]

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
                if style == "shader":
                    print("   └── Rendering organic shader background...")
                    gradient_img = render_shader_background(
                        img.width,
                        img.height,
                        current_colors,
                        scale=shader_scale,
                        time_speed=shader_speed,
                    )
                elif style == "topographic":
                    print("   └── Rendering topographic background...")
                    gradient_img = render_topographic_background(
                        img.width,
                        img.height,
                        current_colors,
                        scale=shader_scale,
                    )
                elif len(current_colors) == 2:
                    gradient_img = create_linear_gradient(
                        img.width, img.height, current_colors, angle
                    )
                else:
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
            print(f"❌ Error processing {path}: {str(e)}")
            raise e

if __name__ == "__main__":
    main()
