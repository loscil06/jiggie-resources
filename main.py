import click
import os
import random
import math
from PIL import Image, ImageDraw, ImageColor, ImageFilter


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

def hex_to_rgb(hex_color):
    """Convert #RRGGBB hex to RGB tuple"""
    return ImageColor.getrgb(hex_color)

def create_linear_gradient(width, height, colors, angle=90):
    """
    Create linear gradient with customizable angle
    - angle: 0=left-right, 90=top-bottom, 45=diagonal, etc.
    """
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
    kernel = ImageFilter.Kernel(
        (kernel_size, kernel_size),
        [1] * (kernel_size * kernel_size),
        scale=1
    )

    # Closing operation (dilation followed by erosion)
    dilated = binary_mask.filter(ImageFilter.MaxFilter(kernel_size))
    eroded = dilated.filter(ImageFilter.MinFilter(kernel_size))

    # Median filter to smooth edges
    median_size = median_radius * 2 + 1
    smoothed = eroded.filter(ImageFilter.MedianFilter(median_size))

    # Convert back to original alpha range
    return smoothed.point(lambda x: x * 255 // 255)

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
@click.option("--refine-mask", is_flag=True,
              help="Apply intelligent mask refinement to clean edges and fill holes")
@click.option("--close-radius", default=2, type=click.IntRange(0, 10),
              help="Mask closing radius for filling holes (0-10, default: 2)")
@click.option("--median-radius", default=1, type=click.IntRange(0, 5),
              help="Median filter radius for smoothing edges (0-5, default: 1)")
@click.option("--only-transparent", "-ot", is_flag=True,
              help="ONLY replace transparent background, preserve all colored pixels (including white)")
@click.option("--orientation", "-o", default="vertical",
              help="Gradient orientation: 'vertical' (90°), 'horizontal' (0°), 'diagonal' (45°), 'diagonal-reverse' (135°), or custom angle (0-360)")
@click.argument("imagefiles", nargs=-1, type=click.Path(exists=True))
def main(fuzziness, gradient, bgcolor, overwrite, refine_mask, close_radius, median_radius, only_transparent, orientation, imagefiles):
    """
    🎨 GRADIENTIFY - Replace backgrounds with beautiful gradients 🎨

    Features:
    - Customizable gradient orientation (angle or named direction)
    - Two processing modes: normal background removal OR transparent-only replacement
    - Preserves all colored pixels in --only-transparent mode
    - Handles images with existing transparency
    - 16 preset gradients with 2-3 colors
    - Custom gradient support
    - Adjustable background detection sensitivity
    - Intelligent mask refinement
    - Batch processing with overwrite protection

    Gradient Orientation:
    --orientation / -o:
        vertical (90°)     Top to bottom
        horizontal (0°)    Left to right
        diagonal (45°)     Top-left to bottom-right
        diagonal-reverse (135°)  Top-right to bottom-left
        <angle>            Custom angle in degrees (0-360)

    Examples:
    1. Diagonal gradient:
       gradientify.py --orientation diagonal image.png

    2. Custom angle gradient:
       gradientify.py --orientation 30 photo.jpg

    3. Horizontal gradient with transparent mode:
       gradientify.py --only-transparent -o horizontal logo.png
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

                # Skip existing files unless overwrite is specified
                if os.path.exists(output) and not overwrite:
                    print(f"⏩ Skipping {path} (output exists: {output})")
                    continue

                mode = "TRANSPARENT-ONLY" if only_transparent else "NORMAL"
                print(f"🔧 Processing {path} [{mode}] with gradient: {current_preset} at {angle}°...")

                # Create gradient background
                if len(current_colors) == 2:
                    gradient_img = create_linear_gradient(img.width, img.height, current_colors, angle)
                else:
                    # 3-color gradients use fixed barycentric method
                    print("   └── Note: Orientation ignored for 3-color gradients")
                    gradient_img = create_barycentric_gradient(img.width, img.height, current_colors)

                # Process background based on mode
                transparent_img = remove_background(
                    img,
                    bgcolor,
                    fuzziness,
                    refine=refine_mask,
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

if __name__ == "__main__":
    main()
