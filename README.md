# Jig Gradientify: Replace Image Backgrounds with Beautiful Gradients & Patterns

Gradientify is a powerful command-line tool that replaces image backgrounds with stunning gradients and shader-based patterns. Perfect for making jigs less boring, especially with images where there are lone subjects and/or the background is very dull. This tool will make those images more entertaining for online jigsaw puzzles. You could also use this tool for whatever.

Features

    🎨 Multiple background styles: Linear gradients, liquid effects, topographic maps, spirals and more

    ✨ Smart background removal: Remove backgrounds with adjustable sensitivity

    🖼️ Transparency support: Preserve existing transparency or replace only transparent areas

    🧪 Mask refinement: Clean up edges with morphological operations

    🔄 Batch processing: Process multiple images at once

    🏷️ Preset gradients: Choose from a vast range of (semi)professionally designed color schemes

    🎚️ User Provided or Custom gradients: Pass your own color combinations or even have your own library of gradients

    📐 Orientation control: Adjust gradient direction

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/gradientify.git
cd gradientify
```

### Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

## Important notes
It's highly recommended your images to be in PNG format and have already transparency. While the script can handle images without transparency with solid color backgrounds, the script will not be able to accurately remove the background in most cases. For best results, use images with transparent backgrounds. Below some recommendations how to make your images transparent:
1. **Use a photo editing tool**: Open your image in software like Photoshop, GIMP, Krita or an online editor. Use the magic wand or lasso tool to select the background and delete it, leaving a transparent layer.
2. **Use an online background remover**: Websites like `withoutbg.com` or `pixlr.com` can automatically remove backgrounds from images, making them transparent. Note: some of these websites have limitations on how many you can process per day. Personally, `withoutbg.com` is my favorite.
3. **Ask the original creator**: If you have access to the original image files, ask the creator to provide a version with a transparent background.
4. Once you have your images with transparency, use the tool with the `--only-transparent`  or `-ot` option to replace only the transparent areas with gradients or patterns.

### Basic Usage

Replace background with default gradient, replacing only transparent areas:
```bash
python main.py image.png -ot
```

Use a specific preset gradient:
```bash
python main.py --gradient sunset -ot image.png
```

Use topographic style with ocean bliss gradient:
```bash
python main.py --style topographic --gradient -ot oceanbliss image.png
```

### Advanced Usage

Batch Processing

Process all images in current directory:
```bash
python main.py
```

Process specific images with custom settings:
```bash
python main.py --style liquid --gradient fireandice -ot image1.png image2.jpg
```

### Background Styles

1. **Linear/Barycentric (Gradient)** :
Creates smooth color transitions. Supports linear (2 colors) and barycentric (3 colors) gradients.
![Alt gradient example](./doc/alpha-ykk-transparent_output_cyberpunk.png)

2. **Mesh (Gradient)**:
Creates complex, multi-color transitions across a shape by creating a grid of random nodes. Supports 2-3 colors.
![Alt mesh example](./doc/5_output_goldenlagoon_mesh.png)

3. **Liquid**:
Organic, flowing patterns that simulate liquid movements. Requires exactly 2 colors.
![Alt liquid example](./doc/rin-touhou-transparent_output_spicy_sunset_liquid.png)

4. **Voronoi**:
Cellular patterns based on Voronoi diagrams. Creates a geometric, tech-inspired look.
![Alt voronoi example](./doc/1_output_bubblegum_voronoi.png)

5. **Topographic**:
Contour map-like patterns that simulate elevation. Works best with 2-4 colors.
![Alt voronoi example](./doc/2_output_weddingdayblues_topographic.png)

6. **Spiral**:
Hypnotic spiral patterns that create visual depth. Requires exactly 2 colors.
![Alt voronoi example](./doc/3_output_skymint_spiral.png)

7. **Squiggle**:
Playful squiggly line patterns that add artistic flair. Supports 2-4 colors.
![Alt voronoi example](./doc/4_output_goldenlagoon_squiggle.png)

8. **Scales**:
Scales create a fish-like texture with overlapping shapes. Requires 2 or 3 colors.
![Alt scales example](./doc/6_output_paloalto_scales.png)

9. **Watercolor**:
Soft, blended patterns that mimic watercolor painting. Requires 2 or 3 colors.
![Alt watercolor example](./doc/7_output_abyssalblue_watercolor.png)

### Advanced Options
To check out all available options, run the following command:
```bash
python main.py --help
```

### User Provided Gradients
You can provide your own gradients using the following JSON format:
```json
{
    "my_custom_gradient_2": ["#ff0000", "#00ff00"],
    "my_custom_gradient_3": ["#ff0000", "#00ff00", "#0000ff"]
}
```

Then, you can pass your custom gradients passing this JSON file using the `--user-gradients` option:
```bash
python main.py --user-gradients mypresets.json --gradient my_custom_gradient_2 image.png
```

When you let the script choose random gradients, you can choose between either combine them with script's preset gradients or to only use your gradients by passing either `--combine-presets` or `--only-user-gradients` respectively:
```bash
# Leaving it defaults to combine both, although you can also use --combine-presets to be explicit
python main.py image.png -ot

# Only use your gradients
python main.py --only-user-gradients image.png -ot
```

Currently, the program supports only 2-3 color gradients.

Contribution

Contributions are welcome! Please open an issue or pull request for:

    New gradient presets

    Additional shader styles

    Performance improvements

    Documentation enhancements
