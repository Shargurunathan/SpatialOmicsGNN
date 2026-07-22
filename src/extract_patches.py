import openslide
import os

# Open the Whole Slide Image
slide = openslide.OpenSlide("data/CMU-1.svs")

# Create output folder
os.makedirs("outputs/patches", exist_ok=True)

# Get slide dimensions
width, height = slide.dimensions
print(width, height)

patch_count = 1

for y in range(0, height, 256):
    for x in range(0, width, 256):
        patch = slide.read_region((x, y), 0, (256, 256))
        patch = patch.convert("RGB")
        patch.save(f"outputs/patches/patch_{patch_count}.png")
        patch_count += 1

print("All patches extracted successfully!")