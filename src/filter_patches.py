from PIL import Image
import os

# Input and output folders
input_folder = "outputs/patches"
output_folder = "outputs/tissue_patches"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Threshold for white pixels
threshold = 220

# Process every patch
for filename in os.listdir(input_folder):

    if filename.endswith(".png"):

        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        grayscale = image.convert("L")

        pixels = list(grayscale.getdata())

        white_pixels = sum(pixel > threshold for pixel in pixels)

        white_ratio = white_pixels / len(pixels)

        if white_ratio < 0.80:
            image.save(os.path.join(output_folder, filename))

print("Finished filtering tissue patches!")