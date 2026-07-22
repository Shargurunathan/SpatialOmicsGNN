import os
import csv
import openslide

# Open the Whole Slide Image
slide = openslide.OpenSlide("data/CMU-1.svs")

# Get slide dimensions
width, height = slide.dimensions

# Patch size
patch_size = 256

# Folder containing extracted patches
patch_folder = "outputs/patches"

# Get all patch filenames
patch_files = sorted(
    os.listdir(patch_folder),
    key=lambda x: int(x.split("_")[1].split(".")[0])
)

# Create CSV file
csv_path = "outputs/patch_coordinates.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)

    # Header
    writer.writerow(["Patch_Name", "X", "Y"])

    patch_index = 0

    # Generate coordinates
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):

            if patch_index >= len(patch_files):
                break

            writer.writerow([
                patch_files[patch_index],
                x,
                y
            ])

            patch_index += 1

print("Coordinate file created successfully!")
print("Total patches:", patch_index)
print("Saved as:", csv_path)