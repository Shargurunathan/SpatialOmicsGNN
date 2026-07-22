# Import the os module to work with folders and file paths
import os

# Import the PyTorch library
import torch

# Import the Image class from PIL to open image files
from PIL import Image

# Import pretrained models and image transformation utilities from torchvision
from torchvision import models, transforms


# Load the pretrained ResNet-50 model (trained on the ImageNet dataset)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Remove the final Fully Connected (classification) layer.
# The model will now output only the feature vector (2048 features).
model = torch.nn.Sequential(*list(model.children())[:-1])

# Set the model to evaluation mode.
# This disables training-specific layers like Dropout and Batch Normalization updates.
model.eval()


# Define the preprocessing steps that every image must go through
transform = transforms.Compose([

    # Resize every image to 224 × 224 pixels
    transforms.Resize((224, 224)),

    # Convert the PIL image into a PyTorch tensor
    # Shape changes from (H, W, C) → (C, H, W)
    # Pixel values change from 0–255 → 0–1
    transforms.ToTensor(),

    # Normalize each RGB channel using the ImageNet mean and standard deviation
    # This makes the input similar to the data used to train ResNet-50
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # Mean values for Red, Green, Blue
        std=[0.229, 0.224, 0.225]     # Standard deviation for Red, Green, Blue
    )
])


# Path of the folder containing all extracted tissue patches
patch_folder = "outputs/tissue_patches"

# Get a list of all image filenames inside the tissue_patches folder
patch_files = sorted(
    os.listdir(patch_folder),
    key=lambda name: int(name.split("_")[1].split(".")[0])
)

# Create an empty list to store the extracted feature vectors
features = []

# Process every tissue patch one by one
# enumerate() gives:
#   i          → index number: 0, 1, 2, ...
#   patch_file → filename: patch_1.png, patch_2.png, ...
for i, patch_file in enumerate(patch_files):

    # Create the complete path to the current patch image
    # Example:
    # outputs/tissue_patches/patch_1.png
    image_path = os.path.join(patch_folder, patch_file)

    # Open the patch image using PIL
    # convert("RGB") ensures that the image has exactly 3 channels:
    # Red, Green and Blue
    image = Image.open(image_path).convert("RGB")

    # Apply preprocessing:
    # 1. Resize to 224 × 224
    # 2. Convert image to tensor
    # 3. Normalize using ImageNet mean and standard deviation
    #
    # Shape after transformation:
    # [3, 224, 224]
    image = transform(image)

    # Add a batch dimension because ResNet expects:
    # [Batch, Channels, Height, Width]
    #
    # Before:
    # [3, 224, 224]
    #
    # After:
    # [1, 3, 224, 224]
    image = image.unsqueeze(0)

    # Disable gradient calculation because we are only
    # extracting features, not training ResNet50.
    #
    # This reduces memory usage and improves speed.
    with torch.no_grad():

        # Pass the current tissue patch through ResNet50.
        #
        # Input shape:
        # [1, 3, 224, 224]
        #
        # Output shape:
        # [1, 2048, 1, 1]
        feature = model(image)

    # Store this patch's 2048-dimensional feature vector
    # in the features list
    features.append(feature)

    # Print progress after every 100 processed patches
    #
    # Example:
    # Processed 100/4880 patches
    # Processed 200/4880 patches
    # ...
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(patch_files)} patches")


# Save all extracted feature vectors into one file
torch.save(features, "outputs/features.pt")


# Print final information
print("\nFeature extraction completed successfully!")
print("Total feature vectors:", len(features))
print("Shape of first feature vector:", features[0].shape)


 