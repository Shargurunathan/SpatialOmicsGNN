import openslide
import matplotlib.pyplot as plt

# Open the Whole Slide Image
slide = openslide.OpenSlide("data/CMU-1.svs")

# Create a thumbnail
thumbnail = slide.get_thumbnail((1000, 1000))

# Display the thumbnail
plt.imshow(thumbnail)
plt.title("Whole Slide Image Thumbnail")
plt.axis("off")
plt.show()