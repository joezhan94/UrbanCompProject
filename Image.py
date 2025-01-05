from PIL import Image

# Open the PNG image
image = Image.open('dataset/train/amsterdam_0_0_mask.png')

# Inspect basic metadata
print("Format:", image.format)  # Output: PNG
print("Size:", image.size)      # Output: (width, height)
print("Mode:", image.mode)      # Output: e.g., RGB, RGBA, L (grayscale)

# Access pixel data
pixels = list(image.getdata())
print("First 10 pixels:", pixels[:10])  # Print the first 10 pixels

# Inspect additional metadata (if available)
print("Metadata:", image.info)