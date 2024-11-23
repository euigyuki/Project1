import os
from PIL import Image

# Directory containing the JPG files
directory = "."

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a JPG
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        # Open the JPG image
        img = Image.open(os.path.join(directory, filename))
        # Replace .jpg/.jpeg with .png
        new_filename = os.path.splitext(filename)[0] + ".png"
        # Save as PNG
        img.save(os.path.join(directory, new_filename), "PNG")
        print(f"Converted {filename} to {new_filename}")

print("All JPG files have been converted to PNG.")
