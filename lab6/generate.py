from math import ceil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def semitone(old_image):
    return (0.3 * old_image[:, :, 0] + 0.59 * old_image[:, :, 1] + 0.11 * old_image[:, :, 2]).astype(np.uint8)


def simple_binarization(old_image, threshold, semitone_needed=True):
    if semitone_needed:
        semi = semitone(old_image)
    else:
        semi = old_image
    new_image = np.zeros(shape=semi.shape)

    new_image[semi > threshold] = 255

    return new_image.astype(np.uint8)

SENTENCE = "seni seviyorum"

if __name__ == '__main__':
    # Load font - make sure DejaVuSans.ttf is available in your system
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 52)
    except IOError:
        # Fallback to default font if DejaVu is not available
        font = ImageFont.load_default()
        print("Warning: DejaVuSans.ttf not found, using default font")

    # Create output directory if it doesn't exist
    import os
    os.makedirs("lab6/out/turkish_language", exist_ok=True)

    # Generate image for each letter
    for letter in SENTENCE:
        # Calculate image width with some padding
        width = font.getlength(letter) + 20
        
        # Create blank white image
        img = Image.new(mode="RGB", size=(ceil(width), 60), color="white")
        draw = ImageDraw.Draw(img)
        
        # Draw the letter (centered vertically)
        draw.text((10, 2), letter, (0, 0, 0), font=font)
        
        # Binarize and save
        binarized = Image.fromarray(simple_binarization(np.array(img), 120), 'L')
        binarized.save(f"lab6/out/turkish_language/{letter}.png")

    print(f"Generated {len(SENTENCE)} Turkish language images")