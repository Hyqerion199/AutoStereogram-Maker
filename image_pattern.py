import numpy as np
from PIL import Image

def create_pattern(width, height, square_size=2):
  """Creates a random dot pattern image of the given size, with dots of the given square size.

  Args:
    width: The width of the image in pixels.
    height: The height of the image in pixels.
    square_size: The size of the squares in pixels.

  Returns:
    A PIL Image object containing the dot pattern.
  """

  # Create a blank image.
  image = Image.new('RGB', (width, height), (255, 255, 255))

  # Create a random dot pattern.
  for i in range(0, width, square_size):
    for j in range(0, height, square_size):
      if np.random.rand() > 0.3:
        for x in range(i, i + square_size):
          for y in range(j, j + square_size):
            image.putpixel((x, y), (0, 0, 0))
      else:
        for x in range(i, i + square_size):
          for y in range(j, j + square_size):
            image.putpixel((x, y), (200, 200, 200))

  return image

# Create a 100x100 pixel dot pattern image, with 2x2 squares.
pattern = create_pattern(100, 100, square_size=2)

# Save the image to a file.
pattern.save('pattern.png')
