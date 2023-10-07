import numpy as np
from PIL import Image
import random
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DPI = 72  # Output device has 72 pixels per inch
E = round(2.5 * DPI)  # Eye separation is assumed to be 2.5 in
mu = 1 / 3.0  # Depth of field (fraction of viewing distance)
# maxX, maxY = 256, 256  # Image and object are both maxX by maxY pixels


def separation(Z):
    factor = 2  # Increase this to increase the shift
    return round(factor * (1 - mu * Z) * E / (2 - mu * Z))


depth_map = Image.open("test.jpg")
depth_map_width, depth_map_height = depth_map.size
depth_map_grayscale = depth_map.convert('I')

Z = np.array(depth_map_grayscale)

maxX, maxY = depth_map.size

totalimagelist = [[0 for x in range(maxX)]
                  for y in range(maxY)]  # to store whole image?


def draw_autostereogram(Z):
    far = separation(0)

    for y in range(maxY):
        pix = []
        same = []
        s = 0
        left, right = 0, 0

        for x in range(maxX):
            same.append(x)
            pix.append(0)

        for x in range(maxX):
            s = separation(Z[y][x])
            left = x - s // 2
            right = left + s

            if 0 <= left and right < maxX:
                visible = 0
                t = 1
                zt = 0.0

                while 0 == 0:
                    zt = Z[y][x] + 2 * (2 - mu * Z[y][x]) * t / (mu*E)
                    visible = Z[y][x-t] < zt and Z[y][x+t] < zt
                    t += 1
                    if(visible and zt < 1):
                        pass
                    else:
                        break

                if visible:
                    l = same[left]

                    while l != left and l != right:
                        if l < right:
                            left = l
                            l = same[left]
                        else:
                            same[left] = right
                            left = right
                            l = same[left]
                            right = l

                    same[left] = right

        # Step 1: Create the pattern
        pattern_width = maxX // 8
        pattern = [[random.randint(0, 1) for _ in range(
            pattern_width)] for _ in range(maxY)]

        # Step 2: Modify the loop to copy values from the pattern
        for x in range(maxX-1, -1, -1):
            if same[x] == x:
                # Use the modulo operator to repeat the pattern across the image
                pix[x] = pattern[y][x % pattern_width]
            else:
                pix[x] = pix[same[x]]
            totalimagelist[y][x] = pix[x]

    numpyArray = np.asarray(totalimagelist, dtype=np.uint8)
    img = Image.fromarray(numpyArray * 255, 'L')
    # img.show()
    img.save("herhehere.png")


# Test with a depth map
#Z = np.random.rand(maxX, maxY)

Z = Z - np.min(Z)  # Ensure the minimum value in Z is 0
Z = Z / np.max(Z)
x = time.time()
draw_autostereogram(Z)
print(time.time() - x)