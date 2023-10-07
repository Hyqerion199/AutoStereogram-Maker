from numba import jit
import numpy as np
from PIL import Image
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DPI = 72  # Output device has 72 pixels per inch
E = round(2.5 * DPI)  # Eye separation is assumed to be 2.5 in
mu = 1 / 3.0  # Depth of field (fraction of viewing distance)


@jit(nopython=True)
def separation(Z):
    factor = 2  # Increase this to increase the shift
    return round(factor * (1 - mu * Z) * E / (2 - mu * Z))


depth_map = Image.open("test.jpg")
depth_map_grayscale = depth_map.convert('I')
Z = np.array(depth_map_grayscale)
maxX, maxY = depth_map.size
Z = Z - np.min(Z)  # Ensure the minimum value in Z is 0
Z = Z / np.max(Z)

@jit(nopython=True)
def draw_autostereogram(Z):
    far = separation(0)
    pattern_width = maxX // 8
    pattern = np.random.randint(0, 2, (maxY, pattern_width))
    pix = np.zeros((maxY, maxX), dtype=np.uint8)
    same = np.zeros((maxY, maxX), dtype=np.int64)  # Changed np.int to np.int64
    s = np.zeros((maxY, maxX), dtype=np.int64)  # Changed np.int to np.int64
    left = np.zeros((maxY, maxX), dtype=np.int64)  # Changed np.int to np.int64
    # Changed np.int to np.int64
    right = np.zeros((maxY, maxX), dtype=np.int64)
    for y in range(maxY):
        same[y, :] = np.arange(maxX)
        for x in range(maxX):
            s[y, x] = separation(Z[y, x])
            left[y, x] = x - s[y, x] // 2
            right[y, x] = left[y, x] + s[y, x]
            if 0 <= left[y, x] and right[y, x] < maxX:
                t = 1
                zt = Z[y, x] + 2 * (2 - mu * Z[y, x]) * t / (mu*E)
                visible = Z[y, x-t] < zt and Z[y, x+t] < zt
                t += 1
                while visible and zt < 1:
                    zt = Z[y, x] + 2 * (2 - mu * Z[y, x]) * t / (mu*E)
                    visible = Z[y, x-t] < zt and Z[y, x+t] < zt
                    t += 1
                if visible:
                    l = same[y, left[y, x]]
                    while l != left[y, x] and l != right[y, x]:
                        if l < right[y, x]:
                            left[y, x] = l
                            l = same[y, left[y, x]]
                        else:
                            same[y, left[y, x]] = right[y, x]
                            left[y, x] = right[y, x]
                            l = same[y, left[y, x]]
                            right[y, x] = l
                    same[y, left[y, x]] = right[y, x]
        for x in range(maxX-1, -1, -1):
            if same[y, x] == x:
                pix[y, x] = pattern[y, x % pattern_width]
            else:
                pix[y, x] = pix[y, same[y, x]]
    return pix


x = time.time()
pix = draw_autostereogram(Z)
img = Image.fromarray(pix * 255, 'L')
img.save("herhehere.png")
print(time.time() - x)
