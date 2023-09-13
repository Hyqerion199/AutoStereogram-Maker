import os
from PIL import Image
from random import randint
import glob
import time

def gen_random_dot_strip(width, height):
    """
    Given a target width and height in pixels, return an RGB Image of those
    dimensions consisting of random dots.

    This strip will be repeated as the background for the autostereogram.
    """
    strip = Image.new("RGB", (width, height))
    pix = strip.load()
    for x in range(width):
        for y in range(height):
            r = randint(0, 256)
            g = randint(0, 256)
            b = randint(0, 256)
            pix[x,y] = (r, g, b)

    return strip


def gen_strip_from_tile(tile, width, height):
    """
    Given an open tile Image, return an Image of the specified width and height,
    repeating tile as necessary to fill the image.

    This strip will be repeated as the background for the autostereogram.
    """
    tile_pixels = tile.load()
    tile_width, tile_height = tile.size

    strip = Image.new("RGB", (width, height))
    pix = strip.load()
    for x in range(width):
        for y in range(height):
            x_offset = x % tile_width
            y_offset = y % tile_height
            pix[x,y] = tile_pixels[x_offset,y_offset]

    return strip


def gen_autostereogram(depth_map, tile=None):
    """
    Given a depth map, return an autostereogram Image computed from that depth
    map.
    """
    depth_map_width, height = depth_map.size

    # If we have a tile, we want the strip width to be a multiple of the tile
    # width so it repeats cleanly.
    if tile:
        tile_width = tile.size[0]
        strip_width = tile_width
    else:
        strip_width = depth_map_width // 8

    num_strips = depth_map_width / strip_width
    image = Image.new("RGB", (depth_map_width, height))

    if tile:
        background_strip = gen_strip_from_tile(tile, strip_width, height)
    else:
        background_strip = gen_random_dot_strip(strip_width, height)

    strip_pixels = background_strip.load()

    depth_map = depth_map.convert('I')
    depth_pixels = depth_map.load()
    image_pixels = image.load()

    for x in range(depth_map_width):
        for y in range(height):
            # Need one full strip's worth to borrow from.
            if x < strip_width:
                image_pixels[x, y] = strip_pixels[x, y]
            else:
                depth_offset = depth_pixels[x, y] / num_strips
                image_pixels[x, y] = image_pixels[x - strip_width + depth_offset, y]

    return image

for file in glob.glob("./depth/*.jpg"):
    y = time.time()
    x = file.split("\\")[-1]
    print(x)
    depth_map= file
    outfile=f"./final/{x}"
    tile=None
    if tile:
        autostereogram = gen_autostereogram(Image.open(depth_map),
                                            tile=Image.open(tile))
    else:
        autostereogram = gen_autostereogram(Image.open(depth_map))
    autostereogram.save(outfile)
    
    print(time.time()-y)