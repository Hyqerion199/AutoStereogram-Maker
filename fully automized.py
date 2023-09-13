import os
from random import randint
import glob
import time
import subprocess
import cv2
import torch
import numpy as np
import urllib.request
from PIL import Image, ImageOps
import torchvision.transforms as transforms


os.chdir(os.path.dirname(os.path.abspath(__file__)))

if not os.path.exists("./depth"):
    os.makedirs("./depth")
if not os.path.exists("./rgb"):
    os.makedirs("./rgb")
if not os.path.exists("./averaged"):
    os.makedirs("./averaged")
if not os.path.exists("./final"):
    os.makedirs("./final")
if not os.path.exists("./merged"):
    os.makedirs("./merged")
if not os.path.exists("./merged2"):
    os.makedirs("./merged2")
if not os.path.exists("./merged2"):
    os.makedirs("./merged2")

original_file = input("video file name relative to the  python file directory >> ")

subprocess.call(['ffmpeg', '-i', original_file, '-qmin', '1', '-qscale:v', '1', './rgb/%06d.jpg'])
fps_of_vid = input("what is the fps of the video file >> ")


#--------------------------------------------------------- depth.py

use_large_model = True

if use_large_model:
	midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
else:
	midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
print(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

if use_large_model:
	transform = midas_transforms.dpt_transform
	print('Using large (slow) model.')
else:
	transform = midas_transforms.small_transform
	print('Using small (fast) model.')


for file in glob.glob('./rgb/*.jpg'):

	img = cv2.imread(file)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	input_batch = transform(img).to(device)
	
	with torch.no_grad():
		prediction = midas(input_batch)
	
		prediction = torch.nn.functional.interpolate(
			prediction.unsqueeze(1),
			size = img.shape[:2],
			mode = 'bicubic',
			align_corners = False,
		).squeeze()
	
	output = prediction.cpu().numpy()
	
	output_normalized = (output * 255 / np.max(output)).astype('uint8')
	output_image = Image.fromarray(output_normalized)
	output_image_converted = output_image.convert('RGB').save(file.replace('rgb', 'depth'))
	print('Converted: ' + file)
print('Done.')

#------------------------------------------------------------------------ averaged.py

items = len(glob.glob('./depth/*.jpg')) - 2
first = './depth/000001.jpg'
last = './depth/' + str(items + 2).zfill(6) + '.jpg'
w, h = Image.open(first).size
Image.open(first).save(first.replace('depth', 'averaged'))

for idx in range(items):
	current = idx + 2
	arr = np.zeros((h, w, 3), np.float64)
	
	prev = np.array(Image.open('./depth/' + str(current - 1).zfill(6) + '.jpg'), dtype = np.float64)
	curr = np.array(Image.open('./depth/' + str(current).zfill(6) + '.jpg'), dtype = np.float64)
	next = np.array(Image.open('./depth/' + str(current + 1).zfill(6) + '.jpg'), dtype = np.float64)
	
	arr = arr+prev/3
	arr = arr+curr/3
	arr = arr+next/3
	
	arr = np.array(np.round(arr), dtype = np.uint8)
	
	out = Image.fromarray(arr,mode = 'RGB')
	out.save('./averaged/' + str(current).zfill(6) + '.jpg')
	print('Averaged: ' + str(current).zfill(6) + '.jpg')

Image.open(last).save(last.replace('depth', 'averaged'))
print('Done.')

#------------------------------------------------------------------------ merge.py

def get_concat_v(im1, im2):
	dst = Image.new('RGB', (im1.width, im1.height + im2.height))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (0, im1.height))
	return dst

for file in glob.glob("./rgb/*.jpg"):
	im1 = Image.open(file)
	try:
		im2 = Image.open(file.replace('rgb', 'averaged'))
	except:
		im2 = Image.open(file.replace('rgb', 'depth'))
	get_concat_v(im1, im2).save(file.replace('rgb', 'merged'))
	im2 = Image.open(file.replace('rgb', 'depth'))
	get_concat_v(im1, im2).save(file.replace('rgb', 'merged2'))
	print("Merged: " + file)
for file in glob.glob("./depth/*.jpg"):
	im1 = Image.open(file)
	im2 = Image.open(file.replace('depth', 'averaged'))
	get_concat_v(im1, im2).save(file.replace('depth', 'merged3'))
	print("Merged: " + file)
print('Done.')

#------------------------------------------------------------------------ stereogram.py


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

#------------------------------------------------------------------------



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


subprocess.call(['ffmpeg', '-framerate', fps_of_vid, '-i', './final/%06d.jpg', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'{original_file}_depth.mp4'])

subprocess.call(['ffmpeg', '-i', f'{original_file}_depth.mp4', '-i', f'{original_file}.mp4', '-c', 'copy', '-map', '0:0', '-map', '1:1', '-shortest', f'{original_file}_depth_sound.mpr'])