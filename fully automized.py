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
import multiprocessing
import shutil


os.chdir(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    directories = ["./rgb", "./averaged", "./final", "./merge_normal", "./merge_average", "./merge_averageandnormal"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)



if __name__ == "__main__":
    fps_of_vid = input("what is the fps of the video file. ")

dotheaveraged = ''
make_videos_of_these_stuff = ''
compare_average_and_depth = ''
#--------------------------------------------------------- depth.py



def depth_map_do():
    while True:
        large_model_or_not = input('Would you like the use the larger model of training for the depth map? The Larger model will result in better results but requires more gpu power and more space on your computer. The smaller model will result in lower quality results but is faster and takes up less space (y/n) ')
        if large_model_or_not == 'y':  
            use_large_model = True
            break
        elif large_model_or_not == 'n':
            use_large_model = False
            break
        else:
            print('Please enter y or n.')

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

def depth_map_averaged_maker():
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

def merge_images_together():


    for file in glob.glob("./rgb/*.jpg"):
        im1 = Image.open(file)
        if dotheaveraged == 'y':
            im2 = Image.open(file.replace('rgb', 'averaged'))
            get_concat_v(im1, im2).save(file.replace('rgb', 'merge_average'))
        im2 = Image.open(file.replace('rgb', 'depth'))
        get_concat_v(im1, im2).save(file.replace('rgb', 'merge_normal'))
        print("Merged: " + file)
    if compare_average_and_depth == 'y':
        for file in glob.glob("./depth/*.jpg"):
            im1 = Image.open(file)
            im2 = Image.open(file.replace('depth', 'averaged'))
            get_concat_v(im1, im2).save(file.replace('depth', 'merge_averageandnormal'))
            print("Merged: " + file)
    print('Done.')
    if make_videos_of_these_stuff == 'y':
        subprocess.call(['ffmpeg', '-framerate', fps_of_vid, '-i', './merge_normal/%06d.jpg', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'normal_merged.mp4'])
        if dotheaveraged == 'y':
            subprocess.call(['ffmpeg', '-framerate', fps_of_vid, '-i', './merge_average/%06d.jpg', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'averaged_merged.mp4'])
        if compare_average_and_depth == 'y':
            subprocess.call(['ffmpeg', '-framerate', fps_of_vid, '-i', './merge_averageandnormal/%06d.jpg', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'average_and_normal_merged.mp4'])


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
xxxx = []
def do_the_stereogram(start, end, aordf):
    for file in glob.glob(f"./{aordf}/*.jpg")[start:end]:
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
        xxxx.append(file)
        
    

#------------------------------------------------------------------------


if __name__ == "__main__":
    all_at_once = input("Do you want to answer all of the questions at once (y) or do it after each segment (n)? (y/n) ")
    if all_at_once == 'y':
        has_depth_map_done = input("Do you have the depth map done? (y/n) ")
        do_averaged = input("Do you want to do the averaged depth images? (y/n) ")
        make_videos_from_depth_maps = input("Do you want to make videos from the depth maps merged with the original frames? (Recommended if you did the averaged depth maps) (y/n) ")
        if make_videos_from_depth_maps == 'y':
            dotheaveraged = do_averaged
            # dotheaveraged = input("Did you do the averaged depth images? (y/n) ")
            compare_average_and_depth = input("Do you want to compare the average and depth images side by side? (y/n) ")
            make_videos_of_these_stuff = input("Do you want to make videos of the merged images(all of them)? (y/n) ")

    # all_at_once = 'y'
    # has_depth_map_done = 'n'
    # do_averaged = 'n'
    # make_videos_from_depth_maps = 'n'
    # dotheaveraged = do_averaged
    # compare_average_and_depth = 'n'
    # make_videos_of_these_stuff = 'n'




    original_file = input("Video file name relative to the python file directory. Format is ./filename.mp4.   ")
    subprocess.call(['ffmpeg', '-i', original_file, '-qmin', '1', '-qscale:v', '1', './rgb/%06d.jpg'])

    if all_at_once == 'n':
        has_depth_map_done = input("Do you have the depth map done? (y/n) ")
    if has_depth_map_done == 'n':
        depth_map_do()
    if all_at_once == 'n':
        do_averaged = input("Do you want to do the averaged depth images? (y/n) ")
    if do_averaged == 'y':
        depth_map_averaged_maker()
    if all_at_once == 'n':
        make_videos_from_depth_maps = input("Do you want to make videos from the depth maps merged with the original frames? (Recommended if you did the averaged depth maps) (y/n) ")
    if make_videos_from_depth_maps == 'y':
        merge_images_together()
        input("Here you can analyze the videos made .")
    if do_averaged == 'y':
        averageordepth = input("Do you want to use the averaged images or depth images? (a/d) ")
        if averageordepth == 'a':
            aord = 'average'
        if averageordepth == 'd':
            aord = 'depth'
    else:
        aord = 'depth'


    input("Press enter to do the stereogram. ")
    amount_per = len(glob.glob("./depth/*.jpg"))
    amount_per = np.linspace(0, amount_per, 11).astype(int)
    amount_per[0] = amount_per[0]
    print(amount_per)

    # creating thread
    tt1 = time.time()
    threads = []

    for x in range(10):
        t = multiprocessing.Process(target=do_the_stereogram, args=(amount_per[x], amount_per[x+1], aord,))
        t.daemon = True
        threads.append(t)

    for x in range(10):
        threads[x].start()

    for x in range(10):
        threads[x].join()
    tt1tot = time.time()

    
    # all threads are completely executed
    print(tt1tot-tt1)
 
    print("Done!")
    
    subprocess.call(['ffmpeg', '-framerate', fps_of_vid, '-i', './final/%06d.jpg', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'{original_file}_depth.mp4'])

    subprocess.call(['ffmpeg', '-i', f'{original_file}_depth.mp4', '-i', f'{original_file}', '-c', 'copy', '-map', '0:0', '-map', '1:1', '-shortest', f'{original_file}_depth_sound.mp4'])

    subprocess.call(['ffmpeg', '-i', f'{original_file}_depth_sound.mp4', '-vcodec', 'libx265', '-crf', '28', f'{original_file}_final.mp4',])