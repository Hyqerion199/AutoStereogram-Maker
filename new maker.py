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
from numba import jit


DPI = 72  # Output device has 72 pixels per inch
E = round(2.5 * DPI)  # Eye separation is assumed to be 2.5 in
mu = 1 / 3.0  # Depth of field (fraction of viewing distance)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    directories = ["./rgb", "./averaged", "./final",
                   "./merge_normal", "./merge_average", "./merge_averageandnormal"]
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


@jit(nopython=True)
def separation(Z):
    factor = 2  # Increase this to increase the shift
    return round(factor * (1 - mu * Z) * E / (2 - mu * Z))


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
    print(torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

    if use_large_model:
        transform = midas_transforms.dpt_transform
        print('Using large (slow) model.')
    else:
        transform = midas_transforms.small_transform
        print('Using small (fast) model.')

    for file in glob.glob('./rgb/*.png'):

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        output_normalized = (output * 255 / np.max(output)).astype('uint8')
        output_image = Image.fromarray(output_normalized)
        output_image_converted = output_image.convert(
            'RGB').save(file.replace('rgb', 'depth'))
        print('Converted: ' + file)
    print('Done.')

#------------------------------------------------------------------------ averaged.py


def depth_map_averaged_maker():
    items = len(glob.glob('./depth/*.png')) - 2
    first = './depth/000001.png'
    last = './depth/' + str(items + 2).zfill(6) + '.png'
    w, h = Image.open(first).size
    Image.open(first).save(first.replace('depth', 'averaged'))

    for idx in range(items):
        current = idx + 2
        arr = np.zeros((h, w, 3), np.float64)

        prev = np.array(Image.open('./depth/' + str(current -
                        1).zfill(6) + '.png'), dtype=np.float64)
        curr = np.array(Image.open(
            './depth/' + str(current).zfill(6) + '.png'), dtype=np.float64)
        next = np.array(Image.open('./depth/' + str(current +
                        1).zfill(6) + '.png'), dtype=np.float64)

        arr = arr+prev/3
        arr = arr+curr/3
        arr = arr+next/3

        arr = np.array(np.round(arr), dtype=np.uint8)

        out = Image.fromarray(arr, mode='RGB')
        out.save('./averaged/' + str(current).zfill(6) + '.png')
        print('Averaged: ' + str(current).zfill(6) + '.png')
    Image.open(last).save(last.replace('depth', 'averaged'))
    print('Done.')

#------------------------------------------------------------------------ merge.py


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def merge_images_together():

    for file in glob.glob("./rgb/*.png"):
        im1 = Image.open(file)
        if dotheaveraged == 'y':
            im2 = Image.open(file.replace('rgb', 'averaged'))
            get_concat_v(im1, im2).save(file.replace('rgb', 'merge_average'))
        im2 = Image.open(file.replace('rgb', 'depth'))
        get_concat_v(im1, im2).save(file.replace('rgb', 'merge_normal'))
        print("Merged: " + file)
    if compare_average_and_depth == 'y':
        for file in glob.glob("./depth/*.png"):
            im1 = Image.open(file)
            im2 = Image.open(file.replace('depth', 'averaged'))
            get_concat_v(im1, im2).save(file.replace(
                'depth', 'merge_averageandnormal'))
            print("Merged: " + file)
    print('Done.')
    if make_videos_of_these_stuff == 'y':
        subprocess.call(['ffmpeg', '-framerate', fps_of_vid, '-i', './merge_normal/%06d.png',
                        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'normal_merged.mp4'])
        if dotheaveraged == 'y':
            subprocess.call(['ffmpeg', '-framerate', fps_of_vid, '-i', './merge_average/%06d.png',
                            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'averaged_merged.mp4'])
        if compare_average_and_depth == 'y':
            subprocess.call(['ffmpeg', '-framerate', fps_of_vid, '-i', './merge_averageandnormal/%06d.png',
                            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'average_and_normal_merged.mp4'])


#------------------------------------------------------------------------ stereogram.py

def gen_autostereogram(depth_map):
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
        # Changed np.int to np.int64
        same = np.zeros((maxY, maxX), dtype=np.int64)
        # Changed np.int to np.int64
        s = np.zeros((maxY, maxX), dtype=np.int64)
        # Changed np.int to np.int64
        left = np.zeros((maxY, maxX), dtype=np.int64)
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


    pix = draw_autostereogram(Z)
    return pix



xxxx = []


def do_the_stereogram(start, end, aordf):
    for file in glob.glob(f"./{aordf}/*.png")[start:end]:
        x = file.split("\\")[-1]
        print(x)
        depth_map = file
        outfile = f"./final/{x}"
        autostereogram = gen_autostereogram(Image.open(depth_map))
        img = Image.fromarray(autostereogram * 255, 'L')
        img.save(outfile)
        xxxx.append(file)


# ------------------------------------------------------------------------

if __name__ == "__main__":
    all_at_once = input(
        "Do you want to answer all of the questions at once (y) or do it after each segment (n)? (y/n) ")
    if all_at_once == 'y':
        has_depth_map_done = input("Do you have the depth map done? (y/n) ")
        if has_depth_map_done == "n":
            shutil.rmtree("depth")
            os.makedirs("depth")
        do_averaged = input(
            "Do you want to do the averaged depth images? (y/n) ")
        make_videos_from_depth_maps = input(
            "Do you want to make videos from the depth maps merged with the original frames? (Recommended if you did the averaged depth maps) (y/n) ")
        if make_videos_from_depth_maps == 'y':
            dotheaveraged = do_averaged
            # dotheaveraged = input("Did you do the averaged depth images? (y/n) ")
            compare_average_and_depth = input(
                "Do you want to compare the average and depth images side by side? (y/n) ")
            make_videos_of_these_stuff = input(
                "Do you want to make videos of the merged images(all of them)? (y/n) ")

    original_file = input(
        "Video file name relative to the python file directory. Format is ./filename.mp4.   ")
    if has_depth_map_done == "n":
        subprocess.call(['ffmpeg', '-i', original_file, '-qmin',
                        '1', '-qscale:v', '1', './rgb/%06d.png'])

    if all_at_once == 'n':
        has_depth_map_done = input("Do you have the depth map done? (y/n) ")
    if has_depth_map_done == 'n':
        depth_map_do()
    if all_at_once == 'n':
        do_averaged = input(
            "Do you want to do the averaged depth images? (y/n) ")
    if do_averaged == 'y':
        depth_map_averaged_maker()
    if all_at_once == 'n':
        make_videos_from_depth_maps = input(
            "Do you want to make videos from the depth maps merged with the original frames? (Recommended if you did the averaged depth maps) (y/n) ")
    if make_videos_from_depth_maps == 'y':
        merge_images_together()
        input("Here you can analyze the videos made .")
    if do_averaged == 'y':
        averageordepth = input(
            "Do you want to use the averaged images or depth images? (a/d) ")
        if averageordepth == 'a':
            aord = 'average'
        if averageordepth == 'd':
            aord = 'depth'
    else:
        aord = 'depth'

    input("Press enter to do the stereogram. ")
    amount_per = len(glob.glob("./depth/*.png"))
    amount_per = np.linspace(0, amount_per, 11).astype(int)
    amount_per[0] = amount_per[0]
    print(amount_per)

    # creating thread
    tt1 = time.time()
    threads = []

    for x in range(10):
        t = multiprocessing.Process(target=do_the_stereogram, args=(
            amount_per[x], amount_per[x+1], aord,))
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

    subprocess.call(['ffmpeg', '-framerate', fps_of_vid, '-i', './final/%06d.png',
                    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'{original_file}_depth.mp4'])

    subprocess.call(['ffmpeg', '-i', f'{original_file}_depth.mp4', '-i', f'{original_file}', '-c',
                    'copy', '-map', '0:0', '-map', '1:1', '-shortest', f'{original_file}_depth_sound.mp4'])

    subprocess.call(['ffmpeg', '-i', f'{original_file}_depth_sound.mp4',
                    '-vcodec', 'libx265', '-crf', '28', f'{original_file}_final.mp4',])
