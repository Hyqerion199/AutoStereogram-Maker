from PIL import Image
import numpy as np
import random
import time
import glob

def generate_autostereogram(depth_map_path, output_path):
    # Load the depth map image and convert it to grayscale
    depth_map = Image.open(depth_map_path).convert('L')
    depth = np.array(depth_map)

    height, width = depth.shape
    output = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a random dot pattern
    for y in range(height):
        for x in range(width):
            output[y, x] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    # Generate the autostereogram
    for y in range(height):
        for x in range(width):
            if x - depth[y, x] >= 0:
                output[y, x] = output[y, x - depth[y, x]]

    # Convert the numpy array to an image
    img = Image.fromarray(output)

    # Save the image
    img.save(output_path)

for file in glob.glob('./depth/*.jpg'):
    start = time.time()
    x = file.split('\\')[-1]
    generate_autostereogram('depth_map.jpg', f"./final/{x}")
    print(file.split('\\')[-1].split('.')[0])
    end = time.time()
    delta = end - start
    print(f"took {delta} seconds to process")
    
