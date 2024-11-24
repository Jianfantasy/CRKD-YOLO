from PIL import Image
import numpy as np
import os
from tqdm import tqdm
def add_gaussian_noise(image_path, output_path, sigma=0.1):

    img = Image.open(image_path)
    img_array = np.array(img)


    noise = np.random.normal(0, sigma, img_array.shape)
    noisy_img_array = img_array + noise


    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)


    noisy_img = Image.fromarray(noisy_img_array)


    noisy_img.save(os.path.join(output_path, os.path.basename(image_path)))

def process_images(input_dir, output_dir, sigma=0.1):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img in tqdm(filenames,desc= "adding noise"):
        image_path = os.path.join(input_dir, img)
        add_gaussian_noise(image_path, output_dir, sigma)

input_dir = 'FPS_TEST'
output_dir = 'noise_img_50'
sigma = 50

process_images(input_dir, output_dir, sigma)