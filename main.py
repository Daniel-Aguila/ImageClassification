from bing_image_downloader import downloader
from six import BytesIO
from PIL import Image, ImageOps
import numpy as np
import glob
import os
import pathlib
import time
import matplotlib
import matplotlib.pyplot as plt
from concurrent import futures


#downloader.download("dogs", limit=50, output_dir='images', adult_filter_off=True, force_replace=False)
dog_images_np = []
def preprocess(path):
    img_data = open(path,'rb').read()
    image = Image.open(BytesIO(img_data))
    image = image.resize((2048,1362))
    image = ImageOps.grayscale(image)
    image.save(f"{path}","JPEG")
    #return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
def changeShape(image):
    dog_images_np.append(preprocess(image))
    print(f'{image} was processed...')

def main():
    start = time.perf_counter()
    dog_image_path = "images\dogs"
    images = pathlib.Path(dog_image_path).iterdir()

#Multiprocessing

    with futures.ProcessPoolExecutor() as executor:
        results = executor.map(changeShape, images)
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds(s)')

if __name__ == '__main__':
    main()