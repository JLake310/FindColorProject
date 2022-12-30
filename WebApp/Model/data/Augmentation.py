import cv2
import random
import numpy as np
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import os


def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def horizontal_flip(img):
    return cv2.flip(img, 1)


def vertical_flip(img):
    return cv2.flip(img, 0)


def Augment_n_save(keyword, idx, open_path, aug):
    save_path = f'Images/augment/{keyword}/img{idx}.jpg'
    img = cv2.imread(open_path)
    img = cv2.resize(img, (384, 384))
    if aug == "brightness":
        img_aug = brightness(img, 0.5, 3)
    elif aug == "h_flip":
        img_aug = horizontal_flip(img)
    elif aug == "v_flip":
        img_aug = vertical_flip(img)
    save_path = f'Images/augment/{aug}/{keyword}/img{idx}.jpg'
    cv2.imwrite(save_path, img_aug)


def Open_image(keyword, aug):
    path_list = []
    dir_path = f"Images/{keyword}"
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            path_list.append(file_path)
    os.makedirs(f"Images/augment/{aug}/{keyword}", exist_ok=True)
    Parallel(n_jobs=-1, backend='threading')(delayed(Augment_n_save)(keyword, idx, open_path, aug) \
                                             for idx, open_path in tqdm(enumerate(path_list), total=len(path_list)))


if __name__ == '__main__':
    start = time.time()
    img_idx = 5
    keywords = ["military%20base", "small%20buildings", "war", "sports",
                "riots", "crowd", "city", "park", "korea%20military", "korea%20mountains"]

    for keyword in keywords:
        Open_image(keyword, "v_flip")

    print("time : ", time.time() - start)