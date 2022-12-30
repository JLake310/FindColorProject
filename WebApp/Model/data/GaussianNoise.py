import cv2
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import time


def gaussian_noise(img, var):
    row, col, ch = img.shape
    gauss = np.random.normal(0, var, img.size)
    gauss = gauss.reshape(row, col, ch).astype('uint8')
    img_gauss = cv2.add(img, gauss)
    return img_gauss


def Add_noise_n_save(keyword, idx, open_path):
    save_path = f'Images/noise/{keyword}/img{idx}.jpg'
    img = cv2.imread(open_path)
    img = cv2.resize(img, (224, 224))
    var = 0.8
    img_gauss = gaussian_noise(img, var)
    cv2.imwrite(save_path, img_gauss)


def Open_image(keyword):
    os.makedirs(f"Images/noise/{keyword}", exist_ok=True)
    path_list = []
    dir_path = f"Images/{keyword}"
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            path_list.append(file_path)
    Parallel(n_jobs=-1, backend='threading')(delayed(Add_noise_n_save)(keyword, idx, open_path) \
                                             for idx, open_path in tqdm(enumerate(path_list), total=len(path_list)))


if __name__ == '__main__':
    start = time.time()
    keywords = ["military%20base", "small%20buildings", "war", "sports",
                "riots", "crowd", "city", "park", "korea%20military", "korea%20mountains"]

    for keyword in keywords:
        Open_image(keyword)

    print("time : ", time.time() - start)