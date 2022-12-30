import os
import urllib.request
from joblib import Parallel, delayed
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from PIL import Image


def open_image(keyword, idx, url_):
    try:
        img = Image.open(requests.get(url_, stream=True).raw)
        img.save(f'Images/{keyword}/img{idx}.jpg')
    except:
        pass


def scrape(keyword):
    os.makedirs(f"Images/{keyword}", exist_ok=True)

    url_list = []

    # gettyImage의 전체 page crawling (최대 100page 까지 있음)
    for page in tqdm(range(1, 101)):
        try:
            url = f'https://www.gettyimages.com/search/2/image?family=creative&phrase={keyword}&page={page}'
            html = urllib.request.urlopen(url)
            soup = BeautifulSoup(html, 'html.parser')
            get_urls = soup.select("source")

            # url_list : 전체 page의 image url
            for i in range(len(get_urls)):
                url_list.append(str(get_urls[i]).split('"')[1].replace("s=612x612", ""))
        except:
            break

    # keyword에 대한 image 저장 (multithreading)
    Parallel(n_jobs=-1, backend='threading')(delayed(open_image)(keyword, idx, url_) \
                                             for idx, url_ in tqdm(enumerate(url_list), total=len(url_list)))


if __name__ == '__main__':
    keywords = ["military%20base", "small%20buildings", "war", "sports",
                "riots", "crowd", "city", "park", "korea%20military", "korea%20mountains"]
    for keyword in keywords:
        print("Crawling " + keyword)
        scrape(keyword)