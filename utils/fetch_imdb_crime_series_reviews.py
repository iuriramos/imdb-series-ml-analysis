import os
import re
import codecs
import string
import random
import requests
# import logging
from imdbpie import Imdb
from bs4 import BeautifulSoup

BASE_DIRECTORY_PATH = './reviews'

def create_dir_structure(directory_name, rating_range):
    directory_path = os.path.abspath(
        os.path.join(BASE_DIRECTORY_PATH, directory_name)
    )

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    for rating in rating_range:
        subdirectory_path = os.path.join(directory_path, str(rating))
        if not os.path.exists(subdirectory_path):
            os.mkdir(subdirectory_path)

    return directory_path

def download_imdb_review(directory_path, review):
    # helper inner class
    def random_word(length):
        return ''.join(random.choice(string.lowercase) for _ in range(length))

    rating_directory_path = os.path.join(directory_path, str(review.rating))
    if review.username is None:
        review.username = random_word(random.randint(5, 10))
    file_path = os.path.join(rating_directory_path, review.username)
    if not os.path.exists(file_path):
        with codecs.open(file_path, mode='w', encoding='utf-8') as f:
            f.write(review.text)

def download_imdb_reviews(rating_range, directory_path, imdb_id, max_results=1000):
    imdb = Imdb()
    for review in imdb.get_title_reviews(imdb_id, max_results=max_results):
        if review.rating in rating_range:
            download_imdb_review(directory_path, review)

def download_imdb_crime_series(top, rating_range):
    if not os.path.exists(BASE_DIRECTORY_PATH):
        os.mkdir(BASE_DIRECTORY_PATH)

    base_url = 'http://www.imdb.com/search/title'
    payload = dict(
        genres='crime',
        title_type=['tv_series', 'mini_series',],
        page=1
    )

    n_series = 0
    while n_series < top:
        r = requests.get(base_url, params=payload)
        html_text = r.text

        soup = BeautifulSoup(html_text, 'html.parser')
        '''
        <h3 class="lister-item-header">
            <span class="lister-item-index unbold text-primary">1.</span>
            <a href="/title/tt2193021/?ref_=adv_li_tt">Arrow</a>
            <span class="lister-item-year text-muted unbold">(2012 - )</span>
        </h3>
        '''
        titles = soup.find_all('h3', class_='lister-item-header')
        imdb_regex = re.compile(ur'/title/(\w+)/')

        for title in titles:
            if n_series >= top:
                break

            link = title.find('a')
            title_name = link.get_text()
            href = link['href']

            #get imdb id, first item of matching object (ex.: tt2193021)
            mo = imdb_regex.search(href)
            imdb_id = mo.group(1)

            directory_path = create_dir_structure(directory_name=title_name, rating_range=rating_range)
            download_imdb_reviews(rating_range, directory_path=directory_path, imdb_id=imdb_id)

            # increment series counter
            n_series += 1

        # go to the next page
        payload['page'] += 1

def fetch_review_text(file_path):
    with codecs.open(file_path, mode='r', encoding='utf-8') as f:
        return f.read()

def fetch_imdb_crime_series_reviews(top=25,neg_rating_range=None, pos_rating_range=None):
    neg_rating_range = neg_rating_range or (1, 2, 3, 4, 5)
    pos_rating_range = pos_rating_range or (6, 7, 8, 9, 10)

    rating_range = list(neg_rating_range + pos_rating_range)
    download_imdb_crime_series(top, rating_range)

    X, y = [], []
    regex = re.compile(ur'/(\d{1,2})$')

    def classify_review_rating(rating):
        if rating in neg_rating_range:
            return 0
        if rating in pos_rating_range:
            return 1
        return -1

    for root, _, files in os.walk(BASE_DIRECTORY_PATH):
        rating_directory = regex.search(root)
        if rating_directory:
            rating = int(rating_directory.group(1))
            if rating in rating_range:
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    X.append(fetch_review_text(file_path))
                    y.append(classify_review_rating(rating))

    return X, y

def run():
    X, y = fetch_imdb_crime_series_reviews()

if __name__ == '__main__':
    run()
