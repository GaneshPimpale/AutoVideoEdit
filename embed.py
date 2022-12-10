from sentence_transformers import SentenceTransformer
import re
import numpy as np
import requests
import bs4

URL = "https://en.wikipedia.org/wiki/SpaceX"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def cosSim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def scrape(url):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    string = soup.body.text
    string =  re.sub(r'\n', ' ', string)
    return re.split(r'[.!?]', string)

scraped = scrape(URL)
embeddings = model.encode(scraped)