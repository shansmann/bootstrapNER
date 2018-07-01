import requests
from bs4 import BeautifulSoup
import os
from nltk.tokenize import word_tokenize
import time

url_ps = 'https://arxiv.org/list/physics/1706?show=2000'
url_cs = 'https://arxiv.org/list/cs/1706?show=2000'
url_ms = 'https://arxiv.org/list/cond-mat.mtrl-sci/1706?show=2000'

pdf_link = "https://arxiv.org"
page = requests.get(url_ms)

soup = BeautifulSoup(page.content, 'html.parser')

ids = []

def get_ids():
	spans = soup.find_all("span", class_="list-identifier")
	for span in spans:
		links = span.find_all('a', href=True)
		#print("Found the URL:", links[1]['href'])
		ids.append(links[0]['href'])

def get_text(ids):
	tokens = 0
	for n, idx in enumerate(ids):
		url = pdf_link + idx
		response = requests.get(url)
		text = extract_abstract(response)
		if text:
			file_name = "science_corpus/" + idx.split('/')[-1].replace(".", "") + ".txt"
			save_file(text, file_name)
			tokens += len(word_tokenize(text))
		else:
			print('error parsing abstract.')
		if (n + 1) % 100 == 0:
			print("finished {} paper".format(n + 1))
			print('collected tokens: {}'.format(tokens))
		time.sleep(1.1)
	print('number of tokens in corpus: {}'.format(tokens))

def extract_abstract(response):
	soup = BeautifulSoup(response.content, 'html.parser')
	abstracts = soup.find_all("span", class_="descriptor")
	try:
		abstract = abstracts[-1]
		text = abstract.next_sibling.strip()
	except:
		text = None
	return text

def save_file(text, path):
	with open(path, 'a') as ofile:
			ofile.write(text)

def delete_file(path):
	os.remove(path)


get_ids()
print(len(ids))
get_text(ids)

