import os
import time

import requests
from bs4 import BeautifulSoup
import numpy as np
import arxiv


def get_ids(ids, soup):
	spans = soup.find_all("span", class_="list-identifier")
	for span in spans:
		links = span.find_all('a', href=True)
		idx = links[0]['href']
		ids.append(idx.split('/')[-1])


def extract_abstract_api(ids):
	start = time.time()
	for n, entry in enumerate(ids):
		try:
			doc = arxiv.query(id_list=[str(entry)])[0]
			summary = doc.get('summary', None)
			file_name = "science_corpus/" + entry.replace(".", "") + ".txt"
			save_file(summary, file_name)
			if (n + 1) % 100 == 0:
				print('finished parsing {} documents'.format(n + 1))
				print('{}m elapsed'.format(np.round((time.time()-start)/60), 2))
		except:
			print('failed retrieving data from {}'.format(entry))
		time.sleep(.3)


def save_file(text, path):
	with open(path, 'a') as ofile:
		ofile.write(text)


def gather_papers():
	# all of 2017
	linklist = [
				#'https://arxiv.org/list/physics/1701?show=2000',
				#'https://arxiv.org/list/cs/1701?show=2000',
				#'https://arxiv.org/list/cond-mat.mtrl-sci/1701?show=2000',
				#'https://arxiv.org/list/physics/1702?show=2000',
				#'https://arxiv.org/list/cs/1702?show=2000',
				#'https://arxiv.org/list/cond-mat.mtrl-sci/1702?show=2000',
				#'https://arxiv.org/list/physics/1703?show=2000',
				#'https://arxiv.org/list/cs/1703?show=2000',
				#'https://arxiv.org/list/cond-mat.mtrl-sci/1703?show=2000',
				#'https://arxiv.org/list/physics/1704?show=2000',
				# 'https://arxiv.org/list/cs/1704?show=2000',
				# 'https://arxiv.org/list/cond-mat.mtrl-sci/1704?show=2000',
				# 'https://arxiv.org/list/physics/1707?show=2000',
				# 'https://arxiv.org/list/cs/1707?show=2000',
				# 'https://arxiv.org/list/cond-mat.mtrl-sci/1707?show=2000',
				# 'https://arxiv.org/list/physics/1708?show=2000',
				# 'https://arxiv.org/list/cs/1708?show=2000',
				# 'https://arxiv.org/list/cond-mat.mtrl-sci/1708?show=2000',
				# 'https://arxiv.org/list/physics/1709?show=2000',
				# 'https://arxiv.org/list/cs/1709?show=2000',
				# 'https://arxiv.org/list/cond-mat.mtrl-sci/1709?show=2000',
				# 'https://arxiv.org/list/physics/1710?show=2000',
				# 'https://arxiv.org/list/cs/1710?show=2000',
				# 'https://arxiv.org/list/cond-mat.mtrl-sci/1710?show=2000',
				'https://arxiv.org/list/physics/1711?show=2000',
				'https://arxiv.org/list/cs/1711?show=2000',
				'https://arxiv.org/list/cond-mat.mtrl-sci/1711?show=2000',
				'https://arxiv.org/list/physics/1712?show=2000',
				'https://arxiv.org/list/cs/1712?show=2000',
				'https://arxiv.org/list/cond-mat.mtrl-sci/1712?show=2000',
				]

	for link in linklist:
		print('starting scraping files from: {}'.format(link))
		page = requests.get(link)
		soup = BeautifulSoup(page.content, 'html.parser')
		ids = []
		get_ids(ids, soup)
		print('number of papers found: {}'.format(len(ids)))
		extract_abstract_api(ids)
		time.sleep(30)

if __name__ == '__main__':
	gather_papers()
