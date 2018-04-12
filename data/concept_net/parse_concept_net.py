import csv
from collections import defaultdict
from heapq import nlargest
from operator import itemgetter
import numpy as np
import json

raw_path = "data-concept-instance-relations-with-blc.tsv"
concept_net_path = "concept_net.json"

def parse_concept_net(fname):
	db = defaultdict(list)
	with open(fname) as fd:
		rd = csv.reader(fd, delimiter="\t")
		for n, row in enumerate(rd):
			if n % 100000 == 0:
				print("finished %:", np.round(n/33377000, 2)*100)
			db[row[1]].append((row[0], row[5]))
		return db

def save_concept_net(fname):
	with open(fname, 'w') as fp:
		json.dump(data, fp)

def load_concept_net(fname):
	with open(fname, 'r') as fp:
		data = json.load(fp)
		print("finished reading concept net")
		return data

def query_concept_net_for_surface(query, cn, n=10):
	data = cn.get(query, None)
	if data:
		return nlargest(n, data, key=itemgetter(1))
	else:
		return None

def query_concept_net_for_concept():
	#TODO: implement
	return

if __name__ == '__main__':
	#data = parse_concept_net(raw_path)
	#save_concept_net(concept_net_path)
	data = load_concept_net(concept_net_path)
	res = query_concept_net_for_surface("delta", data)
	for entry in res:
		print(entry[0])

