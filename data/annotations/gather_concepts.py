import requests
import json
import time

import sys
sys.path.append('../concept_net')
import parse_concept_net

concept_url = "https://concept.research.microsoft.com/api/Concept/ScoreByCross?"
timeout = 10


def load_products(fname):
	with open(fname) as f:
		products = f.readlines()
		return products

def create_entry(name, concepts):
	return {
		"name": name,
		"concepts": concepts
	}

def request_concepts(product):
	r = requests.get(concept_url, params={
		"instance": product,
		"topK" : 10,
	})
	return r.json()

def parse_response(product, response, data):
	message = response.get('Message', None)
	if not message:
		print('product {} added'.format(product))
		datum = create_entry(product, response)
		data["data"].append(datum)
	else:
		print('error:', product)
		time.sleep(30)
	if not response or message:
		data["no_response"] += 1
		print('no response:', data["no_response"])

def query_products(mode, products, data):
	if mode == 'on':
		for product in products:
			response = request_concepts(product)
			parse_response(product, response, data)
			time.sleep(timeout)
	else:
		lower_products = [x.lower() for x in products]
		concept_net = parse_concept_net.load_concept_net('../concept_net/concept_net.json')
		for product in lower_products:
			if not product:
				continue
			res = parse_concept_net.query_concept_net_for_surface(product, concept_net)
			if res:
				response = {}
				for (concept, score) in res:
					response[concept] = score
				parse_response(product, response, data)
			else:
				parse_response(product, {}, data)

	return data


if __name__ == '__main__':
	fname = 'task.txt'
	ofile = 'task_concepts.json'

	products = load_products(fname)
	products = [x.strip() for x in products]

	total_number = len(products)

	# removing duplicates
	products = list(set(products))
	unique_number = len(products)

	data = {
		"total": total_number,
		"unique": unique_number,
		"no_response": 0,
		"data": []
	}

	num_products = len(products)

	print('{} products found. process will take {} mins'.format(num_products, (num_products * timeout) / 60))
	data = query_products('off', products, data)
	#data = query_products('on', products, data)

	print(data)
	# writing
	json.dump(data, open(ofile, 'w'))