from os.path import basename
import glob


def xml_text_to_plain(fname, outfolder):
	with open(fname) as fd:
		doc = xmltodict.parse(fd.read())
		base_name = basename(fname).split('.')[0]
		try:
			text = doc['Document']['TEXT']['#text']
		except:
			text = doc['Document']['TEXT']

		base_name = basename(fname).split('.')[0]
		out_path = outfolder + base_name + '.txt'
		f = open(out_path, 'w')
		f.write(text)
		f.close()

def parse_all_xml(folder, out_path):
	# parse folder
	folder = glob.glob(folder + '*.xml')
	for file in folder:
		xml_text_to_plain(file, out_path)

def read_files():
	import glob
	import os
	ids = []
	files = glob.glob("science_corpus/*.txt")
	for f in files:
		idx = os.path.basename(f).split('.')[0]
		ids.append(idx)
	return ids

def write_files(ids):
	for entry in ids:
		name = 'science_corpus/' + entry + '.xml'
		#print('writing file: {}'.format(name))
		with open(name, 'w') as the_file:
			pass

if __name__ == '__main__':
	#folder = 'uber_vico/original/'
	#out_path = 'uber_vico/original/texts/'
	#parse_all_xml(folder, out_path)
	ids = read_files()
	write_files(ids)