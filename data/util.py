import xmltodict
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


if __name__ == '__main__':
	folder = 'uber_vico/original/'
	out_path = 'uber_vico/original/texts/'
	parse_all_xml(folder, out_path)