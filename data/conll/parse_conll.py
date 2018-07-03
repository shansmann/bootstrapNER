file = 'test.txt'
with open(file) as f:
	content = f.readlines()

with open('test_parsed.txt', 'a') as the_file:
	for entry in content:
		items = entry.split()
		if len(items) == 0:
			the_file.write('\n')
		elif len(items) == 4:
			entity = items[-1]
			if entity == 'O':
				items.append('O')
			else:
				items.append(entity[-3:])
			line = '\t'.join(items)
			#print(line)
			the_file.write(line + '\n')
		else:
			print('parsing error, line: {}'.format(items))
