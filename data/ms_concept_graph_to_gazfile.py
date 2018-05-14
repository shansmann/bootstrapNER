#!/usr/bin/python
# -*- coding: utf8 -*-
'''

@date: 17.04.18
@author: leonhard.hennig@dfki.de
'''
# from http://stackoverflow.com/questions/2276200/changing-default-encoding-of-python
import codecs


mincount = 10

def _validate(elems):
    for e in elems:
        if ':' in e or '|' in e or '"' in e:
            print("Invalid line '%s'" % ' '.join(elems))
            return False
    return True

def convert(ne_type, infile, outfile, mincount):
    print('Converting type [%s]: [%s] -> [%s]' % (ne_type, infile, outfile))
    with codecs.open(outfile, 'wb', 'utf8') as output:
        with codecs.open(infile, 'rb', 'utf8') as input:
            rowcount = 0
            writecount = 0
            for (i, l) in enumerate(input):
                rowcount += 1
                elems = l.strip().split('\t')
                if len(elems) == 3:
                    print('line with 3 elems: ', l.strip().split('\t'))
                    (concept, entity, count) = elems
                    if int(count) >= mincount and float(rep) > 0.0:
                        if _validate(elems):
                            output.write('%s | GTYPE:%s | SUB:"%s" | N:"%s"\n' % (entity, ne_type, concept, count))
                            writecount += 1
                elif len(elems) == 6:
                    (concept, entity, count, p_c_given_e, p_e_given_c, rep) = elems
                    if int(count) >= mincount:
                        if float(rep) > 0.0:
                            if _validate(elems):
                                writecount += 1
                                output.write('%s | GTYPE:%s | SUB:"%s" | N:"%s" | C_E:"%s" | E_C:"%s" | REP:"%s"\n'
                                         % (entity, ne_type, concept, count, p_c_given_e, p_e_given_c, rep))
                        else:
                            print('# too low, word:', entity)
                    else:
                        print('wordcount too low, word:', entity)
    print('Converted %d/%d entries' % (writecount, rowcount))




#types = {'task':'gaz_task_type', 'material':'gaz_mat_type', 'product':'gaz_prod_type', 'organization':'gaz_org_type', 'process':'gaz_proc_type'}
types = {'prod': 'gaz_product_type'}
for (ne_type, gaz_type) in types.items():
    basedir = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/concepts'
    infile = '%s/%s_surfaces.txt' % (basedir, ne_type)
    outfile = '%s/sdw_bootner_%s.gaz' % (basedir, ne_type)
    convert(gaz_type, infile, outfile, mincount) # use 'gaz_type' because that's the type expected by Sprout grammar rules

