#!/usr/bin/python
# -*- coding: utf8 -*-
'''

@date: 17.04.18
@author: leonhard.hennig@dfki.de
'''
# from http://stackoverflow.com/questions/2276200/changing-default-encoding-of-python
import sys
import codecs

mincount = 1
minrep = 0.0

def _validate(elems):
    for e in elems:
        if ':' in e or '|' in e or '"' in e:
            print("Invalid line '%s'" % ' '.join(elems))
            return False
    return True

def convert(ne_type, infile, outfile, mincount):
    print('Converting type [%s]: [%s] -> [%s]' % (ne_type, infile, outfile))
    print('Min count >= %d and min rep value >= %.2f' % (mincount, minrep))
    with codecs.open(outfile, 'wb', 'utf8') as output:
        with codecs.open(infile, 'rb', 'utf8') as input:
            rowcount = 0
            writecount = 0
            for (i, l) in enumerate(input):
                rowcount += 1
                elems = l.strip().split('\t')
                if len(elems) == 3:
                    (concept, entity, count) = elems
                    if int(count) >= mincount:
                        if _validate(elems):
                            output.write('%s | GTYPE:%s | SUB:"%s" | N:"%s"\n' % (entity, ne_type, concept, count))
                            writecount += 1
                        else:
                            print('Not validated [%s]' % l.strip())
                    else:
                        print('Skipping %s' % l.strip())
                elif len(elems) == 6:
                    (concept, entity, count, p_c_given_e, p_e_given_c, rep) = elems
                    if int(count) >= mincount and float(rep) >= minrep:
                        if _validate(elems):
                            writecount += 1
                            output.write('%s | GTYPE:%s | SUB:"%s" | N:"%s" | C_E:"%s" | E_C:"%s" | REP:"%s"\n'
                                     % (entity, ne_type, concept, count, p_c_given_e, p_e_given_c, rep))
                        else:
                            print('Not validated [%s]' % l.strip())
                    #else:
                    #    print 'Skipping %s' % l
                else:
                    print('Len elems != 3 or 6! %s' % l.strip())
    print('Converted %d/%d entries' % (writecount, rowcount))

types = {'task':'gaz_task_type', 'material':'gaz_mat_type', 'process':'gaz_proc_type'}
#types = {'product': 'gaz_product_type', 'organization':'gaz_org_type'}
for (ne_type, gaz_type) in types.items():
    #basedir = '/home/leonhard/Dokumente/forschung/gitlab/spree/tap/tap-models/src/main/resources/tap-models/sprout/gazetteer/gazfiles'
    basedir = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/concepts'
    infile = '%s/gazetteer_%s.txt' % (basedir, ne_type)
    outfile = '%s/sdw_bootner_%s.gaz' % (basedir, ne_type)
    convert(gaz_type, infile, outfile, mincount) # use 'gaz_type' because that's the type expected by Sprout grammar rules