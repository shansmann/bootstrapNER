#!/usr/bin/python
# -*- coding: utf8 -*-
'''

@date: 30.03.17
@author: leonhard.hennig@dfki.de
'''
# from http://stackoverflow.com/questions/2276200/changing-default-encoding-of-python
import sys
import codecs
from scipy.sparse import csr_matrix
from scipy import log, exp
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


def calc_p_e_given_c(m, total_count):
    #global log_smoothed_p_e_given_c, concept_idx, entity_idx
    # not vectorized, rather inefficient
    log_smoothed_p_e_given_c = {}
    for concept_idx in range(m.shape[0]):
        concept_row = m.getrow(concept_idx)
        concept_row_sum = float(concept_row.sum())
        if concept_row_sum > 0:
            for entity_idx in concept_row.nonzero()[1]:
                log_smoothed_p_e_given_c[(concept_idx, entity_idx)] = log(concept_row[0, entity_idx] + eps) - log(
                    concept_row_sum + eps * total_count)
        if concept_idx % 1000000 == 0:
            print "Computed %d/%d p(e|c)" % (concept_idx, m.shape[0])
    return log_smoothed_p_e_given_c


def calc_p_c_given_e(m):

    log_p_c_given_e = {}
    m_by_col = m.tocsc()
    for entity_idx in range(m_by_col.shape[1]):
        entity_col = m_by_col.getcol(entity_idx)
        entity_col_sum = entity_col.sum()
        if entity_col_sum > 0:
            for concept_idx in entity_col.nonzero()[0]:
                log_p_c_given_e[(concept_idx, entity_idx)] = log(entity_col[concept_idx, 0]) - log(entity_col_sum)
        if entity_idx % 1000000 == 0:
            print "Computed %d/%d p(c|e)" % (entity_idx, m_by_col.shape[1])
    return log_p_c_given_e


def write_results(outfile, m, log_smoothed_p_e_given_c, log_p_c_given_e, concept_dict_rev, entity_dict_rev):

    with codecs.open(outfile, 'wb', 'utf8') as f:
        for (i, (concept_idx, entity_idx)) in enumerate(log_smoothed_p_e_given_c):
            if (concept_idx, entity_idx) in log_p_c_given_e:
                rep = log_smoothed_p_e_given_c[(concept_idx, entity_idx)] + log_p_c_given_e[
                    (concept_idx, entity_idx)]
                # write rep
                f.write("%s\t%s\t%d\t%.6f\t%.6f\t%.6f\n" % (concept_dict_rev[concept_idx], entity_dict_rev[entity_idx],
                                                            m[concept_idx, entity_idx],
                                                            exp(log_p_c_given_e[(concept_idx, entity_idx)]),
                                                            exp(log_smoothed_p_e_given_c[(concept_idx, entity_idx)]),
                                                            exp(rep)))
            if i % 1000000 == 0:
                print "Wrote %d/%d lines" % (i, len(log_smoothed_p_e_given_c))


def init_data(infile, min_count, eps):

    concept_dict = {}
    entity_dict = {}
    concept_dict_rev = {}
    entity_dict_rev = {}
    # from example on https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    concept_row_indices = []
    entity_col_indices = []
    data = []
    total_count = 0
    with codecs.open(infile, 'rb', 'utf8') as f:
        for (i, l) in enumerate(f):
            (concept, entity, count) = l.strip().split('\t')
            # skip low-count entries
            if int(count) <= min_count: continue

            concept_index = concept_dict.setdefault(concept, len(concept_dict))
            concept_dict_rev[concept_dict[concept]] = concept
            entity_index = entity_dict.setdefault(entity, len(entity_dict))
            entity_dict_rev[entity_dict[entity]] = entity
            concept_row_indices.append(concept_index)
            entity_col_indices.append(entity_index)
            data.append(int(count))
            total_count += int(count)
            if i % 1000000 == 0:
                print "Processed %d lines" % i
    print "Creating matrix..."
    m = csr_matrix((data, (concept_row_indices, entity_col_indices)), dtype=int)
    return (m, total_count, concept_dict_rev, entity_dict_rev)

if __name__ == '__main__':

    min_count = 0
    eps = 0.0001
    infile = '/home/leonhard/Dokumente/forschung/data/ms-concept-graph/data-concept-instance-relations.txt'
    #infile = '/home/leonhard/Dokumente/forschung/data/ms-concept-graph/test-microsoft.txt'
    outfile = '/home/leonhard/Dokumente/forschung/data/ms-concept-graph/data-concept-instance-relations-with-blc.tsv'

    (m, total_count, concept_dict_rev, entity_dict_rev) = init_data(infile, min_count, eps)

    # compute BLC as per https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/scoring.pdf
    # P(e|c), smoothed = N(c,e) + eps / (Sum over e_i N(c,e_i) + eps  * total_instances)
    # BLC = P(e|c) * P(c|e)
    print "Computing BLC..."
    log_smoothed_p_e_given_c = calc_p_e_given_c(m, m.shape[1]) # 12501527
    log_p_c_given_e = calc_p_c_given_e(m)

    write_results(outfile, m, log_smoothed_p_e_given_c, log_p_c_given_e, concept_dict_rev, entity_dict_rev)

    print 'Done.'
