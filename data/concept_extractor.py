import logging, coloredlogs
import csv
from collections import defaultdict
from heapq import nlargest
from operator import itemgetter
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import pickle as pkl

import config
from datastores import Collection

coloredlogs.install()


class MacOSFile(object):
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        pkl.dump(obj, MacOSFile(f), protocol=pkl.HIGHEST_PROTOCOL)

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pkl.load(MacOSFile(f))


class ConceptNet:
    def __init__(self, verbose=False, create=False):
        self.verbose = verbose
        self.raw_path = config.CONCEPT_NET_PATH_RAW
        self.parsed_path = config.CONCEPT_NET_PATH_PARSED
        self.parsed_path_surface = config.CONCEPT_NET_PATH_SURFACE_PARSED

        self.concept_net = self.parse_concept_net() if create else self.load_concept_net()
        self.concept_net_surfaces = self.parse_concept_net_surfaces() if create else self.load_concept_net_surface()

    @staticmethod
    def _intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    def parse_concept_net(self):
        if self.verbose:
            logging.info('started parsing concept net.')
        db = defaultdict(list)
        with open(self.raw_path) as fd:
            rd = csv.reader(fd, delimiter="\t")
            for n, row in enumerate(rd):
                if self.verbose:
                    if (n + 1) % 1000000 == 0:
                        logging.info("finished {}%.".format(np.round(n / 33377000, 2) * 100))
                data = (row[0], row[2], row[3], row[4], row[5])
                if len(data) == 5:
                    db[row[1]].append(data)
                else:
                    logging.info('row not complete: {}'.format(data))
        self.save_concept_net(dict(db))
        return dict(db)

    def parse_concept_net_surfaces(self):
        if self.verbose:
            logging.info('started parsing concept net.')
        db = defaultdict(list)
        with open(self.raw_path) as fd:
            rd = csv.reader(fd, delimiter="\t")
            for n, row in enumerate(rd):
                if self.verbose:
                    if (n + 1) % 1000000 == 0:
                        logging.info("finished {}%.".format(np.round(n / 33377000, 2) * 100))
                data = (row[1], row[2], row[3], row[4], row[5])
                if len(data) == 5:
                    db[row[0]].append()
                else:
                    logging.info('row not complete: {}'.format(data))
        self.save_concept_net_surface(dict(db))
        return dict(db)

    def save_concept_net(self, dic):
        if self.verbose:
            logging.info('saving concept net.')

        pickle_dump(dic, self.parsed_path)

        if self.verbose:
            logging.info('concept net saved.')

    def save_concept_net_surface(self, dic):
        if self.verbose:
            logging.info('saving concept net surfaces.')

        pickle_dump(dic, self.parsed_path_surface)

        if self.verbose:
            logging.info('concept net surfaces saved.')

    def load_concept_net(self):
        if self.verbose:
            logging.info('loading concept net.')

        db = pickle_load(self.parsed_path)

        if self.verbose:
            logging.info('finished loading concept net.')

        return db

    def load_concept_net_surface(self):
        if self.verbose:
            logging.info('loading concept net.')

        db = pickle_load(self.parsed_path_surface)

        if self.verbose:
            logging.info('finished loading concept net.')

        return db

    def query_concept_net_for_surface(self, query, n=10):
        data = self.concept_net.get(query.lower(), None)
        if data:
            return nlargest(n, data, key=itemgetter(1))
        else:
            return None

    def query_concept_net_for_concept(self, query):
        data = self.concept_net_surfaces.get(query.lower(), None)
        if data:
            return data
        else:
            return None

    def query_concept_net_for_concepts(self, list_of_concepts):
        surface_words = []
        n = 0
        for surface, concepts in self.concept_net.items():
            if concepts:
                concept_words = [x[0] for x in concepts]
                try:
                    if self._intersection(concept_words, list_of_concepts):
                        surface_words.append(surface)
                except:
                    logging.warning('ERROR. surface: {}, concept_words: {}'.format(surface, concept_words))
            else:
                if self.verbose:
                    logging.info('no concepts found for surface: {}'.format(surface))
            if (n + 1) % 1000000 == 0:
                logging.info("finished checking {}% of surface words.".format(np.round(n / len(self.concept_net), 2) * 100))
            n += 1
        return surface_words


class ConceptExtractor:
    def __init__(self, collection, concept_net, entity, verbose=False,
                 gathered_concepts=None, additional_surfaces=None, top_n_concepts=None,
                 load_concepts=None):
        self.collection = collection
        self.entity = entity
        self.verbose = verbose
        self.additional_surfaces = additional_surfaces
        self.surfaces = self._extract_surfaces(collection)
        self.unique_surfaces = list(set(self.surfaces))
        self.num_entities = len(self.surfaces)
        self.missmatches = []
        self.gazetteer = None
        self.gathered_concepts = gathered_concepts if gathered_concepts else []
        self.unique_gathered_concepts = None
        self.concept_net = concept_net
        self.top_n_concepts= top_n_concepts
        self.used_concepts = None
        self.load_concepts = load_concepts

    def _extract_surfaces(self, collection):
        logging.info('starting surface extraction for entity: {}'.format(self.entity))
        surfaces = []
        for n_doc, document in enumerate(collection.annotations):
            for annotation in document.tokens:
                if annotation.entity == self.entity:
                    surfaces.append(annotation.word)
            if self.verbose:
                if (n_doc + 1) % 100 == 0:
                    logging.info('finished extracting surfaces from {} documents.'.format(n_doc + 1))
        if self.additional_surfaces:
            surfaces.extend(self.additional_surfaces)
            #surfaces = self.additional_surfaces
        return surfaces

    def query_concepts(self):
        # query surfaces for concepts
        if self.verbose:
            logging.info('starting concept extraction.')
        for n_sur, surface in enumerate(self.surfaces):
            concepts = self.concept_net.query_concept_net_for_surface(surface, n=config.NUM_TOP_CONCEPTS)
            if concepts:
                self.gathered_concepts.extend([x[0] for x in concepts])
            else:
                self.missmatches.append(surface)
            if (n_sur + 1) % 1000 == 0:
                logging.info('finished {} surfaces.'.format(n_sur))
        if self.verbose:
            logging.info('{}% of surfaces returned no concepts.'.format(np.round((len(self.missmatches) / self.num_entities) * 100, 2)))
        self.unique_gathered_concepts = list(set(self.gathered_concepts))
        self.concept_net.concept_net = None

    def query_surfaces(self):
        # query concepts for surfaces
        if self.load_concepts:
            with open(self.load_concepts) as f:
                content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]
            concepts = content
        else:
            if self.top_n_concepts:
                concepts = [x[0] for x in Counter(self.gathered_concepts).most_common(self.top_n_concepts)]
                self.used_concepts = concepts
                logging.warning(len(concepts))
            else:
                concepts = self.unique_gathered_concepts
                logging.warning(len(concepts))
        gazetteer = []
        if self.verbose:
            logging.info('starting surface extraction.')
        for n_con, concept in enumerate(concepts):
            surfaces = self.concept_net.query_concept_net_for_concept(concept)
            if surfaces:
                processed_surfaces = self.parse_surfaces(surfaces, concept)
                gazetteer.extend(processed_surfaces)
            if (n_con + 1) % 1000 == 0:
                logging.info('finished {} concepts.'.format(n_con))
        if self.verbose:
            logging.info('concepts queried, surfaces extracted.')
        self.gazetteer = gazetteer
        self.concept_net.concept_net_surfaces = None

    def write_files(self):
        if self.gathered_concepts and not self.used_concepts:
            # write file
            path = 'concepts/concepts_{}.txt'.format(self.entity)
            with open(path, 'w') as ofile:
                ofile.write('\n'.join(self.gathered_concepts))
        elif self.used_concepts:
            # write file
            path = 'concepts/concepts_{}.txt'.format(self.entity)
            with open(path, 'w') as ofile:
                ofile.write('\n'.join(self.used_concepts))

        if self.gazetteer:
            # write file
            path = 'concepts/gazetteer_{}.txt'.format(self.entity)
            with open(path, 'a') as ofile:
                for surface in self.gazetteer:
                    ofile.write('\t'.join(surface))
                    ofile.write('\n')

        if self.missmatches:
            path = 'concepts/missmatches_{}.txt'.format(self.entity)
            with open(path, 'w') as ofile:
                ofile.write('\n'.join(self.missmatches))

        logging.info('saved files into folder concepts.')


    def get_statistics(self):
        # entity statistics
        unique_entities = len(list(set(self.surfaces)))
        avg_tokens_entity = np.sum([len(x.split()) for x in self.surfaces]) / len(self.surfaces)

        # concept statistics
        num_concepts = len(self.gathered_concepts)
        unique_concepts = len(list(set(self.gathered_concepts)))

        # gazetteer statistics
        if self.gazetteer:
            num_entries_gaz = len([x[1] for x in self.gazetteer])
            unique_entries_gaz = len(list(set([x[1] for x in self.gazetteer])))

        #self.plot_box(self.gathered_concepts)
        self.plot_histo(self.gathered_concepts, 1000)

        logging.info('Number of entities for entity {}: {}'.format(self.entity, self.num_entities))
        logging.info('Number of unique entities for entity {}: {}'.format(self.entity, unique_entities))
        logging.info('Avg tokens per entity: {}'.format(avg_tokens_entity))
        logging.info('Number of returned concepts from CN: {}'.format(num_concepts))
        logging.info('Number of unique returned concepts: {}'.format(unique_concepts))
        logging.info('Number of missmatches: {}'.format(len(self.missmatches)))
        if self.gazetteer:
            logging.info('Number of entries in gazetteer: {}'.format(num_entries_gaz))
            logging.info('Number of unique entries in gazetteer: {}'.format(unique_entries_gaz))

    def plot_histo(self, word_list, top_n=1000):
        plt.figure()
        ### plot for top 50
        counts = dict(Counter(word_list).most_common(50))
        labels, values = zip(*counts.items())
        # sort your values in descending order
        indSort = np.argsort(values)[::-1]
        # rearrange your data
        labels = np.array(labels)[indSort]
        values = np.array(values)[indSort]
        indexes = np.arange(len(labels))

        bar_width = 0.2
        plt.subplot(2, 1, 2)
        plt.bar(indexes, values)
        plt.xticks(indexes + bar_width, labels, rotation=90)
        plt.title('Concept Distribution - {} - top N: {}'.format(self.entity, 50))

        ### plot for top_n
        counts = dict(Counter(word_list).most_common(top_n))
        labels, values = zip(*counts.items())
        # sort your values in descending order
        indSort = np.argsort(values)[::-1]
        # rearrange your data
        labels = np.array(labels)[indSort]
        values = np.array(values)[indSort]
        indexes = np.arange(len(labels))

        plt.subplot(2, 1, 1)
        plt.bar(indexes, values)
        plt.title('Concept Distribution - {} - top N: {}'.format(self.entity, 50))

        plt.savefig('concept_dist_{}.pdf'.format(self.entity))


    def plot_box(self, word_list):
        counts = dict(Counter(word_list))
        labels, values = zip(*counts.items())

        plt.figure()
        plt.boxplot(values)
        plt.title('Boxplot - {} - average: {}'.format(self.entity, np.mean(values)))
        plt.savefig('box_{}.pdf'.format(self.entity))

    def parse_surfaces(self, surfaces, concept):
        parsed_surfaces = []
        for surface in surfaces:
            tmp = list(surface)
            tmp.insert(0, concept.lower())
            parsed_surfaces.append(tuple(tmp))
        return parsed_surfaces

