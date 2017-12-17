import os
import sys
import pickle
import operator
import numpy as np
import math

from preprocessing import tokenize 

pathname = os.path.dirname(sys.argv[0])
ROOT_DIR = os.path.abspath(pathname)
DATA_DIR = 'data'

def make_vector(tf):
    return np.array(tf.values())

def get_tf_for_doc(tf, doc_id):
    doc_id = str(doc_id)
    return { term : tf[term][doc_id] if doc_id in tf[term] else 0 for term in tf }

def get_tf_idf_for_doc(tfidf, doc_id):
    doc_id = str(doc_id)
    return { term : tfidf[term][doc_id] if doc_id in tfidf[term] else 0 for term in tfidf }

def get_tf_for_query(tf, query):
    my_tf = {}
    words = tokenize(query)
    for word in words:
        if (word not in my_tf):
            my_tf[word] = 0
        my_tf[word] += 1
    return { term : my_tf[term] if term in my_tf else 0 for term in tf }

def get_ranking_with_info(scores, songs, n=10):
    res = []
    for i in range(n):
        doc_id, score = scores[i]
        if score == 0 or math.isnan(score): break
        res.append({ 'rank': i+1, 'score': score, 'song': songs[doc_id] })
    return res

def display_result(results):
    if len(results) == 0:
        print 'No matching song'
        print

    for i, result in enumerate(results):
        print('Position #' + str(i+1) + ', score: ' + str(result['score']))
        print('-' * 32)
        print result['song']
        print

def sorted_by_value(dicts):
    return sorted(dicts.items(), key=operator.itemgetter(1), reverse=True)

def dump_to_pickle(obj, filename, dir=None):
    if not dir: dir = os.path.join(ROOT_DIR, DATA_DIR)
    with open(os.path.join(dir, filename + '.pickle'), 'wb') as file:
		pickle.dump(obj, file)

def load_from_pickle(filename, dir=None):
    if not dir: dir = os.path.join(ROOT_DIR, DATA_DIR)
    file_path = os.path.join(dir, filename + '.pickle')
    if not os.path.isfile(file_path):
        return False
    return pickle.load(open(file_path, 'rb'))
