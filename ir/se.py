import os
import sys
import pickle

from models import VSM
from utils import display_result

pathname = os.path.dirname(sys.argv[0])
ROOT_DIR = os.path.abspath(pathname)
DATA_DIR = 'data'

def get_vsm(dir=None):
    if not dir:
        dir = os.path.join(ROOT_DIR, DATA_DIR)
    corpus = pickle.load(open(os.path.join(dir, 'corpus.pickle'), 'rb'))
    tf = pickle.load(open(os.path.join(dir, 'tf.pickle'), 'rb'))
    idf = pickle.load(open(os.path.join(dir, 'idf.pickle'), 'rb'))
    tfidf = pickle.load(open(os.path.join(dir, 'tfidf.pickle'), 'rb'))
    return VSM(corpus, tf, idf, tfidf)

if __name__ == '__main__':
    vsm = get_vsm()

    while True:
        q = raw_input('|?- ')
        results = vsm.search(q)
        display_result(results)
