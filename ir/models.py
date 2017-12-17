import numpy as np
from tqdm import tqdm

from utils import make_vector, get_tf_for_doc, get_tf_for_query, sorted_by_value, get_ranking_with_info, dump_to_pickle, load_from_pickle
from similarities import cosine

class VSM:
    def __init__(self, corpus, tf, idf, tfidf, use_cache=True):
        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        self.tfidf = tfidf
        self.songs = { doc['index']: doc for doc in corpus }

        print('caching tf_idf vectors...')
        vsm_tfidf = {}
        for doc_id in tqdm(self.get_doc_ids()):
            doc_vector = make_vector(self.get_tf_idf_for_doc(doc_id))
            vsm_tfidf[doc_id] = doc_vector
        self.doc_tf_idf_vectors = vsm_tfidf

    def get_tf_idf_for_doc(self, doc_id):
        doc_id = str(doc_id)
        return { term : self.tfidf[term][doc_id] if doc_id in self.tfidf[term] else 0 for term in self.tfidf }

    def get_doc_ids(self):
        return [ doc['index'] for doc in self.corpus]

    def get_tf_idf(self, tf):
        return { term : float(tf[term]) / self.idf[term] for term in tf }

    def get_scores_tf(self, query):
        query_vector = make_vector(get_tf_for_query(self.tf, query))
        scores = { doc_id: cosine(query_vector, make_vector(get_tf_for_doc(self.tf, doc_id))) for doc_id in tqdm(self.get_doc_ids()) }
        return sorted_by_value(scores)

    def get_scores_tf_idf(self, query):
        query_vector = make_vector(self.get_tf_idf(get_tf_for_query(self.tf, query)))
        scores = { doc_id : cosine(query_vector, self.doc_tf_idf_vectors[doc_id]) for doc_id in tqdm(self.get_doc_ids()) }
        return sorted_by_value(scores)
        
    def search(self, query, n=10):
        scores = self.get_scores_tf_idf(query)
        return get_ranking_with_info(scores, self.songs, n)
