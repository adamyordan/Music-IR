import numpy as np
from tqdm import tqdm

from utils import make_vector, get_tf_idf, get_tf_for_doc, get_tf_idf_for_doc, get_tf_for_query, sorted_by_value, get_ranking_with_info, dump_to_pickle, load_from_pickle
from similarities import sim

class VSM:
    def __init__(self, corpus, tf, idf, tfidf, idf_artist, tfidf_artist, idf_genre, tfidf_genre, idf_title, tfidf_title):
        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        self.tfidf = tfidf
        self.songs = { doc['index']: doc for doc in corpus }
        self.idf_artist = idf_artist
        self.tfidf_artist = tfidf_artist
        self.idf_genre = idf_genre
        self.tfidf_genre = tfidf_genre 
        self.idf_title = idf_title
        self.tfidf_title = tfidf_title 

        print('caching tf_idf vectors...')
        self.doc_tf_idf_vectors        = { doc_id: make_vector(get_tf_idf_for_doc(tfidf, doc_id))        for doc_id in tqdm(self.get_doc_ids()) }
        self.doc_tf_idf_vectors_artist = { doc_id: make_vector(get_tf_idf_for_doc(tfidf_artist, doc_id)) for doc_id in tqdm(self.get_doc_ids()) }
        self.doc_tf_idf_vectors_genre  = { doc_id: make_vector(get_tf_idf_for_doc(tfidf_genre, doc_id))  for doc_id in tqdm(self.get_doc_ids()) }
        self.doc_tf_idf_vectors_title  = { doc_id: make_vector(get_tf_idf_for_doc(tfidf_title, doc_id))  for doc_id in tqdm(self.get_doc_ids()) }

    def get_tf_idf_for_doc(self, doc_id, tfidf):
        doc_id = str(doc_id)
        return { term : tfidf[term][doc_id] if doc_id in tfidf[term] else 0 for term in tfidf }

    def get_doc_ids(self):
        return [ doc['index'] for doc in self.corpus]

    def get_scores_tf(self, query, sim_algo='cosine'):
        query_vector = make_vector(get_tf_for_query(self.tf, query))
        scores = { doc_id: sim(query_vector, make_vector(get_tf_for_doc(self.tf, doc_id)), sim_algo) for doc_id in tqdm(self.get_doc_ids()) }
        return sorted_by_value(scores)

    def get_scores_tf_idf(self, query, sim_algo='cosine'):
        query_vector = make_vector(self.get_tf_idf(get_tf_for_query(self.tf, query)))
        scores = { doc_id : sim(query_vector, self.doc_tf_idf_vectors[doc_id], sim_algo) for doc_id in tqdm(self.get_doc_ids()) }
        return sorted_by_value(scores)

    def get_scores_tf_idf_weighted(self, query, weight, sim_algo='cosine'):
        if not weight: weight = { 'lyrics': 0.3, 'artist': 0.4, 'genre': 0.4, 'title': 0.5 }

        query_vector_lyrics = make_vector(get_tf_idf(get_tf_for_query(self.tf, query), self.idf))
        query_vector_artist = make_vector(get_tf_idf(get_tf_for_query(self.tfidf_artist, query), self.idf_artist))
        query_vector_genre  = make_vector(get_tf_idf(get_tf_for_query(self.tfidf_genre, query), self.idf_genre))
        query_vector_title  = make_vector(get_tf_idf(get_tf_for_query(self.tfidf_title, query), self.idf_title))

        scores_lyrics = { doc_id : sim(query_vector_lyrics, self.doc_tf_idf_vectors[doc_id], sim_algo)        for doc_id in tqdm(self.get_doc_ids()) }
        scores_artist = { doc_id : sim(query_vector_artist, self.doc_tf_idf_vectors_artist[doc_id], sim_algo) for doc_id in tqdm(self.get_doc_ids()) }
        scores_genre  = { doc_id : sim(query_vector_genre , self.doc_tf_idf_vectors_genre[doc_id], sim_algo)  for doc_id in tqdm(self.get_doc_ids()) }
        scores_title  = { doc_id : sim(query_vector_title , self.doc_tf_idf_vectors_title[doc_id], sim_algo)  for doc_id in tqdm(self.get_doc_ids()) }

        scores = { doc_id: weight['lyrics'] * scores_lyrics[doc_id] + weight['artist'] * scores_artist[doc_id] + weight['genre'] * scores_genre[doc_id] + weight['title'] * scores_title[doc_id] for doc_id in tqdm(self.get_doc_ids()) }
        return sorted_by_value(scores), { 'lyrics': scores_lyrics, 'artist': scores_artist, 'genre': scores_genre, 'title': scores_title }
        
    def search(self, query, weight=None, n=10):
        scores, score_details = self.get_scores_tf_idf_weighted(query, weight)
        return get_ranking_with_info(scores, self.songs, n, score_details)
