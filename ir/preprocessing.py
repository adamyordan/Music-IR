import pickle
import re
import sys
import os.path
import math
import string
import re

import pandas as pd
from tqdm import tqdm
from nltk.stem import PorterStemmer

split_pattern = re.compile("(?<!^)\s+(?=[A-Z])(?!.\s)")
ps = PorterStemmer()

def tokenize(document):
	# lowercase
	document = document.lower()

	# keep only alphabet and whitespaces
	document = re.sub(r'[^A-Za-z\s\-]+', '', document)

	# split by whitespaces
	words = re.split(r'[\s\-]+', document)

	words = [ps.stem(word) for word in words]
	return words

def get_tf(tf, index, document):
	words = tokenize(document)

	for word in words:
		if (not word in tf):
			tf[word] = {}

		if (not index in tf):
			tf[word][index] = 0

		tf[word][index] += 1

	return tf

def beautify_title(title):
	title = re.sub('-', ' ', title)
	title = re.sub('i m', 'i\'m', title)
	title = string.capwords(title)
	return title

def get_idf(idf, document):
	words = tokenize(document)

	current_idf = {}
	for word in words:
		if (not word in current_idf):
			current_idf[word] = True

	for token in current_idf:
		if (not token in idf):
			idf[token] = 0

		idf[token] += 1

	return idf

def get_tf_idf(tf, idf, N):
	tfidf = {}
	N = float(N)
	for word in tf:
		tfidf[word] = {}
		for index in tf[word]:
			tfidf[word][index] = tf[word][index] * math.log(N / idf[word])
	return tfidf

def get_tf_idf_per_doc(tfidf, doc_ids):
	tfidf_per_doc = {}
	print('caching tfidf per document...')
	for doc_id in tqdm(doc_ids):
		doc_id_s = str(doc_id)
		tfidf_per_doc[doc_id] = { term : tfidf[term][doc_id] if doc_id in tfidf[term] else 0.0 for term in tfidf }
	return tfidf_per_doc

def dump_to_pickle(obj, filename, dir=None):
    if not dir: dir = os.path.join(ROOT_DIR, DATA_DIR)
    with open(os.path.join(dir, filename + '.pickle'), 'wb') as file:
		pickle.dump(obj, file)

if __name__ == '__main__':
	pathname = os.path.dirname(sys.argv[0])
	ROOT_DIR = os.path.abspath(pathname)
	DATA_DIR = 'data'

	print('Loading data...')
	column_names = ['index', 'song', 'year', 'artist', 'genre', 'lyrics']
	df = pd.read_csv(os.path.join(ROOT_DIR, DATA_DIR, 'lyrics.csv'), names=column_names, index_col=0)
	df = df.iloc[1:]
	print('Data loaded')

	print('Generating TF-IDF...')
	corpus = []
	tf  = {}
	idf = {}
	cnt = 0

	tf_artist  = {}
	idf_artist = {}
	tf_genre   = {}
	idf_genre  = {}
	tf_title   = {}
	idf_title  = {}

	for i, song in tqdm(df.iterrows()):
		if (song.isnull().values.any()):
			continue
		if (cnt == 3000):
			break
		cnt += 1
		data = {
			'index': i,
			'title': song['song'],
			'good_title': beautify_title(song['song']),
			'year': song['year'],
			'artist': song['artist'],
			'genre': song['genre'],
			'lyrics': song['lyrics'],
		}

		tf = get_tf(tf, i, data['lyrics'])
		idf = get_idf(idf, data['lyrics'])

		tf_artist  = get_tf(tf_artist, i, data['artist'])
		idf_artist = get_idf(idf_artist, data['artist'])
		tf_genre   = get_tf(tf_genre, i, data['genre'])
		idf_genre  = get_idf(idf_genre, data['genre'])
		tf_title   = get_tf(tf_title, i, data['title'])
		idf_title  = get_idf(idf_title, data['title'])

		corpus.append(data)

	tfidf = get_tf_idf(tf, idf, len(corpus))
	tfidf_artist = get_tf_idf(tf_artist, idf_artist, len(corpus))
	tfidf_genre  = get_tf_idf(tf_genre , idf_genre , len(corpus))
	tfidf_title  = get_tf_idf(tf_title , idf_title , len(corpus))

	with open(os.path.join(ROOT_DIR, DATA_DIR, 'corpus.pickle'), 'wb') as file:
		print('Number of data in corpus = {}'.format(len(corpus)))
		print('Dumping corpus...')
		pickle.dump(corpus, file)

	print('Dumping TF-IDF...')
	with open(os.path.join(ROOT_DIR, DATA_DIR, 'tf.pickle'), 'wb') as file:
		pickle.dump(tf, file)

	with open(os.path.join(ROOT_DIR, DATA_DIR, 'idf.pickle'), 'wb') as file:
		print('Number of unique words = {}'.format(len(idf)))
		pickle.dump(idf, file)

	with open(os.path.join(ROOT_DIR, DATA_DIR, 'tfidf.pickle'), 'wb') as file:
		pickle.dump(tfidf, file)

	dump_to_pickle(tf_artist, 'tf.artist')
	dump_to_pickle(idf_artist, 'idf.artist')
	dump_to_pickle(tfidf_artist, 'tfidf.artist')
	dump_to_pickle(tf_genre, 'tf.genre')
	dump_to_pickle(idf_genre, 'idf.genre')
	dump_to_pickle(tfidf_genre, 'tfidf.genre')
	dump_to_pickle(idf_title, 'idf.title')
	dump_to_pickle(tfidf_title, 'tfidf.title')
