import pickle
import re
import sys
import os.path

import pandas as pd
from tqdm import tqdm

def tokenize(document):
	# lowercase
	document = document.lower()

	# keep only alphabet and whitespaces
	document = re.sub(r'[^A-Za-z\s]+', '', document)

	# split by whitespaces
	words = document.split()

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
	tf = {}
	idf = {}

	for i, song in tqdm(df.iterrows()):
		if (song.isnull().values.any()):
			continue

		data = {
			'index': int(i),
			'title': song['song'],
			'year': song['year'],
			'artist': song['artist'],
			'genre': song['genre'],
			'lyrics': song['lyrics'],
		}


		tf = get_tf(tf, i, data['lyrics'])
		idf = get_idf(idf, data['lyrics'])

		corpus.append(data)

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
