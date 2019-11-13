# encoding = utf-8
# /usr/bin/python3

'''
Let's first try doc2vec to retrive most simliar paragraphs from different documents
In our experiment, the bm25 retriver reach a 0.912 accuracy, which is fast and accurate
But the disadvantage is this retriver is hard to optimizer...
'''

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization import bm25


from config import config

import pandas as pd
import numpy as np
import jieba
import re
import json
from jieba import cut
import codecs


#stopwords
#stop_words = '/Users/yiiyuanliu/Desktop/nlp/demo/stop_words.txt'
stop_words = '../../../data/stopwords/stop_words_zh.txt'
stopwords = codecs.open(stop_words,'r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords ]

stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']



def refine(line):
    line = re.sub("[\s\p']", "", line)
    #line = re.sub(r'[0-9]+', 'n', line)
    #line = re.sub(r'[a-zA-Z]+', 'α', line)
    return line



def doc2vec_Retriver(contents, query, top_k = 1, mod = 'mean'):
	sentences, paragraphs, i = [], {}, 1

	for content in contents:
		tmp_para = []
		for sentence in content.split('。'):
			if not sentence: continue
			sentence += '。'
			sentences.append(refine(sentence))
			tmp_para.append(refine(sentence))
		paragraphs['content{}'.format(i)] = tmp_para
		i += 1
	del i


	#doc2vec modeling
	sentences = [list(jieba.cut(x)) for x in sentences]

	documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
	try:
		model = Doc2Vec(vector_size=512, window=5, min_count=2, workers=4, compute_loss = True)
		model.build_vocab(documents)
		model.train(documents, total_examples = model.corpus_count, epochs = model.iter)
		model.alpha -= 0.002
		model.min_alpha = model.alpha
	except Exception as e:
		raise Exception(e)


	#retrive
	q = model.infer_vector(jieba.cut(refine(query)))

	scores = {}
	for cnt in paragraphs.keys():
		s = []
		for sent in paragraphs[cnt]:
			vec = model.infer_vector(list(jieba.cut(sent)))
			sim_score = cosine_similarity([q, vec])[1, 0]
			s.append(sim_score)

		scores[cnt] = np.mean(s) if mod == 'mean' else len([x for x in s if x > 0.55])

	return [x[0] for x in sorted(scores.items(), key=lambda scores:scores[1])[-top_k:]]



def bm25_retriver(contents, query, k = 1, mod = 'multi-paragraphs'):
	corpus = [refine(content) for content in contents]

	# modeling
	corpus = [list(jieba.cut(x)) for x in corpus]

	retriver = bm25.BM25(corpus)

	#q
	q = list(cut(refine(query)))
	average_idf = sum(map(lambda k: float(retriver.idf[k]), retriver.idf.keys())) / len(retriver.idf.keys())
	#return ['content{}'.format(i+1) for i in np.argsort(retriver.get_scores(q, average_idf))[::-1][:k]]
	if mod == 'multi-paragraphs': 
		return ['content{}'.format(i+1) for i in np.argsort(retriver.get_scores(q, average_idf))[::-1][:k]]
	elif mod == 'multi-sentences':
		return np.argmax(retriver.get_scores(q, average_idf))




def main():
	data = pd.read_csv(config.train_file)
	patient = 1000


	for k in range(1, 6):
		bear = 0
		acc = 0.
		for i in range(data.shape[0]):
			contents = [data['content1'][i], data['content2'][i], data['content3'][i], data['content4'][i], data['content5'][i]]
			query = data['question'][i]
			target = re.findall("@(\w*)@", data['supporting_paragraph'][i])
			res = bm25_retriver(contents, query, k)
                        
			acc += 1 if len([x for x in target if x in res]) > 0 else 0


		print(k, 'accuracy: ', acc/data.shape[0])


if __name__ == '__main__':
	#main()
	pass
