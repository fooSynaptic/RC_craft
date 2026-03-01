# encoding=utf-8
# /usr/bin/python3

"""
Document Retrieval Module

This module implements document retrieval algorithms for machine reading comprehension:
- Doc2Vec: Neural network-based document embeddings
- BM25: Probabilistic retrieval function

Example:
    >>> from document_retriever import bm25_retriever
    >>> results = bm25_retriever(documents, query, k=3)
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization import bm25
import pandas as pd
import numpy as np
import jieba
import re
import json
import codecs

# Configuration
STOP_WORDS_PATH = '../../../data/stopwords/stop_words_zh.txt'

# Load stopwords
try:
    with codecs.open(STOP_WORDS_PATH, 'r', encoding='utf8') as f:
        STOPWORDS = [w.strip() for w in f.readlines()]
except FileNotFoundError:
    STOPWORDS = []

STOP_FLAGS = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']


def refine_text(line: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        line: Raw text string
        
    Returns:
        Cleaned text string
    """
    line = re.sub("[\s\p']", "", line)
    return line


def doc2vec_retriever(
    contents: list, 
    query: str, 
    top_k: int = 1, 
    mode: str = 'mean'
) -> list:
    """
    Retrieve documents using Doc2Vec embeddings.
    
    Args:
        contents: List of document contents
        query: Query string
        top_k: Number of top documents to return
        mode: Scoring mode ('mean' or 'count')
        
    Returns:
        List of top-k document identifiers
        
    Example:
        >>> docs = ["doc1 content", "doc2 content"]
        >>> results = doc2vec_retriever(docs, "query", top_k=2)
    """
    sentences = []
    paragraphs = {}
    idx = 1
    
    for content in contents:
        tmp_para = []
        for sentence in content.split('。'):
            if not sentence:
                continue
            sentence += '。'
            sentences.append(refine_text(sentence))
            tmp_para.append(refine_text(sentence))
        paragraphs[f'content{idx}'] = tmp_para
        idx += 1
    
    # Doc2Vec modeling
    sentences = [list(jieba.cut(x)) for x in sentences]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
    
    try:
        model = Doc2Vec(
            vector_size=512, 
            window=5, 
            min_count=2, 
            workers=4, 
            compute_loss=True
        )
        model.build_vocab(documents)
        model.train(
            documents, 
            total_examples=model.corpus_count, 
            epochs=model.epochs
        )
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    except Exception as e:
        raise Exception(f"Doc2Vec training failed: {e}")
    
    # Retrieve
    query_vector = model.infer_vector(jieba.cut(refine_text(query)))
    
    scores = {}
    for cnt in paragraphs.keys():
        similarity_scores = []
        for sent in paragraphs[cnt]:
            sent_vector = model.infer_vector(list(jieba.cut(sent)))
            sim_score = cosine_similarity([query_vector, sent_vector])[1, 0]
            similarity_scores.append(sim_score)
        
        if mode == 'mean':
            scores[cnt] = np.mean(similarity_scores)
        else:
            scores[cnt] = len([x for x in similarity_scores if x > 0.55])
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    return [x[0] for x in sorted_scores[-top_k:]]


def bm25_retriever(
    contents: list, 
    query: str, 
    k: int = 1, 
    mode: str = 'multi-paragraphs'
) -> list or int:
    """
    Retrieve documents using BM25 algorithm.
    
    Args:
        contents: List of document contents
        query: Query string
        k: Number of top documents to return (for multi-paragraphs mode)
        mode: Retrieval mode ('multi-paragraphs' or 'multi-sentences')
        
    Returns:
        - List of document identifiers (multi-paragraphs mode)
        - Index of best sentence (multi-sentences mode)
        
    Example:
        >>> docs = ["doc1 content", "doc2 content"]
        >>> results = bm25_retriever(docs, "query", k=2)
    """
    corpus = [refine_text(content) for content in contents]
    
    # Tokenization
    corpus = [list(jieba.cut(x)) for x in corpus]
    
    # BM25 modeling
    retriever = bm25.BM25(corpus)
    
    # Query processing
    query_tokens = list(jieba.cut(refine_text(query)))
    average_idf = sum(
        float(retriever.idf[token]) for token in retriever.idf.keys()
    ) / len(retriever.idf.keys())
    
    scores = retriever.get_scores(query_tokens, average_idf)
    
    if mode == 'multi-paragraphs':
        top_indices = np.argsort(scores)[::-1][:k]
        return [f'content{i+1}' for i in top_indices]
    elif mode == 'multi-sentences':
        return int(np.argmax(scores))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    """
    Main evaluation function for BM25 retriever.
    
    Evaluates BM25 retrieval accuracy across different k values.
    """
    try:
        from config import config
        data = pd.read_csv(config.train_file)
    except (ImportError, FileNotFoundError):
        print("Config or data file not found. Please set up the data path.")
        return
    
    for k in range(1, 6):
        correct_count = 0
        for i in range(data.shape[0]):
            contents = [
                data['content1'][i], data['content2'][i], 
                data['content3'][i], data['content4'][i], 
                data['content5'][i]
            ]
            query = data['question'][i]
            target = re.findall(r"@(\w*)@", data['supporting_paragraph'][i])
            result = bm25_retriever(contents, query, k)
            
            if any(x in result for x in target):
                correct_count += 1
        
        accuracy = correct_count / data.shape[0]
        print(f'k={k}, accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    # main()
    pass
