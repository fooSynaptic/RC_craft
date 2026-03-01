# encoding=utf-8
# /usr/bin/python3

"""
Document Reader Module

This module implements a simple document reader for machine reading comprehension.
It combines named entity recognition (NER) filtering with BM25 retrieval to find
answer sentences from context paragraphs.

Example:
    >>> from document_reader import doc_reader
    >>> answer = doc_reader(context, query)
"""

import re
import pandas as pd
from typing import List, Optional

try:
    from document_retriever import bm25_retriever
except ImportError:
    from .document_retriever import bm25_retriever

try:
    import fool
    FOOL_AVAILABLE = True
except ImportError:
    FOOL_AVAILABLE = False
    fool = None

try:
    from config import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    config = None


def refine_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Remove extra whitespace
    return ' '.join(text.split())


def extract_named_entities(text: str) -> List[str]:
    """
    Extract named entities from text using fool NER.
    
    Args:
        text: Input text
        
    Returns:
        List of named entities
    """
    if not FOOL_AVAILABLE:
        return []
    
    try:
        analysis_result = fool.analysis(text)
        if analysis_result and len(analysis_result) > 0:
            entities = analysis_result[-1][0] if len(analysis_result) > 0 else []
            return [entity[2] for entity in entities]
    except Exception:
        pass
    
    return []


def doc_reader(context: str, query: str) -> str:
    """
    Read context and extract answer based on query.
    
    Uses NER filtering to reduce candidate sentences, then applies
    BM25 to find the most relevant sentence.
    
    Args:
        context: Context paragraph
        query: Query question
        
    Returns:
        Extracted answer text
        
    Example:
        >>> context = "This is a test. Another sentence here."
        >>> query = "What is this?"
        >>> answer = doc_reader(context, query)
    """
    # Clean inputs
    context = refine_text(context)
    query = refine_text(query)
    
    # Build sentence list
    sentences = [x + '。' for x in context.split() if x]
    
    # Extract NER from query
    query_entities = extract_named_entities(query)
    
    # Filter candidate sentences by NER
    if query_entities:
        candidate_sentences = [
            sent for sent in sentences 
            if any(ner in sent for ner in query_entities)
        ]
    else:
        candidate_sentences = sentences
    
    if not candidate_sentences:
        return ""
    
    # Tokenization
    try:
        import jieba
        tokenized_sentences = [list(jieba.cut(x)) for x in candidate_sentences]
    except ImportError:
        tokenized_sentences = [list(x) for x in candidate_sentences]
    
    # BM25 retrieval
    try:
        retriever_idx = bm25_retriever(
            tokenized_sentences, 
            query, 
            mode='multi-sentences'
        )
        result = candidate_sentences[retriever_idx]
    except Exception:
        result = candidate_sentences[0] if candidate_sentences else ""
    
    return ''.join(result) if isinstance(result, list) else result


def run():
    """
    Run the document reader pipeline.
    
    Loads data and processes each example through the reader.
    """
    if not CONFIG_AVAILABLE:
        print("Config not available. Please set up the configuration.")
        return
    
    try:
        data = pd.read_csv(config.train_file)
    except FileNotFoundError:
        print(f"Data file not found: {config.train_file}")
        return
    
    for i in range(data.shape[0]):
        contents = [
            data['content1'][i], data['content2'][i], 
            data['content3'][i], data['content4'][i], 
            data['content5'][i]
        ]
        query = data['question'][i]
        answer = re.sub(r'@(\w*)@', '', data['answer'][i])
        
        # Multi-document retrieval
        content_indices = bm25_retriever(contents, query, k=1, mode='multi-paragraphs')
        
        # Get the selected content
        selected_content_idx = int(content_indices[0].replace('content', '')) - 1
        reader_output = doc_reader(contents[selected_content_idx], query)
        
        print(f'Query: {query}')
        print(f'Answer: {answer}')
        print(f'Reader Output: {reader_output}')
        print()


if __name__ == '__main__':
    run()
