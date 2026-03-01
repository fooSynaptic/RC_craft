# RC_craft - Machine Reading Comprehension from Scratch

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 1.x](https://img.shields.io/badge/tensorflow-1.x-orange.svg)](https://www.tensorflow.org/)

A collection of machine reading comprehension (MRC) algorithms built from scratch using TensorFlow. This project demonstrates the implementation of key components in modern MRC systems, including document retrieval and neural reading comprehension models.

## <img src=".github/icons/book.svg" width="16" height="16" alt="book"> Overview

Machine Reading Comprehension is a challenging NLP task where a machine reads a given passage and answers questions based on its understanding. This project implements:

- **Document Retrieval**: BM25 and Doc2Vec-based passage retrieval
- **Neural Reading Comprehension**: Bi-directional attention flow (BiDAF) based model
- **End-to-End Pipeline**: From document retrieval to answer extraction

## <img src=".github/icons/rocket.svg" width="16" height="16" alt="rocket"> Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/fooSynaptic/RC_craft.git
cd RC_craft

# Install dependencies
pip install tensorflow==1.15 numpy tqdm gensim scikit-learn jieba
```

## <img src=".github/icons/folder.svg" width="16" height="16" alt="folder"> Project Structure

```
RC_craft/
├── src/
│   ├── mrc/
│   │   └── tutorial1.py          # Neural MRC model implementation
│   ├── retrieve/
│   │   ├── document_reader.py    # Document reader with NER filtering
│   │   └── document_retriever.py # BM25 and Doc2Vec retrievers
│   └── config.py                 # Configuration settings
├── TeachMachineComprehendTextAndAnswerQuestion_part1.md  # Tutorial Part 1
├── TeachMachineComprehendTextAndAnswerQuestion_part2.md  # Tutorial Part 2
└── README.md                     # This file
```

## <img src=".github/icons/book.svg" width="16" height="16" alt="book"> Components

### 1. Document Retrieval (`src/retrieve/document_retriever.py`)

Implements two retrieval algorithms:

- **BM25**: Probabilistic retrieval function that ranks documents based on query terms
- **Doc2Vec**: Neural network-based document embeddings with cosine similarity

```python
from src.retrieve.document_retriever import bm25_retriever, doc2vec_retriever

# BM25 retrieval
results = bm25_retriever(documents, query, k=3, mode='multi-paragraphs')

# Doc2Vec retrieval
results = doc2vec_retriever(documents, query, top_k=3, mode='mean')
```

### 2. Document Reader (`src/retrieve/document_reader.py`)

Implements a simple reading comprehension system:

- Uses NER (Named Entity Recognition) to filter relevant sentences
- Applies BM25 to find the most relevant sentence
- Returns the answer span

```python
from src.retrieve.document_reader import doc_reader

answer = doc_reader(context, query)
```

### 3. Neural MRC Model (`src/mrc/tutorial1.py`)

A TensorFlow implementation of a neural reading comprehension model featuring:

- **Embedding Layer**: Pre-trained word embeddings
- **Encoding Layer**: Bi-directional GRU for passage and question encoding
- **Match Layer**: Bi-directional attention flow between passage and question
- **Fusing Layer**: CNN-based feature fusion
- **Decoding Layer**: Pointer network for answer span prediction

## <img src=".github/icons/target.svg" width="16" height="16" alt="target"> Model Architecture

```
Input (Passage + Question)
    ↓
Embedding Layer
    ↓
Encoding Layer (Bi-GRU)
    ↓
Match Layer (BiDAF Attention)
    ↓
Fusing Layer (CNN)
    ↓
Decoding Layer (Pointer Network)
    ↓
Answer Span (start_pos, end_pos)
```

## <img src=".github/icons/chart.svg" width="16" height="16" alt="chart"> Results

The BM25 retriever achieves ~91.2% accuracy on the test dataset for document retrieval.

## <img src=".github/icons/link.svg" width="16" height="16" alt="link"> Related Resources

- [Tutorial Part 1](TeachMachineComprehendTextAndAnswerQuestion_part1.md) - Introduction to MRC
- [Tutorial Part 2](TeachMachineComprehendTextAndAnswerQuestion_part2.md) - Advanced techniques

## <img src=".github/icons/note.svg" width="16" height="16" alt="note"> License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
