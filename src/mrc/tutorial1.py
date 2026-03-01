# encoding=utf-8
# /usr/bin/python3

"""
Neural Machine Reading Comprehension Model

This module implements a neural reading comprehension model with:
- Bi-directional GRU encoders for passage and question
- Bi-directional attention flow (BiDAF) matching layer
- CNN-based fusing layer
- Pointer network decoding for answer span prediction

Reference:
    Bi-directional Attention Flow for Machine Comprehension (Seo et al., 2017)
"""

import tensorflow as tf
import numpy as np
from random import randint
from tensorflow.nn.rnn_cell import GRUCell
from tqdm import tqdm


# Model hyperparameters
VOCAB_SIZE = 5000
VOCAB_EMBED_DIM = 300
HIDDEN_SIZE = 512
PASSAGE_LENGTH = 100
QUESTION_LENGTH = 20
MAX_ANSWER_LENGTH = 50

# Pre-trained embeddings (random init for demo)
PRE_EMBEDDINGS = np.random.rand(VOCAB_SIZE, VOCAB_EMBED_DIM)


def create_embedding_layer():
    """Create and initialize the embedding layer."""
    embed_shape = [VOCAB_SIZE, VOCAB_EMBED_DIM]
    embed_placeholder = tf.placeholder(tf.float32, embed_shape)
    word_embeddings = tf.get_variable(
        "word_embeddings", 
        embed_shape, 
        trainable=False
    )
    embed_init_op = word_embeddings.assign(embed_placeholder)
    return word_embeddings, embed_placeholder, embed_init_op


def encode_question(word_embeddings, question, question_length):
    """
    Encode question using bi-directional GRU.
    
    Args:
        word_embeddings: Word embedding lookup table
        question: Question token indices
        question_length: Length of question sequence
        
    Returns:
        Encoded question representation
    """
    q_emb = tf.nn.embedding_lookup(word_embeddings, question)
    
    with tf.variable_scope("Question_Encoder", reuse=tf.AUTO_REUSE):
        cell_fw = GRUCell(num_units=HIDDEN_SIZE)
        cell_bw = GRUCell(num_units=HIDDEN_SIZE)
        
        output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, 
            cell_bw, 
            q_emb,
            sequence_length=[question_length], 
            dtype=tf.float32
        )
        
        # Concatenate forward and backward encoded information
        q_encodes = tf.concat(output, 2)
    
    return q_encodes


def encode_passage(word_embeddings, passage, passage_length):
    """
    Encode passage using bi-directional GRU.
    
    Args:
        word_embeddings: Word embedding lookup table
        passage: Passage token indices
        passage_length: Length of passage sequence
        
    Returns:
        Encoded passage representation
    """
    p_emb = tf.nn.embedding_lookup(word_embeddings, passage)
    
    with tf.variable_scope("Paragraph_Encoder", reuse=tf.AUTO_REUSE):
        cell_fw = GRUCell(num_units=HIDDEN_SIZE)
        cell_bw = GRUCell(num_units=HIDDEN_SIZE)
        output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, 
            cell_bw, 
            p_emb,
            sequence_length=[passage_length], 
            dtype=tf.float32
        )
        p_encodes = tf.concat(output, 2)
    
    return p_encodes


def match_layer(p_encodes, q_encodes, passage_length, question_length):
    """
    Bi-directional attention flow matching layer.
    
    Computes similarity matrix and applies both passage-to-question
    and question-to-passage attention.
    
    Args:
        p_encodes: Encoded passage
        q_encodes: Encoded question
        passage_length: Length of passage
        question_length: Length of question
        
    Returns:
        Match layer output
    """
    # Create masks
    p_mask = tf.sequence_mask(
        passage_length, 
        tf.shape(p_encodes)[1], 
        dtype=tf.float32, 
        name="passage_mask"
    )
    q_mask = tf.sequence_mask(
        question_length, 
        tf.shape(q_encodes)[1], 
        dtype=tf.float32, 
        name="question_mask"
    )
    
    # Similarity matrix
    sim_matrix = tf.matmul(p_encodes, q_encodes, transpose_b=True)
    sim_mask = tf.matmul(
        tf.expand_dims(p_mask, -1), 
        tf.expand_dims(q_mask, -1), 
        transpose_b=True
    )
    
    # Mask out zeros
    sim_matrix -= (1 - sim_mask) * 1e30
    
    # Passage-to-question attention
    passage2question_attn = tf.matmul(
        tf.nn.softmax(sim_matrix, -1), 
        q_encodes
    )
    
    # Question-to-passage attention
    b = tf.nn.softmax(
        tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), 
        -1
    )
    question2passage_attn = tf.tile(
        tf.matmul(b, p_encodes),
        [1, tf.shape(p_encodes)[1], 1]
    )
    
    # Apply mask
    p_mask = tf.expand_dims(p_mask, -1)
    passage2question_attn *= p_mask
    question2passage_attn *= p_mask
    
    # Concatenate features
    match_out = tf.concat([
        p_encodes,
        p_encodes * passage2question_attn,
        p_encodes * question2passage_attn
    ], -1)
    
    return match_out


def fuse_layer(match_out, p_encodes, conv_dim=64, window_len=10):
    """
    Fusing layer using CNN.
    
    Args:
        match_out: Output from match layer
        p_encodes: Encoded passage
        conv_dim: Convolution output dimension
        window_len: Convolution window length
        
    Returns:
        Fused representation
    """
    conv_match = tf.layers.conv1d(
        match_out, 
        conv_dim, 
        window_len, 
        strides=window_len
    )
    conv_match_up = tf.squeeze(
        tf.image.resize_images(
            tf.expand_dims(conv_match, axis=-1),
            [tf.shape(match_out)[1], conv_dim],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        ), 
        axis=-1
    )
    
    fuse_out = tf.concat([p_encodes, match_out, conv_match_up], axis=-1)
    return fuse_out


def decode_layer(fuse_out, start_label, end_label, p_mask):
    """
    Decoding layer with pointer network.
    
    Args:
        fuse_out: Fused representation
        start_label: Start position label
        end_label: End position label
        p_mask: Passage mask
        
    Returns:
        Loss tensor, start logits, end logits
    """
    start_logit = tf.layers.dense(fuse_out, 1, trainable=False)
    end_logit = tf.layers.dense(fuse_out, 1, trainable=False)
    
    # Mask padded symbols
    start_logit -= (1 - p_mask) * 1e30
    end_logit -= (1 - p_mask) * 1e30
    
    # Compute loss
    start_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=start_label, 
        logits=start_logit
    )
    end_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=end_label, 
        logits=end_logit
    )
    loss = (start_loss + end_loss) / 2
    
    return loss, start_logit, end_logit


def get_answer_span(start_logit, end_logit, max_answer_len):
    """
    Extract answer span from logits.
    
    Args:
        start_logit: Start position logits
        end_logit: End position logits
        max_answer_len: Maximum answer length
        
    Returns:
        Start and end positions
    """
    start_prob = tf.nn.softmax(start_logit, axis=1)
    end_prob = tf.nn.softmax(end_logit, axis=1)
    
    start_prob = tf.reduce_sum(start_prob, -1)
    end_prob = tf.reduce_sum(end_prob, -1)
    
    # Outer product
    outer = tf.matmul(
        tf.expand_dims(start_prob, axis=2), 
        tf.expand_dims(end_prob, axis=1)
    )
    outer = tf.matrix_band_part(outer, 0, max_answer_len)  # Filter
    
    start_pos = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
    end_pos = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
    
    return start_pos, end_pos


def main():
    """Main training loop."""
    # Create embedding layer
    word_embeddings, embed_placeholder, embed_init_op = create_embedding_layer()
    
    # Initialize with pre-trained embeddings
    with tf.Session() as sess:
        sess.run(embed_init_op, feed_dict={embed_placeholder: PRE_EMBEDDINGS})
    
    # Generate dummy data
    p_corpus = [[randint(0, 4999) for _ in range(PASSAGE_LENGTH)] for _ in range(64)]
    q_corpus = [[randint(0, 4999) for _ in range(QUESTION_LENGTH)] for _ in range(64)]
    start_labels = [randint(0, PASSAGE_LENGTH // 2) for _ in range(64)]
    end_labels = [randint(PASSAGE_LENGTH // 2 + 1, PASSAGE_LENGTH - 1) for _ in range(64)]
    
    with tf.Session() as sess:
        for i in tqdm(range(64)):
            passage = [p_corpus[i]]
            question = [q_corpus[i]]
            start_label = [start_labels[i]]
            end_label = [end_labels[i]]
            
            # Encode
            q_encodes = encode_question(word_embeddings, question, QUESTION_LENGTH)
            p_encodes = encode_passage(word_embeddings, passage, PASSAGE_LENGTH)
            
            # Match
            match_out = match_layer(
                p_encodes, 
                q_encodes, 
                PASSAGE_LENGTH, 
                QUESTION_LENGTH
            )
            
            # Fuse
            fuse_out = fuse_layer(match_out, p_encodes)
            
            # Decode
            p_mask = tf.sequence_mask(
                PASSAGE_LENGTH, 
                tf.shape(passage)[1], 
                dtype=tf.float32
            )
            p_mask = tf.expand_dims(p_mask, -1)
            
            loss, start_logit, end_logit = decode_layer(
                fuse_out, 
                start_label, 
                end_label, 
                p_mask
            )
            
            sess.run(tf.initialize_all_variables())
            print(sess.run(loss))
            
            # Optimization
            with tf.variable_scope('optimization', reuse=tf.AUTO_REUSE):
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.001, 
                    beta1=0.9, 
                    beta2=0.98, 
                    epsilon=1e-8
                )
                optimizer.minimize(loss)
            
            # Generate final answer
            start_pos, end_pos = get_answer_span(
                start_logit, 
                end_logit, 
                MAX_ANSWER_LENGTH
            )
            
            sess.run(tf.initialize_all_variables())
            s, e = sess.run([start_pos, end_pos])
            final_answer = passage[0][s[0]:(e[0] + 1)]


if __name__ == "__main__":
    main()
