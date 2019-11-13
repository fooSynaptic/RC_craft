# encoding=utf-8
# /usr/bin/python3


import tensorflow as tf
import numpy as np
from random import randint
from tensorflow.nn.rnn_cell import GRUCell
from tqdm import tqdm



vocab_size = 5000
vocab_embed_dim = 300
hidden_size = 512
p_len, q_len = 100, 20
max_answ_len = 50

pre_embed = np.random.rand(vocab_size, 300)



# Embedding and Encoding Layers
embed_shape = [vocab_size, vocab_embed_dim]
embed_placeholder = tf.placeholder(tf.float32, embed_shape)
word_embed = tf.get_variable("word_embeddings", embed_shape, trainable = False)

embed_init_op = word_embed.assign(embed_placeholder)


# to load precomputed embedding from numpy array `pre_embed` to the graph
with tf.Session() as sess:
	sess.run(embed_init_op, feed_dict = {embed_placeholder: pre_embed})


p_corpus = [[randint(0, 4999) for _ in range(p_len)] for _ in range(64)]
q_corpus = [[randint(0, 4999) for _ in range(q_len)] for _ in range(64)]
start_labels = [randint(0, p_len//2) for _ in range(64)]
end_labels = [randint(p_len//2+1, p_len-1) for _ in range(64)]

# use CNN
out_dim = 64
window_len = 10

with tf.Session() as sess:   
    for i in tqdm(range(64)):
        p, q, start_label, end_label = [p_corpus[i]], \
            [q_corpus[i]], [start_labels[i]], [end_labels[i]]

        # q encodes
        q_emb = tf.nn.embedding_lookup(word_embed, q)

        with tf.variable_scope("Question_Encoder", reuse=tf.AUTO_REUSE):
            cell_fw = GRUCell(num_units=hidden_size)
            cell_bw = GRUCell(num_units=hidden_size)
            
            output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, q_emb, \
                sequence_length = [q_len], dtype = tf.float32)

            # concat the forwaed and backward encoded information
            q_encodes = tf.concat(output, 2)

        # do the same to get `p_encodes`
        # p encodes
        p_emb = tf.nn.embedding_lookup(word_embed, p)
        with tf.variable_scope("Paragraph_Encoder", reuse=tf.AUTO_REUSE):
            cell_fw = GRUCell(num_units=hidden_size)
            cell_bw = GRUCell(num_units=hidden_size)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, p_emb, \
                sequence_length = [p_len], dtype = tf.float32)
            p_encodes = tf.concat(output, 2)


        # Match Layer
        p_mask = tf.sequence_mask(p_len, tf.shape(p)[1], dtype=tf.float32, name="passage_mask")
        q_mask = tf.sequence_mask(q_len, tf.shape(q)[1], dtype=tf.float32, name="question_mask")

        sim_matrix = tf.matmul(p_encodes, q_encodes, transpose_b = True)
        sim_mask = tf.matmul(tf.expand_dims(p_mask, -1), tf.expand_dims(q_mask, -1), transpose_b=True)

        # mask out zeros by replacing it with very small number
        sim_matrix -= (1-sim_mask)*1E30

        passage2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), q_encodes)
        b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)

        question2passage_attn = tf.tile(tf.matmul(b, p_encodes),[1, tf.shape(p_encodes)[1], 1])

        p_mask = tf.expand_dims(p_mask, -1)
        passage2question_attn *= p_mask
        question2passage_attn *= p_mask

        match_out = tf.concat([p_encodes,
            p_encodes*passage2question_attn,
            p_encodes*question2passage_attn], -1)


        '''
        # Fusing Layer
        fusing layer的目的是为了：
        - first:获取到match_out中长程的依赖。
        - second: 获取到目前为止尽可能多的信息然后准备最好的decoding阶段。

        采用的方法有：
        - 将match_out作为双向RNN的输入，输出就是fusing layer.
        - CNN,用多个conv1d to cross-correlated with match-out to produce the output of the fusing layer.
        '''



        conv_match = tf.layers.conv1d(match_out, out_dim, window_len, strides = window_len)
        conv_match_up = tf.squeeze(tf.image.resize_images(tf.expand_dims(conv_match, axis=-1),
            [tf.shape(match_out)[1], out_dim],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), axis=-1)

        fuse_out = tf.concat([p_encodes, match_out, conv_match_up], axis=-1)


        # Decoding Layer & Loss Function
        '''decode `fuse_out` as an answer span.
        A simple way to get such distribution is to reduce the last dimension of `fuse_out` to 1 using a dense layer, and then put a softmax over its output.
        利用交叉熵损失来评估损失
        '''

        start_logit = tf.layers.dense(fuse_out, 1, trainable=False)
        end_logit = tf.layers.dense(fuse_out, 1, trainable=False)

        # mask out those padded symbols before softmax
        start_logit -= (1-p_mask)*1E30
        end_logit -= (1-p_mask)*1E30

        # compute the loss
        start_loss = tf.losses.sparse_softmax_cross_entropy(labels = start_label, logits=start_logit)
        end_loss = tf.losses.sparse_softmax_cross_entropy(labels=end_label, logits=end_logit)
        loss = (start_loss+end_loss)/2
        sess.run(tf.initialize_all_variables())
        print(sess.run(loss))

        # optimization

        with tf.variable_scope('optimization', reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.98, epsilon=1e-8)
            optimizer.minimize(loss)


        # generate final answer
        start_prob = tf.nn.softmax(start_logit, axis=1)
        end_prob = tf.nn.softmax(end_logit, axis= 1)

        start_prob = tf.reduce_sum(start_prob, -1)
        end_prob = tf.reduce_sum(end_prob, -1)

        # do the outer product
        #print(sess.run(tf.shape(start_prob)))
        #print(sess.run(tf.shape(end_prob)))


        outer = tf.matmul(tf.expand_dims(start_prob, axis=2), tf.expand_dims(end_prob, axis=1))
        #outer = tf.matmul(start_prob, end_prob)
        outer = tf.matrix_band_part(outer, 0, max_answ_len) # filter

        start_pos = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        end_pos = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

        # extract the answer from the original passages
        sess.run(tf.initialize_all_variables())
        s, e = sess.run([start_pos, end_pos])
        final_answer = p[0][s[0]: (e[0]+1)]
        #print(final_answer)


