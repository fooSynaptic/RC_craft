材料一：如何教会机器去理解问题和文本并且回答问题（tensorflow实战）

这部分主要是为了阐述机器阅读理解的实现

我们有的输入由这几部分组成：
- 问题q
- 和问题q相关的文档集合（利用symbolic matching做初步的召回）
- 标注好的文档中的起始和结束位点

现在需要解决的问题有：
- 我们需要找到一个映射来讲文档和问题从他们自己的空间映射到二维的整数空间；那么这个函数的具体形式是什么？（neuron network）
- 如何在向量空间中表示问题和文档？
- 什么样的损失函数是一个好的损失？
- 在实际的答案中，如果答案存在于不连续的区间，如何解决这个问题？
- 对于部分问题，如果要回答一定需要生成的手段而不是直接从原内容中抽取，要如何解决？

# Network Architecture
- embedding layer
- matching layer
- funsion layer
- decoding layer


# Embedding and Encoding Layers
- embedding and encoding layers输入一个token的序列，输出一个向量的序列，在下面的演示中，使用一个预训练的embedding matrix和双向的GRU来初始化：
```python
embed_shape = [vocab_size, vocab_embed_dim]
embed_placeholder = tf.placeholder(tf.float32, embed_shape)
word_embed = tf.get_variable("word_embeddings", embed_shape, trainable = False)

embed_init_op = word_embed.assign(embed_placeholder)

# to load precomputed embedding from numpy array `pre_embed` to the graph
with tf.Session() as sess:
	sess.run(embed_init_op, feed_dict = {embed_placeholder: pre_embed})


```

- encode问题和文档
```python
q_emb = tf.nn.embedding_lookup(word_embed, q)

with tf.variable_scope("Question_Encoder"):
	cell_fw = GRUCell(num_units=hidden_size)
	cell_bw = GRUCell(num_units=hidden_size)
	output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, q_emb, sequence_length = q_len)

	# concat the forwaed and backward encoded information
	q_encodes = tf.concat(output, 2)

# do the same to get `p_encodes`
```


# Match Layer
```python
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
```

# Fusing Layer
fusing layer的目的是为了：
- first:获取到match_out中长程的依赖。
- second: 获取到目前为止尽可能多的信息然后准备最好的decoding阶段。

采用的方法有：
- 将match_out作为双向RNN的输入，输出就是fusing layer.
- CNN,用多个conv1d to cross-correlated with match-out to produce the output of the fusing layer.

```python
# use CNN
out_dim = 64
window_len = 10

conv_match = tf.layers.conv1d(match_out, out_dim, window_len, strides = window_len)
conv_match_up = tf.squeeze(tf.image.resize_images(tf.expand_dims(conv_match, axis=-1),
	[tf.shape(match_out)[1], out_dim],
	method = ResizeMethod.NEAREST_NEIGHBOR), axis=-1)

fuse_out - tf.concat([p_encodes, match_out, conv_match_up], axis=-1)

```
***The upsampling step is required for concatenating the convoluted features with match_out and p_encodes. It can be implemented with resize_images from Tensorflow API. The size of fuse_out is [B,L,D], where B is the batch size; L is the passage length and D is the depth controlled by the convolution filters in the fusing layer.***


# Decoding Layer & Loss Function
decode `fuse_out` as an answer span.
A simple way to get such distribution is to reduce the last dimension of `fuse_out` to 1 using a dense layer, and then put a softmax over its output.
利用交叉熵损失来评估损失
```python
start_logit = tf.layers.dense(fuse_out, 1)
end_logit = tf.layers.dense(fuse_out, 1)

# mask out those padded symbols before softmax
start_logit -= (1-p_mask)*1E30
end_logit -= (1-p_mask)*1E30

# compute the loss
start_loss = tf.losses.sparse_softmax_cross_entropy(labels = start_label, logit=start_logit)
end_loss = tf.losses.sparse_softmax_cross_entropy(labels=end_label, logit=end_logit)
loss = (start_loss+end_loss)/2


```

# generate final answer
```python
max_answ_len = 50

start_prob = tf.nn.softmax(start_logit, axis=1)
end_prob = tf.nn.softmax(end_logit, axis= 1)

# do the outer product
outer = tf.matmul(tf.expand_dims(start_prob, axis=2),
		tf.expand_dims(end_prob, axis=1))
outer = tf.matrix_band_part(outer, 0, max_answ_len)

start_pos = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
end_pos = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

# extract the answer from the original passages
final_answer = passage_tokens[start_pos: end_pos+1]

```


# reference

- http://hanxiao.io/2018/04/21/Teach-Machine-to-Comprehend-Text-and-Answer-Question-with-Tensorflow/





