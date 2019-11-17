You may find the source post [here](http://hanxiao.io/2018/09/09/Dual-Ask-Answer-Network-for-Machine-Reading-Comprehension/), here is a self-learning trail for MRC.


## recap
In the last post, i have introduced the task of machine reading comprehension and presented as simple neural arichitecture for tackling suck task. In fact, the architecture can be found in many state-of-art MRC models, eg:
- bidaf
- s-net
- r-net
- match-lstm
- reason-net
- document reader
- Reinforced mnemonic reader
- funsionNet
- QAnet

I also pointed out aan assumption made in the architecture: the answer is always a continuous span of a given passage. Under this assumpotion, an answer can be simplified as a pair of two integers, representing its start and end position in the passage respectively. This greatly reduces the solution space and simplifies the training, yielding promsing score on squad dataset. Unfortunately, beyoud artificial datasets this assumption is often not true in practice.

## background
As we already know, there are three modalities in the reading comprehension setting: quesiton, answer and context. One can define two problems from different directions:
- Question answering(QA): infer an answer given a question and the context;
- Question generation(QG): infer a question given an answer and the context;

Careful reader may notice that there is some sort of cycle consistency between QA and QG: they are both defined as inferring one modality given the conterpary and the context. This makes one wondering if the roles of questions and answers are simply invertible. After all, they are both short text and only make sense when given the context. it is like an object stands side-on to a mirror, whrereas the context is the mirror itself, as illustrated in the next figure.

![figure form source post](http://hanxiao.io/2018/09/09/Dual-Ask-Answer-Network-for-Machine-Reading-Comprehension/d20488e5.png)

In the real world, we know that a good reading comprehension ability means not only givening perfect answer but alsogood question. In fact, there is an effectivestategy used to teach reading at school called partner reading, in which two students read an assined text and ask one another questions in turn.

At tecent AI lab, they takes a deep give into the relationship between questions and answers, or what we call it, the duality. We consider QA and QG at two strongly correlated tasks and equally important to the reading comprehension ability. In particular, they develop a unified nueral network that:
- Learns QA and QG simutaneously in an end-to-end manner;
- Exploits the duality between QA and QG to benefit each other.

## Dual learning paradigms
![paradigms from source post](http://hanxiao.io/2018/09/09/Dual-Ask-Answer-Network-for-Machine-Reading-Comprehension/262269bc.png)

## Dual ask-answer network(note here the question and answer infering both implementted NLG)
![illustater from source](http://hanxiao.io/2018/09/09/Dual-Ask-Answer-Network-for-Machine-Reading-Comprehension/05bd204f.png)

# Embedding Layer
The embedding layer maps each word to a high-dimensional vector space. The vector representation includes the word-level and the character-level information. The parameters of this layer are shared by context, question and answer. For the word embedding, we use pre-trained 256-dimensional Glove word vectors, which are fixed during traning. For the charactor embedding, each character is represented as a 200-dimensional tranable vector. Character embedding is extremely useful for representing OOV words.
But how can one combine the word embedding with the character embedding? each word is first represented as a sequence of character vectors, where the sequence length is either truncated or padded to 16. Next consider 1D CNN with kernel width 3 follwed by. As a result, it gives us a fixed-size 200-dimensional vector for each word. The final output of the embedding layer is the concatednation of the word and character embeddings.

# Encoding Layer
The encoding layer contains three encoders for context, question and answer, respectively. They are shared by QA and QG, as depicted in the last figure. That is given QA and QG dual tasks, the encoder of the primal task and decoder of the dual task are forced to be the same. This parameter sharing scheme serves and as a regularization to influence the traning on both tasks. It also helps the model to find a more general and stable representation for each modality.

Each encoder consists of the following basic building blocks: an element-wise fully connected feed-forward block, stacked LSTMs and a self-attention block. The final output of an encoder is a concatednation of the outputs pf all blocks, as illustrated in the next figure.
![embedding layer to attention layer](http://hanxiao.io/2018/09/09/Dual-Ask-Answer-Network-for-Machine-Reading-Comprehension/2b1c354a.png)

We use some state-of-the-art text encoding algorithms with tensorflow:`tf-nlp-block`. Now it's time to release the hounds! Here is an example implementation of the encoding layer.

```python
from nlp.encode_blocks import TCN_encode, LSTM_encode
from nlp.match_blocks im[ort Transformer_match

def _encoding_layer(self):
    q_encodes, c_encodes, a_encodes = [], [], []
    for algo in self.args.encode_algo:
        if algo == 'TCN':
            kwargs = {
                'dropout_keep_rate': self.ph_dropout_keep_prob,
                'residual': self.args.encode_residual,
                'normalize_output': self.args.encode_normalize_output
            }
            q_encode = TCN_encode(self.q_emb, self.args.encode_num_layers, causality=True, scope='TCN_query', **kwargs)

            c_encode = TCN_encode(self.c_emb, self.args.encode_num_layers, scope='TCN_context', **kwargs)

            a_encode = TCN_encode(self.a_emb, self.args.encode_num_layers, causality=True, scope='TCN_answer', **kwargs)

        elif algo == 'LSTM':
            kwargs = {
                'num_layers': self.args.encode_num_layers,
                'direction': self.args.encode_direction
            }
            q_encode = LSTM_encode(self.q_emb, causality=True, scope='LSTM_query', **kwargs)
            c_encode = LSTM_encode(self.c_emb, scope='LSTM_context', **kwargs)
            a_encode = LSTM_encode(self.a_emb, causality=True, scope='LSTM_answer', **kwargs)
        elif algo == 'TRANSFORMER':
            kwargs = {
                'dropout_keep_rate': self.ph_dropout_keep_prob,
                'residual': self.args.encode_residual,
                'normalize_output': self.args.encode_normalize_output,
                'num_heads': self.args.encode_num_heads
            }
            q_encode = Transformer_match(self.q_emb, self.q_emb, self.q_mask, self.q_mask, causality=True, scope='Transformer_query', **kwargs)
            c_encode = Transformer_match(self.c_emb, self.c_emb, self.c_mask, self.c_mask, scope='Transformer_context', **kwargs)
            a_encode = Transformer_match(self.a_emb, self.a_emb, self.a_mask, self.a_mask, causality=True,scope='Transformer_answer', **kwargs)
        else:
            raise NotImplementedError

        q_encodes.append(q_encode)
        c_encodes.append(c_encode)
        a_encodes.append(a_encode)
    self.q_encode = tf.concat(q_encodes, axis=2) if len(q_encodes) > 1 else q_encodes[0]
    self.c_encode = tf.concat(c_encodes, axis=2) if len(c_encodes) > 1 else c_encodes[0]
    self.a_encode = tf.concat(a_encodes, axis=2) if len(a_encodes) > 1 else a_encodes[0]
"""
Careful readers may notice that `causality=True` is set particularly for question ans answer encoders. As we are going to use question and answer encoders for decoding. Preventing the backward signal is crucial to preserve the auto-regressive property. For LSTM we simply use the uni-directional network;  for self-attention we keep only attention of later position to early position, meanwhile setting the remaining to -inf.
"""
```
# Attention Layer
In the attention layer, we develop a two-step attention that folds in all information observed so far for generating final sequences. The first fold-in step captures the interaction between question/answer and context and represents it as a new context sequence. The second fold-in step mimics the typical encoder-decoder attention mechanisms in the sequence-to-sequence models. The next figure illustrates this process:
![two step attention(match+vanillia seq2seq)](http://hanxiao.io/2018/09/09/Dual-Ask-Answer-Network-for-Machine-Reading-Comprehension/a7bafb75.png)

below is the implementation:
```python
def _attention_layer_QA(self):
    kwargs = {
        'dropout_keep_rate': self.ph_dropout_keep_prob,
        'residual': self.args.match_residual,
        'normalize_output': self.args.match_normalize_output,
    }
    if self.args.match_algo == 'ATT_CNN':
        first_attention = AttentiveCNN_match(self.c_encode, self.q_encode, self.c_mask, self.q_mask, score_func=self.args.match_attentive_score, **kwargs)
        second_attention = AttentiveCNN_match(self.a_encode, first_attention, self.a_mask, self.c_mask, score_func=self.args.match_attentive_score, **kwargs)

    elif self.args.match_algo == 'SIMPLE_ATT':
        first_attention, _ = Attentive_match(self.c_encode, self.q_encode, self.c_mask, self.q_mask, score_func=self.args.match_attentive_score, **kwargs)
        second_attention, _ = Attentive_match(self.a_encode, first_attention, self.c_mask, self.q_mask, score_func=self.args.match_attentive_score, **kwargs)

    elif self.args.match_algo == 'TRANSFORMER':
        first_attention = Transformer_match(self.c_encode, self.q_encode, self.c_mask, self.q_mask, num_heads=self.args.match_num_heads, **kwargs)
        second_attention = Transformer_match(self.a_encode, first_attention, self.c_mask, self.q_mask, num_heads=self.args.match_num_heads, **kwargs)
    else:
        raise NotImplementedError
    return second_attention
```

# output layer
The output layer generates an output sequence one word at a time. At each step the model is auto-regressive, consuming the wrods previously generated as inpput when generating the next. In this work, we can employ the pointer generator as the core component of the output layer. It allows both copying words from the context via pointing, and sampling words from a fixed vocabulary. This aids accureate reproduction of information especially in QA, while retaining the ability to generate novel wordds.

# loss function
During training, the model is fed with a question-context-answer triplet(Q,C,A) and the decoded QHat and AHat from the output layer are trained to be similar to Q and A, respectively. To achieve that, our loss function consists of two parts: the negative log-likelihood loss widely used in the sequence transduction model and a coverage loss to penalize repetition of the generated text.

We employ teacher forcing stategy in training, where the model receives the groundtruth token from time `t-1` as input and predicts the token at time `t`.

# Duality in the Model
Let’s take a quick summary before moving on to the next. Our model exploits the duality of QA and QG in two places.

- As we consider both QA and QG are sequence generation problems, our architecture is reflectional symmetric. The left QG part is a mirror of the right QA part with identical structures. Such symmetry can be also found in the attention calculation and the loss function. Consequently, the answer modality and the question modality are connected in a two-way process through the context modality, allowing the model to infer answers or questions given the counterpart based on context.
- Our model contains shared components between QA and QG at different levels. Starting from the bottom, the embedding layer and the context encoder are always shared between two tasks. Moreover, the answer encoder in QG is reused in QA for generating the answer sequence, and vice versa. On top of that, in the the pointer generator, QA and QG share the same latent space before the final projection to the vocabulary space (please check the paper for more details). The cycle consistency between question and answer is utilized to regularize the training process at different levels, helping the model to find a more general representation for each modality.

## conclusion
In the real world, finding the best answer may require reasoning across multiple evidence snippets and involve a series of cognitive processes such as inferring and summarizing. Note that, all models mentioned in this series can not even answer simple yes/no questions, e.g. “Is this written in English?”, “Is this an article about animals?”. They also can’t refuse to answer questions when the given context provides not enough information. For these reasons, developing a real end-to-end MRC model is still a great challenge today.


