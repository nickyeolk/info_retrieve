import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

class QnaEncoderModel:
    def __init__(self):
        # Set up graph.
        g = tf.Graph()
        with g.as_default():
            embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1")
            # put placeholders
            self.question = tf.placeholder(dtype=tf.string, shape=[None])  # question
            self.response = tf.placeholder(dtype=tf.string, shape=[None])  # response
            self.response_context = tf.placeholder(dtype=tf.string, shape=[None])  # response context
            
            self.question_embeddings = embed(
            dict(input=self.question),
            signature="question_encoder", as_dict=True)

            self.response_embeddings = embed(
            dict(input=self.response,
                context=self.response_context),
            signature="response_encoder", as_dict=True)

            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()

        # Initialize session.
        self.session = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True))
        self.session.run(init_op)
    
    def predict(self, text, type='response'):
        if type=='query':
            return self.session.run(self.question_embeddings, feed_dict={self.question:text})['outputs']
        elif type=='response':
            return self.session.run(self.response_embeddings, feed_dict={
            self.response:text,
            self.response_context:text
            })['outputs']
        else: print('Type of prediction not defined')

    def close(self):
        self.session.close()

class USEModel:
    def __init__(self):
        g=tf.Graph()
        with g.as_default():
            embed = hub.Module("./google_use")
            self.statement = tf.placeholder(dtype=tf.string, shape=[None]) #text input
            self.embeddings = embed(
                dict(text=self.statement),
                as_dict=True
            )
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()
        self.session = tf.Session(graph=g)
        self.session.run(init_op)
        
    def predict(self, text):
        return self.session.run(self.embeddings, feed_dict={self.statement:text})['default']

    def close(self):
        self.session.close()

class InferSent:
    def __init__(self):
        from InferSent.models import InferSent
        import torch
        V = 1
        MODEL_PATH = 'encoder/infersent%s.pkl' % V
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
        self.infersent = InferSent(params_model)
        self.infersent.load_state_dict(torch.load(MODEL_PATH))
        W2V_PATH = 'fastText/crawl-300d-2M.vec'
        self.infersent.set_w2v_path(W2V_PATH)
    
    def build_vocab(self, queries):
        self.infersent.build_vocab(queries, tokenize=True)
    
    def update_vocab(self, text):
        self.infersent.update_vocab(text, tokenize=True)

    def predict(self, text):
        # self.update_vocab(text)
        return self.infersent.encode(text, tokenize=True)
        
