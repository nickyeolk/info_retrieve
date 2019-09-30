import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece
from .metric_learning import triplet_loss

class QnaEncoderModel:
    def __init__(self, lr=0.1, margin=0.1, loss='triplet'):
        '''loss can be 'triplet'(default), or 'cosine'. '''
        self.v=v=['module/QA/Final/Response_tuning/ResidualHidden_1/dense/kernel','module/QA/Final/Response_tuning/ResidualHidden_0/dense/kernel', 'module/QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']
        self.lr = lr
        self.margin = margin
        self.loss = loss
        # Set up graph.
        tf.reset_default_graph() # finetune
        g = tf.Graph()
        with g.as_default():
            embed = hub.Module("./google_use_qa", trainable=True)
            # put placeholders
            self.question = tf.placeholder(dtype=tf.string, shape=[None])  # question
            self.neg_question = tf.placeholder(dtype=tf.string, shape=[None])  # question
            self.response = tf.placeholder(dtype=tf.string, shape=[None])  # response
            self.response_context = tf.placeholder(dtype=tf.string, shape=[None])  # response context
            self.label = tf.placeholder(tf.int32, [None], name='label')
            
            self.question_embeddings = embed(
            dict(input=self.question),
            signature="question_encoder", as_dict=True)

            self.neg_question_embeddings = embed(
            dict(input=self.neg_question),
            signature="question_encoder", as_dict=True)

            self.response_embeddings = embed(
            dict(input=self.response,
                context=self.response_context),
            signature="response_encoder", as_dict=True)

            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

            # finetune
            # tf 1.13 does not have contrastive loss. Might have to self-implement.
            if self.loss=='triplet':
                self.cost = triplet_loss(self.response_embeddings['outputs'], self.question_embeddings['outputs'], self.neg_question_embeddings['outputs'], margin=self.margin)
            elif self.loss=='cosine':
                self.cost = tf.losses.cosine_distance(self.question_embeddings['outputs'], self.response_embeddings['outputs'], axis=1)
            else: raise Exception('invalid loss selected. Choose either triplet or cosine.')
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            var_finetune=[x for x in embed.variables for vv in self.v if vv in x.name] #get the weights we want to finetune.
            self.opt_op = opt.minimize(self.cost, var_list=var_finetune)
        g.finalize()

        # Initialize session.
        self.session = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True))
        self.session.run(init_op)
    
    def predict(self, text, context=None, type='response'):
        if type=='query':
            return self.session.run(self.question_embeddings, feed_dict={self.question:text})['outputs']
        elif type=='response':
            if not context:
                context = text
            return self.session.run(self.response_embeddings, feed_dict={
            self.response:text,
            self.response_context:context
            })['outputs']
        else: print('Type of prediction not defined')

    def finetune(self, question, answer, context, neg_question=None):
        current_loss = self.session.run(self.cost, feed_dict={
            self.question:question,
            self.neg_question:neg_question, 
            self.response:answer,
            self.response_context:context})
        self.session.run(self.opt_op, feed_dict={
            self.question:question,
            self.neg_question:neg_question, 
            self.response:answer,
            self.response_context:context
            })
        return current_loss
        

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
        params_model = {'bsize': 256, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
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
        
