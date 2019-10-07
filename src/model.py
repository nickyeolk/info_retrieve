import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece
from .metric_learning import triplet_loss
from tensorflow.train import Saver
from .utils import split_txt, read_txt
from sklearn.metrics.pairwise import cosine_similarity

class GoldenRetriever:
    """GoldenRetriever model for information retrieval prediction and finetuning.
    """
    
    def __init__(self, lr=0.6, margin=0.3, loss='cosine'):
        """        
        Parameters
        ----------
        lr: Learning rate (default 0.6)
        loss: loss function to use. Options are 'cosine'(default), or 'triplet' which is a triplet loss based on cosine distance.
        margin: margin to be used if loss='triplet' (default 0.1)

        Example:
        >>> gr = GoldenRetriever()
        >>> text_list = ['I love my chew toy!', 'I hate Mondays.']
        >>> gr.load_kb(text_list=text_list)
        >>> gr.make_query('what do you not love?', top_k=1)
        ['I hate Mondays.']
        """
        self.v=v=['module/QA/Final/Response_tuning/ResidualHidden_1/dense/kernel','module/QA/Final/Response_tuning/ResidualHidden_0/dense/kernel', 'module/QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']
        self.lr = lr
        self.margin = margin
        self.loss = loss
        # Set up graph.
        tf.reset_default_graph() # finetune
        g = tf.Graph()
        with g.as_default():
            self.embed = hub.Module("./google_use_qa", trainable=True)
            # put placeholders
            self.question = tf.placeholder(dtype=tf.string, shape=[None])  # question
            self.response = tf.placeholder(dtype=tf.string, shape=[None])  # response
            self.response_context = tf.placeholder(dtype=tf.string, shape=[None])  # response context
            self.neg_response = tf.placeholder(dtype=tf.string, shape=[None])  # response
            self.neg_response_context = tf.placeholder(dtype=tf.string, shape=[None])  # response context
            self.label = tf.placeholder(tf.int32, [None], name='label')
            
            self.question_embeddings = self.embed(
            dict(input=self.question),
            signature="question_encoder", as_dict=True)

            self.response_embeddings = self.embed(
            dict(input=self.response,
                context=self.response_context),
            signature="response_encoder", as_dict=True)

            self.neg_response_embeddings = self.embed(
            dict(input=self.neg_response,
                context=self.neg_response_context),
            signature="response_encoder", as_dict=True)

            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

            # finetune
            # tf 1.13 does not have contrastive loss. Might have to self-implement.
            if self.loss=='triplet':
                self.cost = triplet_loss(self.question_embeddings['outputs'], self.response_embeddings['outputs'], self.neg_response_embeddings['outputs'], margin=self.margin)
            elif self.loss=='cosine':
                self.cost = tf.losses.cosine_distance(self.question_embeddings['outputs'], self.response_embeddings['outputs'], axis=1)
            else: raise NotImplementedError('invalid loss selected. Choose either triplet or cosine.')
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            var_finetune=[x for x in self.embed.variables for vv in self.v if vv in x.name] #get the weights we want to finetune.
            self.opt_op = opt.minimize(self.cost, var_list=var_finetune)
        # g.finalize()

        # Initialize session.
        self.session = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True))
        self.session.run(init_op)
        print('model initiated!')
    
    def predict(self, text, context=None, type='response'):
        """Return the tensor representing embedding of input text."""
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

    def finetune(self, question, answer, context, neg_answer=[], neg_answer_context=[]):
        current_loss = self.session.run(self.cost, feed_dict={
            self.question:question,
            self.response:answer,
            self.response_context:context,
            self.neg_response:neg_answer,
            self.neg_response_context:neg_answer_context
            })
        self.session.run(self.opt_op, feed_dict={
            self.question:question,
            self.response:answer,
            self.response_context:context,
            self.neg_response:neg_answer,
            self.neg_response_context:neg_answer_context
            })
        return current_loss
        
    def export(self, savepath='fine_tuned', step=0):
        '''Path should include partial filename.'''
        saver=Saver(self.embed.variables)
        saver.save(self.session, savepath, global_step=step)
    
    def restore(self, savepath):
        saver=Saver(self.embed.variables)
        saver.restore(self.session, savepath)

    def close(self):
        self.session.close()

    def load_kb(self, path_to_kb=None, text_list=None, is_faq=False):
        """Give either path to .txt document or list of clauses.
        For text document, each clause is separated by 2 newlines (\n)"""
        if text_list:
            self.text = text_list
        else:
            if is_faq:
                self.text, self.questions = split_txt(read_txt(path_to_kb), is_faq)
            else:
                self.text = split_txt(read_txt(path_to_kb), is_faq)
        self.vectorized_knowledge = self.predict(self.text, type='response')

    def make_query(self, querystring, top_k=5, index=False):
        """choose index=True to return sorted index of matches"""
        similarity_score=cosine_similarity(self.vectorized_knowledge, self.predict([querystring], type='query'))
        sortargs=np.flip(similarity_score.argsort(axis=0))
        sortargs=[x[0] for x in sortargs]
        sorted_ans=[self.text[i] for i in sortargs]
        if index:
            return sorted_ans[:top_k], sortargs[:top_k]
        return sorted_ans[:top_k], similarity_score



'''Unused models below'''
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
        
