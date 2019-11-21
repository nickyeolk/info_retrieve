import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece
from metric_learning import triplet_loss, contrastive_loss
from tensorflow.train import Saver
from utils import split_txt, read_txt, clean_txt, read_kb_csv
from sklearn.metrics.pairwise import cosine_similarity

class GoldenRetriever:
    """GoldenRetriever model for information retrieval prediction and finetuning.
    Parameters
    ----------
    lr: Learning rate (default 0.6)
    loss: loss function to use. Options are 'cosine'(default), 'contrastive', or 'triplet' which is a triplet loss based on cosine distance.
    margin: margin to be used if loss='triplet' (default 0.1)

    Example:
    >>> gr = GoldenRetriever()
    >>> text_list = ['I love my chew toy!', 'I hate Mondays.']
    >>> gr.load_kb(text_list=text_list)
    >>> gr.make_query('what do you not love?', top_k=1)
    ['I hate Mondays.']
    """
    
    def __init__(self, lr=0.6, margin=0.3, loss='triplet'):
        # self.v=['module/QA/Final/Response_tuning/ResidualHidden_1/dense/kernel','module/QA/Final/Response_tuning/ResidualHidden_0/dense/kernel', 'module/QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']
        self.v=['module/QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']
        self.lr = lr
        self.margin = margin
        self.loss = loss
        self.vectorized_knowledge = {}
        self.text = {}
        self.questions = {}
        # Set up graph.
        tf.reset_default_graph() # finetune
        g = tf.Graph()
        with g.as_default():
            # self.embed = hub.Module("./google_use_qa", trainable=True)
            self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1", trainable=True)
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
            elif self.loss=='contrastive':
                self.cost = contrastive_loss(self.label, self.question_embeddings['outputs'], self.response_embeddings['outputs'], margin=self.margin)
            else: raise NotImplementedError('invalid loss selected. Choose either triplet or cosine.')
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            var_finetune=[x for x in self.embed.variables for vv in self.v if vv in x.name] #get the weights we want to finetune.
            self.opt_op = opt.minimize(self.cost, var_list=var_finetune)
        # g.finalize()

        # Initialize session.
        self.session = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=False))
        self.session.run(init_op)
        print('model initiated!')
    
    def predict(self, text, context=None, type='response'):
        """Return the tensor representing embedding of input text.
        Type can be 'query' or 'response' """
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

    def finetune(self, question, answer, context, neg_answer=[], neg_answer_context=[], label=[]):
        current_loss = self.session.run(self.cost, feed_dict={
            self.question:question,
            self.response:answer,
            self.response_context:context,
            self.neg_response:neg_answer,
            self.neg_response_context:neg_answer_context,
            self.label:label
            })
        self.session.run(self.opt_op, feed_dict={
            self.question:question,
            self.response:answer,
            self.response_context:context,
            self.neg_response:neg_answer,
            self.neg_response_context:neg_answer_context,
            self.label:label
            })
        return current_loss
        
    def export(self, savepath='fine_tuned', step=0):
        '''Path should include partial filename.'''
        saver=Saver(self.embed.variables)
        saver.save(self.session, savepath, global_step=step)
    
    def restore(self, savepath):
        saver=Saver(self.embed.variables)
        saver.restore(self.session, savepath)
        print('model checkpoint restored!')

    def close(self):
        self.session.close()

    def load_kb(self, path_to_kb=None, text_list=None, question_list=None, 
                raw_text=None, is_faq=False, kb_name='default_kb'):
        r"""Give either path to .txt document or list of clauses.
        For text document, each clause is separated by 2 newlines ('\\n\\n')"""
        if text_list:
            self.text[kb_name] = text_list
            if is_faq:
                self.questions[kb_name] = question_list
        elif path_to_kb:
            if is_faq:
                self.text[kb_name], self.questions[kb_name] = split_txt(read_txt(path_to_kb), is_faq)
            else:
                self.text[kb_name] = split_txt(read_txt(path_to_kb), is_faq)
        elif raw_text:
            self.text[kb_name] = split_txt(raw_text)
        else: raise NameError('invalid kb input!')
        self.vectorized_knowledge[kb_name] = self.predict(clean_txt(self.text[kb_name]), type='response')
        print('knowledge base lock and loaded!')

    def load_csv_kb(self, path_to_kb=None, kb_name='default_kb', meta_col='meta', answer_col='answer', 
                    query_col='question', answer_str_col='answer', cutoff=None):
        self.text[kb_name], self.questions[kb_name] = read_kb_csv(path_to_kb, meta_col=meta_col, answer_col=answer_col, 
                            query_col=query_col, answer_str_col=answer_str_col, cutoff=None)
        self.vectorized_knowledge[kb_name] = self.predict(clean_txt(self.text[kb_name]), type='response')
        print('knowledge base (csv) lock and loaded!')

    def make_query(self, querystring, top_k=5, index=False, predict_type='query', kb_name='default_kb'):
        """Make a query against the stored vectorized knowledge. 
        Choose index=True to return sorted index of matches.
        type can be 'query' or 'response' if you are comparing statements
        """
        similarity_score=cosine_similarity(self.vectorized_knowledge[kb_name], self.predict([querystring], type=predict_type))
        sortargs=np.flip(similarity_score.argsort(axis=0))
        sortargs=[x[0] for x in sortargs]
        sorted_ans=[self.text[kb_name][i] for i in sortargs]
        if index:
            return sorted_ans[:top_k], sortargs[:top_k]
        return sorted_ans[:top_k], similarity_score[sortargs[:top_k]]



'''Unused models below'''
class _USEModel:
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

class _InferSent:
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
        
