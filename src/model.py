import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import numpy as np
import datetime
import tensorflow_text
from .utils import split_txt, read_txt, clean_txt, read_kb_csv
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.optimizers import Adam

from .metric_learning import triplet_loss


class GoldenRetriever:
    """GoldenRetriever model for information retrieval prediction and finetuning.
    Parameters
    ----------
    **kwargs: keyword arguments for Adam() optimizer

    Example:
    >>> gr = GoldenRetriever()
    >>> text_list = ['I love my chew toy!', 'I hate Mondays.']
    >>> gr.load_kb(text_list=text_list)
    >>> gr.make_query('what do you not love?', top_k=1)
    ['I hate Mondays.']
    """
    
    
    def __init__(self, **kwargs):
        # self.v=['QA/Final/Response_tuning/ResidualHidden_1/dense/kernel','QA/Final/Response_tuning/ResidualHidden_0/dense/kernel', 'QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']
        self.v=['QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']
        self.vectorized_knowledge = {}
        self.text = {}
        self.questions = {}
        self.opt_params = kwargs

        # init saved model
        self.embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/2')
        self.init_signatures()


    def init_signatures(self):
        # re-initialize the references to the model signatures
        self.question_encoder = self.embed.signatures['question_encoder']
        self.response_encoder = self.embed.signatures['response_encoder']
        self.neg_response_encoder = self.embed.signatures['response_encoder']
        print('model initiated!')

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(**self.opt_params)
        self.cost_history = []
        self.var_finetune=[x for x in self.embed.variables for vv in self.v if vv in x.name] #get the weights we want to finetune.
        
        
    def predict(self, text, context=None, type='response'):
        """Return the tensor representing embedding of input text.
        Type can be 'query' or 'response' """
        if type=='query':
            return self.question_encoder(tf.constant([text]))['outputs']
            # return self.session.run(self.question_embeddings, feed_dict={self.question:text})['outputs']
        elif type=='response':
            if not context:
                context = text
            return self.response_encoder(input=tf.constant(text),
                                         context=tf.constant(context))['outputs']
        else: print('Type of prediction not defined')
        
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
        
        
    def finetune(self, question, answer, context, margin=0.3, loss='triplet', neg_answer=[], neg_answer_context=[], label=[]):

        with tf.GradientTape() as tape:
            # get encodings
            question_embeddings = self.question_encoder(tf.constant(question))['outputs']
            response_embeddings = self.response_encoder(input=tf.constant(answer), 
                                                        context=tf.constant(context))['outputs']

            if loss == 'cosine':
                """
                # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity
                """
                self.cost = tf.keras.losses.CosineSimilarity(axis=1)
                cost_value = self.cost(question_embeddings, response_embeddings)
                
            elif loss == 'contrastive':
                """
                https://www.tensorflow.org/addons/api_docs/python/tfa/losses/ContrastiveLoss
                
                y_true to be a vector of binary labels
                y_hat to be the respective distances
                """
                self.cosine_dist = tf.keras.losses.CosineSimilarity(axis=1)
                cosine_dist_value = self.cosine_dist(question_embeddings, response_embeddings)
                
                self.cost = tfa.losses.contrastive.ContrastiveLoss(margin = margin)
                cost_value = self.cost(label, cosine_dist_value)
                
            elif loss == 'triplet':
                """
                Triplet loss uses a non-official self-implementated loss function outside of TF based on cosine distance
                """
                neg_response_embeddings = self.neg_response_encoder(input=tf.constant(neg_answer), 
                                                                    context=tf.constant(neg_answer_context))['outputs']
                cost_value = triplet_loss(question_embeddings, response_embeddings, neg_response_embeddings)

                
        # record loss     
        self.cost_history.append(cost_value.numpy().mean())
        
        # apply gradient
        grads = tape.gradient(cost_value, self.var_finetune)
        self.optimizer.apply_gradients(zip(grads, self.var_finetune))

        return cost_value.numpy().mean()
        
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
            delim = '\n'
            self.text[kb_name] = split_txt([front+delim for front in raw_text.split('\n')])
        else: raise NameError('invalid kb input!')
        self.vectorized_knowledge[kb_name] = self.predict(clean_txt(self.text[kb_name]), type='response')
        print('knowledge base lock and loaded!')
        
    def load_csv_kb(self, path_to_kb=None, kb_name='default_kb', meta_col='meta', answer_col='answer', 
                    query_col='question', answer_str_col='answer', cutoff=None):
        self.text[kb_name], self.questions[kb_name] = read_kb_csv(path_to_kb, meta_col=meta_col, answer_col=answer_col, 
                            query_col=query_col, answer_str_col=answer_str_col, cutoff=None)
        self.vectorized_knowledge[kb_name] = self.predict(clean_txt(self.text[kb_name]), type='response')
        print('knowledge base (csv) lock and loaded!')
        
    def export(self, savepath='fine_tuned'):
        '''
        Path should include partial filename.
        https://www.tensorflow.org/api_docs/python/tf/saved_model/save
        '''
        tf.saved_model.save(self.embed, savepath, signatures={
                                                                'default': self.embed.signatures['default'],
                                                                'response_encoder':self.embed.signatures['response_encoder'],
                                                                'question_encoder':self.embed.signatures['question_encoder']  
                                                                })

    def restore(self, savepath):
        """
        Signatures need to be re-init after weights are loaded.
        """
        self.embed = tf.saved_model.load(savepath)
        self.init_signatures()







'''Unused models below'''
# class _USEModel:
#     def __init__(self):
#         g=tf.Graph()
#         with g.as_default():
#             embed = hub.Module("./google_use")
#             self.statement = tf.placeholder(dtype=tf.string, shape=[None]) #text input
#             self.embeddings = embed(
#                 dict(text=self.statement),
#                 as_dict=True
#             )
#             init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
#         g.finalize()
#         self.session = tf.Session(graph=g)
#         self.session.run(init_op)
        
#     def predict(self, text):
#         return self.session.run(self.embeddings, feed_dict={self.statement:text})['default']

#     def close(self):
#         self.session.close()

# class _InferSent:
#     def __init__(self):
#         from InferSent.models import InferSent
#         import torch
#         V = 1
#         MODEL_PATH = 'encoder/infersent%s.pkl' % V
#         params_model = {'bsize': 256, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
#                         'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
#         self.infersent = InferSent(params_model)
#         self.infersent.load_state_dict(torch.load(MODEL_PATH))
#         W2V_PATH = 'fastText/crawl-300d-2M.vec'
#         self.infersent.set_w2v_path(W2V_PATH)
    
#     def build_vocab(self, queries):
#         self.infersent.build_vocab(queries, tokenize=True)
    
#     def update_vocab(self, text):
#         self.infersent.update_vocab(text, tokenize=True)

#     def predict(self, text):
#         # self.update_vocab(text)
#         return self.infersent.encode(text, tokenize=True)
        

