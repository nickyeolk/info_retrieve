import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import datetime
import tensorflow_text
from .utils import split_txt, read_txt, clean_txt, read_kb_csv
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.optimizers import Adam
from transformers import AlbertTokenizer, TFAlbertModel
from .bert_tokenization import FullTokenizer, get_masks, get_segments, get_ids, truncate_str, preprocess_one_str, preprocess_str
from .metric_learning import triplet_loss


class GoldenRetriever:
    """
    GoldenRetriever model for information retrieval prediction and finetuning.


    Example:
    >>> gr = GoldenRetriever()
    >>> text_list = ['I love my chew toy!', 'I hate Mondays.']
    >>> gr.load_kb(text_list=text_list)
    >>> gr.make_query('what do you not love?', top_k=1)
    ['I hate Mondays.']
    
    """
     
    def __init__(self, max_seq_length=512):
        """
        initialize the model. load google USE embedding
   
        """
        # Not used for USE
        self.max_seq_length = max_seq_length

        # self.v=['QA/Final/Response_tuning/ResidualHidden_1/dense/kernel','QA/Final/Response_tuning/ResidualHidden_0/dense/kernel', 'QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']
        self.v=['QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']
        self.vectorized_knowledge = {}
        self.text = {}
        self.questions = {}
        self.opt_params = {'learning_rate':0.001,'beta_1':0.9,'beta_2':0.999,'epsilon':1e-07}

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
        """
        Return the tensor representing embedding of input text.
        
        Parameters:
        type(string) : can be 'query' or 'response'

        Returns:
        representing embedding of input text

        """
        if type=='query':
            encoded_queries = [self.question_encoder(tf.constant([t]))['outputs'] for t in text]
            encoded_queries_tensor = tf.concat(encoded_queries, axis=0)
            return encoded_queries_tensor
            # return self.session.run(self.question_embeddings, feed_dict={self.question:text})['outputs']
        elif type=='response':
            if not context:
                context = text
            
            encoded_responses = [self.response_encoder(input=tf.constant(t),
                                 context=tf.constant(c))['outputs'] for t, c in zip(text, context)]
            encoded_responses_tensor = tf.concat(encoded_responses, axis=0)
            return encoded_responses_tensor
        else: print('Type of prediction not defined')
        
    def make_query(self, querystring, top_k=5, index=False, predict_type='query', kb_name='default_kb'):
        """
        Make a query against the stored vectorized knowledge. 
        
        Parameters:
        type(string): can be 'query' or 'response'. Use to compare statements
        kb_name(string): the name of knowledge base in the knowledge dictionary
        index(boolean): Choose index=True to return sorted index of matches. 

        Returns:
        return the top K vectorized answers and their scores

        """
        similarity_score=cosine_similarity(self.vectorized_knowledge[kb_name], self.predict([querystring], type=predict_type))
        sortargs=np.flip(similarity_score.argsort(axis=0))
        sortargs=[x[0] for x in sortargs]
        sorted_ans=[self.text[kb_name][i] for i in sortargs]
        if index:
            return sorted_ans[:top_k], sortargs[:top_k]
        return sorted_ans[:top_k], similarity_score[sortargs[:top_k]] 
        
        
    def finetune(self, question, answer, margin=0.3, loss='triplet', context=[], neg_answer=[], neg_answer_context=[], label=[]):
        """
        Finetune the model

        Parameters:
        loss(string): loss function can be 'triplet', 'cosine' and 'contrastive'

        """
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

    def load_kb(self, kb_):
        """
        Give either path to .txt document or list of clauses.
        For text document, each clause is separated by 2 newlines ('\\n\\n')
        
        Parameters:
        is_faq(boolean): can be in the format of FAQ. 

        Returns:
        create the knowledge base

        """
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

        """
        load the document in csv format

        """
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


class GoldenRetriever_BERT:
    """GoldenRetriever model for information retrieval prediction and finetuning.


    Example:
    >>> gr = GoldenRetriever()
    >>> text_list = ['I love my chew toy!', 'I hate Mondays.']
    >>> gr.load_kb(text_list=text_list)
    >>> gr.make_query('what do you not love?', top_k=1)
    ['I hate Mondays.']
    """
    
    
    def __init__(self, max_seq_length=512):

        # BERT unique params
        self.max_seq_length = max_seq_length

        # GR params
        self.vectorized_knowledge = {}
        self.text = {}
        self.questions = {}
        self.opt_params = {'learning_rate':0.001,'beta_1':0.9,'beta_2':0.999,'epsilon':1e-07}

        # init saved model
        # self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)  # uncased and smaller model
        self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)
        self.vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()

        # writing the model for the training tasks
        # get inputs 
        res_id = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_ids", dtype='int32')
        res_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_masks", dtype='int32')
        res_segment = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_seg", dtype='int32')

        # encode the three inputs
        res_pooled, res_seq = self.bert_layer([res_id, res_mask, res_segment])

        # dense layer specifically for 
        self.response_encoder = tf.keras.layers.Dense(768, input_shape=(768,), name='response_dense_layer')
        encoded_response = self.response_encoder(res_pooled)

        # init model
        self.bert_model = tf.keras.Model(inputs=[res_id, res_mask, res_segment],
                                    outputs=encoded_response)
        
        print("Downloaded model from Hub, initializing tokenizer and optimizer")
        self.init_signatures()


    def init_signatures(self):
        """
        Re-init references to layers and model attributes        
        When restoring the model, the references to the vocab file / layers would be lost.
        """
        # init tokenizer from hub layer
        self.tokenizer = FullTokenizer(self.vocab_file, self.do_lower_case)
        
        # init optimizer
        self.optimizer = tf.keras.optimizers.Adam(**self.opt_params)
        self.cost_history = []

        # bert layer name
        self.bert_layer_name = [layer.name for layer in self.bert_model.layers if layer.name.startswith('keras_layer')][0]

        # TF-Hub page recommentds finetuning all weights
        # "All parameters in the module are trainable, 
        # and fine-tuning all parameters is the recommended practice."
        self.var_finetune=self.bert_model.variables
        
        print('model initiated!')
        
        
    def predict(self, text, type='response'):
        """
        Return the tensor representing embedding of input text.
        Type can be 'query' or 'response' 
        
        args:
            text: (str or iterable of str) This contains the text that is required to be encoded
            type: (str) Either 'response' or 'query'. Default is 'response'. 
                        This tells GR to either use the response or query encoder
                        but in the case of BERT, this argument is ignored

        Return:
            pooled_embedding: (tf.tensor) contains the 768 dim encoding of the input text
        """
        preprocessed_inputs = preprocess_str(text, self.max_seq_length, self.tokenizer)
        pooled_embedding, _ = self.bert_model.get_layer( self.bert_layer_name )(preprocessed_inputs)
        return pooled_embedding
        
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
        
        
    def finetune(self, question, answer, margin=0.3, loss='triplet', context=[], neg_answer=[], neg_answer_context=[], label=[]):
        """
        Finetune model with GradientTape.

        args:
            question: (str or iterable of str) This contains the questions that is required to be encoded
            response: (str or iterable of str) This contains the response that is required to be encoded
            margin: (float) margin tuning parameter for triplet / contrastive loss
            loss: (str) name of loss function. ('cosine', 'contrastive', 'triplet'). Default setting is 'triplet.
            context: (str or iterable of str) Ignored for BERT/ALBERT
            neg_answer: (str or iterable of str) This contains the distractor responses that is required to be encoded
            neg_answer_context: (str or iterable of str) Ignored for BERT/ALBERT
            label: (list of int) This contain the label for contrastive loss

        return:
            loss_value: (float) This returns the loss of the training task

        """
        question_id_mask_seg = preprocess_str(question, self.max_seq_length, self.tokenizer)
        response_id_mask_seg = preprocess_str(answer, self.max_seq_length, self.tokenizer)
        
        # for eager execution finetuning
        with tf.GradientTape() as tape:
            
            # tf-hub's keras layer can take the lists directly
            # but the bert_model object needs the inputs to be tf.constants
            question_embeddings, q_sequence_output = self.bert_model.get_layer( self.bert_layer_name )(question_id_mask_seg)
            response_embeddings = self.bert_model([tf.constant(response_id_mask_seg[0]),
                                                    tf.constant(response_id_mask_seg[1]),
                                                    tf.constant(response_id_mask_seg[2]),
                                                ])
                    
            
            if loss == 'cosine':
                self.cost = tf.keras.losses.CosineSimilarity(axis=1)
                cost_value = self.cost(question_embeddings ,response_embeddings)
                
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
                # encode the negative response
                neg_answer_id_mask_seg = preprocess_str(neg_answer, self.max_seq_length, self.tokenizer)
                neg_response_embeddings = self.bert_model([tf.constant(neg_answer_id_mask_seg[0]),
                                                            tf.constant(neg_answer_id_mask_seg[1]),
                                                            tf.constant(neg_answer_id_mask_seg[2])
                                                            ])
                
                cost_value = triplet_loss(question_embeddings, response_embeddings, neg_response_embeddings)
                
            # record loss     
            self.cost_history.append(cost_value.numpy().mean())
        
        # apply gradient
        self.grads = tape.gradient(cost_value, self.var_finetune)
        self.optimizer.apply_gradients(zip(self.grads, self.var_finetune))

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
        
    def export(self, savepath='gr_bert.hdf5'):
        '''
        Save the BERT model into a directory

        The original saving procedure taken from:
        https://github.com/tensorflow/models/blob/master/official/nlp/bert/export_tfhub.py
        The tf-hub module does includes the str vocab_file directory and do_lower_case boolean
        The future restore() function should depend on a fresh copy of the vocab file,
        because loading the model in a different directory demands a different directory for the vocab.
        However, the string vocab_file directory and do_lower_case boolean is kept to the saved model anyway
        '''

        self.bert_model.vocab_file = self.vocab_file
        self.bert_model.do_lower_case = self.do_lower_case

        self.bert_model.save(savepath, include_optimizer=False, save_format="h5")

    def restore(self, savepath='gr_bert.hdf5'):
        """
        Load saved model from savepath
        
        hub.KerasLayer is unrecognized by tf.keras' save and load_model methods.
        The trick is to feed a custom_objects dict object
        This solution given by qo-o-op in this link:
        https://github.com/tensorflow/tensorflow/issues/26835

        Args:
            savepath: (str) dir path of the 
        """
        self.bert_model = tf.keras.models.load_model(savepath, custom_objects={'KerasLayer':hub.KerasLayer})
        self.init_signatures()



class GoldenRetriever_ALBERT:
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
    
    
    def __init__(self, max_seq_length=512):

        # ALBERT unique params
        self.max_seq_length = max_seq_length

        # GR params
        self.vectorized_knowledge = {}
        self.text = {}
        self.questions = {}
        self.opt_params = {'learning_rate':0.001,'beta_1':0.9,'beta_2':0.999,'epsilon':1e-07}

        # init saved model
        self.albert_layer = TFAlbertModel.from_pretrained('albert-base-v2')

        # writing the model for the training tasks
        # get inputs
        
        res_id = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_ids", dtype='int32')
        res_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_masks", dtype='int32')
        res_segment = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_seg", dtype='int32')

        # encode the three inputs
        _, res_pooled = self.albert_layer([res_id, res_mask, res_segment])

        # dense layer specifically for 
        self.response_encoder = tf.keras.layers.Dense(768, input_shape=(768,), name='response_dense_layer')
        encoded_response = self.response_encoder(res_pooled)

        # init model
        self.albert_model = tf.keras.Model(inputs=[res_id, res_mask, res_segment],
                                    outputs=encoded_response)
        
        print("Initializing tokenizer and optimizer")
        self.init_signatures()

    def init_signatures(self):
        """
        Re-init references to layers and model attributes        
        When restoring the model, the references to the vocab file / layers would be lost.
        """
        
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        # init optimizer
        self.optimizer = tf.keras.optimizers.Adam(**self.opt_params)
        self.cost_history = []
        
        # TF-Hub page recommentds finetuning all weights
        # "All parameters in the module are trainable, 
        # and fine-tuning all parameters is the recommended practice."
        self.var_finetune=self.albert_model.variables

        print('model initiated!')

        
    def _predict_one_str(self, text, type='response'):
        """
        Return the tensor representing embedding of input text.
        Type can be 'query' or 'response' 
        
        args:
            text: (str or iterable of str) This contains the text that is required to be encoded
            type: (str) Either 'response' or 'query'. Default is 'response'. 
                        This tells GR to either use the response or query encoder
                        but in the case of BERT, this argument is ignored

        Return:
            pooled_embedding: (tf.tensor) contains the 768 dim encoding of the input text
        """
        
        # preprocessed_inputs = np.array(preprocess_str(text, self.max_seq_length, self.tokenizer))
        preprocessed_inputs = self.tokenizer.encode_plus(text, max_length=self.max_seq_length, pad_to_max_length=True, add_special_tokens=True, return_tensors="tf")
        pooled_embedding = self.albert_layer(preprocessed_inputs)[1]
        return pooled_embedding

    def predict(self, text, type='response'):
        encoded_strings = [self._predict_one_str(t) for t in text]
        encoded_responses_tensor = tf.concat(encoded_strings, axis=0)
        return encoded_responses_tensor

    
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
        
        
    def finetune(self, question, answer, margin=0.3, loss='triplet', context=[], neg_answer=[], neg_answer_context=[], label=[]):
        """
        Finetune model with GradientTape.

        args:
            question: (str or iterable of str) This contains the questions that is required to be encoded
            answer: (str or iterable of str) This contains the response that is required to be encoded
            margin: (float) margin tuning parameter for triplet / contrastive loss
            loss: (str) name of loss function. ('cosine', 'contrastive', 'triplet'). Default setting is 'triplet.
            context: (str or iterable of str) Ignored for BERT/ALBERT
            neg_answer: (str or iterable of str) This contains the distractor responses that is required to be encoded
            neg_answer_context: (str or iterable of str) Ignored for BERT/ALBERT
            label: (list of int) This contain the label for contrastive loss

        return:
            loss_value: (float) This returns the loss of the training task

        """
        question_id_mask_seg = preprocess_str(question, self.max_seq_length, self.tokenizer)
        response_id_mask_seg = preprocess_str(answer, self.max_seq_length, self.tokenizer)
        
        # for eager execution finetuning
        with tf.GradientTape() as tape:
            
            # tf-hub's keras layer can take the lists directly
            # but the bert_model object needs the inputs to be tf.constants
            question_embeddings = self.albert_layer([tf.constant(question_id_mask_seg[0]),
                                                     tf.constant(question_id_mask_seg[1]),
                                                     tf.constant(question_id_mask_seg[2]),
                                                ])[1]

            response_embeddings = self.albert_model([tf.constant(response_id_mask_seg[0]),
                                                     tf.constant(response_id_mask_seg[1]),
                                                     tf.constant(response_id_mask_seg[2]),
                                                ])
                    
            
            if loss == 'cosine':
                self.cost = tf.keras.losses.CosineSimilarity(axis=1)
                cost_value = self.cost(question_embeddings ,response_embeddings)
                
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
                # encode the negative response
                neg_answer_id_mask_seg = preprocess_str(neg_answer, self.max_seq_length, self.tokenizer)
                neg_response_embeddings = self.albert_model([tf.constant(neg_answer_id_mask_seg[0]),
                                                            tf.constant(neg_answer_id_mask_seg[1]),
                                                            tf.constant(neg_answer_id_mask_seg[2])
                                                            ])

                cost_value = triplet_loss(question_embeddings, response_embeddings, neg_response_embeddings)
                
            # record loss     
            self.cost_history.append(cost_value.numpy().mean())
        
        # apply gradient
        self.grads = tape.gradient(cost_value, self.var_finetune)
        self.optimizer.apply_gradients(zip(self.grads, self.var_finetune))

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


    def export(self, savepath='./model/albert/gr_albert.hdf5'):
        '''
        Save the BERT model into a directory

        The original saving procedure taken from:
        https://github.com/tensorflow/models/blob/master/official/nlp/bert/export_tfhub.py
        The tf-hub module does includes the str vocab_file directory and do_lower_case boolean
        The future restore() function should depend on a fresh copy of the vocab file,
        because loading the model in a different directory demands a different directory for the vocab.
        However, the string vocab_file directory and do_lower_case boolean is kept to the saved model anyway
        '''
        # model.save does not work if there are layers that are subclassed (eg. huggingface models)

        self.albert_model.save_weights(savepath)


    def restore(self, savepath='./model/albert/gr_albert.hdf5'):
        """
        Load weights from savepath
        
        Args:
            savepath: (str) dir path of the weights
        """
        
        self.albert_model.load_weights(savepath)

