"""
Finetune goldenretriever on data in SQL

Sample usage:
------------

python -m src.finetune_eval \
    --model_name='USE' \
    --random_seed=42 \
    --train_batch_size=2 \
    --predict_batch_size=2 \
    --learning_rate=5e-5 \
    --beta_1=0.9 \
    --beta_2=0.999 \
    --epsilon=1e-07 \
    --num_epochs=1 \
    --max_seq_length=256 \
    --loss_type='triplet' \
    --margin=0.3 \
    --task_type='train_eval' \
    --early_stopping_steps=5
"""
import tensorflow as tf
import os
import pickle
import datetime
import pandas as pd
import numpy as np
import logging
import random
import sys
sys.path.append('/polyaxon-data/goldenretriever')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import rankdata
from absl import flags, app
from polyaxon_client.tracking import Experiment, get_log_level

from .model import GoldenRetriever, GoldenRetriever_BERT, GoldenRetriever_ALBERT
from .dataloader import kb_train_test_split
from .kb_handler import kb, kb_handler



# For logging
def setup_logging():
    log_level = get_log_level()
    if log_level is None:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)

experiment = Experiment()
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Starting experiment")




FLAGS = flags.FLAGS

# eg. 'albert' or 'bert' or 'USE'
flags.DEFINE_string(
    "model_name", None,
    "The name of the model that needs to be finetuned and evaluated"
)

# eg. 42
flags.DEFINE_integer(
    "random_seed", None,
    "The selected random seed to split train-validation dataset"
)

# eg. 16
flags.DEFINE_integer(
    "train_batch_size", None,
    "The selected random seed to split train-validation dataset"
)

# eg. 16
flags.DEFINE_integer(
    "predict_batch_size", None,
    "The selected random seed to split train-validation dataset"
)

# eg. 5e-5
flags.DEFINE_float(
    "learning_rate", None,
    "The initial learning rate for adam optimizer"
)

# eg. 0.9
flags.DEFINE_float(
    "beta_1", None,
    "beta_1 for adam optimizer"
)

# eg. 0.999
flags.DEFINE_float(
    "beta_2", None,
    "beta_2 for adam optimizer"
)

# eg. 1e-07
flags.DEFINE_float(
    "epsilon", None,
    "epsilon for adam optimizer"
)

# eg. 30
flags.DEFINE_integer(
    "num_epochs", None,
    "The number of epochs for training"
)

# eg. 128 or 256 or 512
flags.DEFINE_integer(
    "max_seq_length", None,
    "The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded."
)

# eg. 'cosine', 'contrastive', 'triplet'
flags.DEFINE_string(
    "loss_type", None,
    "The type of training loss to be used for optimization"
)

# eg. 0.3
flags.DEFINE_float(
    "margin", None,
    "The margin value to be used if loss_type == contrastive or triplet"
)

# eg. train_eval or eval_only
flags.DEFINE_string(
    "task_type", None,
    "To define whether to train or evaluate or both train and evaluate the chosen model"
)

# eg. '/polyaxon-data/model/USE/best/1'
flags.DEFINE_string(
    "eval_model_dir", None,
    "To define whether to train or evaluate or both train and evaluate the chosen model"
)

# eg. 5
flags.DEFINE_integer(
    "early_stopping_steps", None,
    "How many epochs without improvement in loss for early stopping"
)



"""
OTHER UTILITY FUNCTIONS
"""
def _flags_to_file(flag_objs, file_path):
    with open(file_path, 'w+') as f:
        for flag in flag_objs:
            f.write("--" + flag.name + "=" + str(flag.value) + "\n")

def _convert_bytes_to_string(byte):
    return str(byte, 'utf-8')

"""
TRIPLET GENERATORS:
Generates training triplets for triplet loss
"""
def _generate_neg_ans(df, train_dict):
    """
    Generates negative answer from dataframe by randomization
    
    Returns a dict, with keys pointing to each kb, pointing to 
    2 arrays of indices, one of correct answers and one of wrong answers,
    generated randomly

    Sample output:
    --------------
    {'PDPA': [array([ 95,  84,  42, 185, 187, 172, 145,  71,   5,  36,  43, 153,  70,
                    66,  53,  98, 180,  94, 138, 176,  79,  87, 103,  67,  24,   8]),
              array([141, 129, 155,   5, 108, 180,  63,   0, 143, 130,  98, 132,  61,
                     138,  24, 187,  86, 153,  94, 140, 162, 109,  56, 105, 185, 165])],
     'nrf': [array([214, 240, 234, 235, 326, 244, 226, 252, 317, 331, 259, 215, 333,
                    283, 299, 263, 220, 204]),
              array([249, 245, 331, 290, 254, 249, 249, 261, 296, 251, 214, 240, 275,
                     210, 223, 259, 212, 205])]}
    """
    train_dict_with_neg = {}
    random.seed(FLAGS.random_seed)

    for kb, ans_pos_idxs in train_dict.items():
        keys = []
        shuffled_ans_pos_idxs = ans_pos_idxs.copy()
        random.shuffle(shuffled_ans_pos_idxs)
        ans_neg_idxs = shuffled_ans_pos_idxs.copy()

        correct_same_as_wrong = df.loc[ans_neg_idxs, 'processed_string'].values == df.loc[ans_pos_idxs, 'processed_string'].values
        while sum(correct_same_as_wrong) > 0:
            random.shuffle(shuffled_ans_pos_idxs)
            ans_neg_idxs[correct_same_as_wrong] = shuffled_ans_pos_idxs[correct_same_as_wrong]
            correct_same_as_wrong = df.loc[ans_neg_idxs, 'processed_string'].values == df.loc[ans_pos_idxs, 'processed_string'].values

        keys.append(ans_pos_idxs)
        keys.append(np.array(ans_neg_idxs))

        train_dict_with_neg[kb] = keys 
    
    return train_dict_with_neg

def _generate_hard_neg_ans(df, train_dict, model):
    """
    Generates negative answer from dataframe by randomization
    
    Sample output:
    --------------
    {'PDPA': [array([ 95,  84,  42, 185, 187, 172, 145,  71,   5,  36,  43, 153,  70,
                    140, 165,   0,  78, 162,  68, 184, 179,  30, 106,  13,  72,  17,
                    18,  38, 109,  47, 113,  56,  27,  63, 147, 105, 121,   2,  80,
                    182,  61,  49, 135, 193,  91,   4, 100, 141, 129, 159, 132, 108,
                    155, 130,  86,  93, 137, 144,  58,  60, 107, 143, 194,  34,  14,
                    66,  53,  98, 180,  94, 138, 176,  79,  87, 103,  67,  24,   8]),
              array([141, 129, 155,   5, 108, 180,  63,   0, 143, 130,  98, 132,  61,
                     103, 137,  13,  17,  71, 107, 144, 121,  68,  66, 184, 179, 135,
                     113, 194,  58,  53, 193,  34,  42,  78,  60, 106, 182,  72, 172,
                     145, 100, 176,  36, 159,  30,  14,  93,  43,  95,  79,   2,  87,
                       8,  18, 147,  91,  49,   4,  70,  67,  84,  80,  27,  47,  38,
                     138,  24, 187,  86, 153,  94, 140, 162, 109,  56, 105, 185, 165])],
     'nrf': [array([214, 240, 234, 235, 326, 244, 226, 252, 317, 331, 259, 215, 333,
                    318, 276, 267, 251, 329, 257, 261, 243, 245, 203, 337, 255, 287,
                    315, 296, 279, 209, 197, 227, 200, 304, 223, 198, 282, 289, 205,
                    319, 212, 254, 256, 303, 338, 230, 210, 262, 249, 294, 290, 275,
                    283, 299, 263, 220, 204]),
              array([249, 245, 331, 290, 254, 249, 249, 261, 296, 251, 214, 240, 275,
                     294, 319, 337, 215, 197, 200, 257, 289, 203, 282, 252, 315, 317,
                     230, 283, 304, 279, 333, 249, 299, 204, 318, 326, 262, 287, 256,
                     234, 303, 235, 243, 276, 198, 338, 220, 329, 255, 209, 263, 267,
                     210, 223, 259, 212, 205])]}
    """
    train_dict_with_neg = {}
    random.seed(FLAGS.random_seed)

    for kb, ans_pos_idxs in train_dict.items():
        keys = []
        train_df = df.loc[ans_pos_idxs]

        # encodings of all possible answers
        all_possible_answers_in_kb = train_df.processed_string.unique().tolist()
        encoded_all_possible_answers_in_kb = model.predict(all_possible_answers_in_kb, type='response')

        # encodings of train questions
        train_questions = train_df.query_string
        encoded_train_questions = model.predict(train_questions, type='query')

        # get similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(encoded_train_questions, encoded_all_possible_answers_in_kb)

        # get index of correct answers, indexed according to unique answers
        correct_answers = train_df.processed_string.tolist()
        idx_of_correct_answers = [all_possible_answers_in_kb.index(correct_answer) for correct_answer in correct_answers]

        # get second best answer index by kb_df
        ans_neg_idxs = []
        for idx_of_correct_answer, similarity_array in zip(idx_of_correct_answers, similarity_matrix):
            similarity_array[idx_of_correct_answer] = -1
            second_best_answer_idx_in_all_possible_answers = similarity_array.argmax()
            second_best_answer_string = all_possible_answers_in_kb[second_best_answer_idx_in_all_possible_answers]
            second_best_answer_idx_in_kb_df = train_df.loc[train_df.processed_string == second_best_answer_string].index[0]
            ans_neg_idxs.append(second_best_answer_idx_in_kb_df)

        # return a list of correct and close wrong answers
        keys.append(ans_pos_idxs)
        keys.append(np.array(ans_neg_idxs))
        train_dict_with_neg[kb] = keys 
    
    return train_dict_with_neg

def gen(batch_size, query, response, neg_response, shuffle_data=False):
    random.seed(FLAGS.random_seed)
    zip_list = list(zip(query,response,neg_response))

    num_samples = len(query)
    while True:
        if shuffle_data:
            random.shuffle(zip_list)

        for offset in range(0, num_samples, batch_size):
            q_batch = [x[0] for x in zip_list[offset:offset+batch_size]]
            r_batch = [x[1] for x in zip_list[offset:offset+batch_size]]
            neg_r_batch = [x[2] for x in zip_list[offset:offset+batch_size]]
        
            yield(q_batch, r_batch, neg_r_batch)

def random_triplet_generator(df, train_dict):
    train_dict_with_neg = _generate_neg_ans(df, train_dict)
    train_pos_idxs = np.concatenate([v[0] for k,v in train_dict_with_neg.items()], axis=0)
    train_neg_idxs = np.concatenate([v[1] for k,v in train_dict_with_neg.items()], axis=0)

    train_query = df.iloc[train_pos_idxs].query_string.tolist()
    train_response = df.iloc[train_pos_idxs].processed_string.tolist()
    train_neg_response = df.iloc[train_neg_idxs].processed_string.tolist()
    
    print("train batch size: {FLAGS.train_batch_size}")
    train_dataset_loader = gen(FLAGS.train_batch_size, train_query, train_response, train_neg_response, shuffle_data=True)
    
    return train_dataset_loader

def hard_triplet_generator(df, train_dict, model):
    """
    Returns a generator that gives batches of training triplets
    """
    train_dict_with_neg = _generate_hard_neg_ans(df, train_dict, model)
    train_pos_idxs = np.concatenate([v[0] for k,v in train_dict_with_neg.items()], axis=0)
    train_neg_idxs = np.concatenate([v[1] for k,v in train_dict_with_neg.items()], axis=0)

    train_query = df.iloc[train_pos_idxs].query_string.tolist()
    train_response = df.iloc[train_pos_idxs].processed_string.tolist()
    train_neg_response = df.iloc[train_neg_idxs].processed_string.tolist()
    
    train_dataset_loader = gen(FLAGS.train_batch_size, train_query, train_response, train_neg_response, shuffle_data=True)
    
    return train_dataset_loader

"""
METRICS
"""
def mrr(ranks):
    """
    Calculate mean reciprocal rank
    Function taken from: https://github.com/google/retrieval-qa-eval/blob/master/squad_eval.py

    Args:
    -----
        ranks: (list) predicted ranks of the correct responses 
    return:
    -------
        mrr: (float)
    """
    return sum([1/v for v in ranks])/len(ranks)

def recall_at_n(ranks, n=3):
    """
    Calculate recall @ N
    Function taken from: https://github.com/google/retrieval-qa-eval/blob/master/squad_eval.py

    Args:
    -----
        ranks: (list) predicted ranks of the correct responses 
    return:
    -------
        Recall@N: (float)
    """
    num = len([rank for rank in ranks if rank <= n])
    return num / len(ranks)

def get_eval_dict(ranks):
    """
    Score the predicted ranks according to various metricss

    args:
    ----
        ranks: (list) predicted ranks of the correct responses 
    return:
    -------
        eval_dict: (dict) contains the metrics and their respective keys
    """
    eval_dict = {}
    eval_dict['mrr_score'] = mrr(ranks)
    eval_dict['r1_score'] = recall_at_n(ranks, 1)
    eval_dict['r2_score'] = recall_at_n(ranks, 2)
    eval_dict['r3_score'] = recall_at_n(ranks, 3)
    return eval_dict


def eval_model(model, df, test_dict):
    """
    Evalutate golden retriever object

    args:
    ----
        model: goldenretriever object
        df: contains the responses and queries
        test_dict: contains the indices of the test data in df

    return:
    ------
        overall_eval: (pd.DataFrame) contains the metrics
        eval_dict: (dict) of the same metrics

    Sample output:
    -------------
                               mrr_score  r1_score  r2_score  r3_score
    PDPA                        0.640719  0.525424  0.627119  0.720339
    nrf                         0.460211  0.275862  0.482759  0.528736
    critical-illness-insurance  0.329302  0.178571  0.342857       0.4
    other-insurance             0.474588  0.259259  0.444444  0.611111
    Steam_engine                0.689601  0.550388  0.744186  0.775194
    1973_oil_crisis             0.781951   0.65625   0.84375  0.890625
    Across_all_kb               0.551312  0.402027  0.570946  0.636824
    """
    eval_dict = {}

    for kb_name in df.kb_name.unique():

        logger.info(f'\n {datetime.datetime.now()} - Evaluating on {kb_name} \n')
        # dict stores eval metrics and relevance ranks
        eval_kb_dict = {}  
        # test-mask is a int array
        # that chooses specific test questions
        # e.g.  test_mask [True, True, False]
        #       query_idx = [0,1]
        kb_df = df.loc[df.kb_name == kb_name]
        kb_idx = df.loc[df.kb_name == kb_name].index
        test_mask = np.isin(kb_idx, test_dict[kb_name])
        # test_idx_mask = np.arange(len(kb_df))[test_mask]

        # get string queries and responses, unduplicated as a list
        kb_df = kb_df.reset_index(drop=True)
        query_list = kb_df.query_string.tolist()
        response_list_w_duplicates = kb_df.processed_string.tolist()
        response_list = kb_df.processed_string.drop_duplicates().tolist() 
        # this index list is important
        # it lists the index of the correct answer for every question
        # e.g. for 20 questions mapped to 5 repeated answers
        # it has 20 elements, each between 0 and 4
        response_idx_list = [response_list.index(nonunique_response_string) 
                            for nonunique_response_string in response_list_w_duplicates]
        response_idx_list = np.array(response_idx_list)[[test_mask]]
        
        encoded_queries = model.predict(query_list, type='query')
        encoded_responses = model.predict(response_list, type='response')

        # get matrix of shape [Q_test x Responses]
        # this holds the relevance rankings of the responses to each test ques
        test_similarities = cosine_similarity(encoded_queries[test_mask], encoded_responses)
        answer_ranks = test_similarities.shape[-1] - rankdata(test_similarities, axis=1) + 1

        # ranks_to_eval
        ranks_to_eval = [answer_rank[correct_answer_idx] 
                        for answer_rank, correct_answer_idx 
                        in zip( answer_ranks, response_idx_list )]


        # get eval metrics -> eval_kb_dict 
        # store in one large dict -> eval_dict
        eval_kb_dict = get_eval_dict(ranks_to_eval)
        eval_kb_dict['answer_ranks'] = answer_ranks
        eval_kb_dict['ranks_to_eval'] = ranks_to_eval
        eval_dict[kb_name] = eval_kb_dict.copy()


    # overall_eval is a dataframe that 
    # tracks performance across the different knowledge bases
    # but individually
    overall_eval = pd.DataFrame(eval_dict).T.drop(['answer_ranks', 'ranks_to_eval'], axis=1)

    # Finally we get eval metrics for across all different KBs
    correct_answer_ranks_across_kb = []
    for key in eval_dict.keys():
        correct_answer_ranks_across_kb.extend(eval_dict[key]['ranks_to_eval'])
        
    # get eval metrics across all knowledge bases combined
    across_kb_scores = get_eval_dict(correct_answer_ranks_across_kb)
    across_kb_scores_ = {'Across_all_kb':across_kb_scores}
    across_kb_scores_ = pd.DataFrame(across_kb_scores_).T

    overall_eval = pd.concat([overall_eval,across_kb_scores_])
    return overall_eval, eval_dict


def main(_):

    # Define file/directory paths
    MAIN_DIR = experiment.get_tf_config()['model_dir']
    MODEL_DIR = os.path.join(MAIN_DIR, 'model_nrf_pdpa_ins', FLAGS.model_name)
    MODEL_BEST_DIR = os.path.join(MAIN_DIR, 'model_nrf_pdpa_ins', FLAGS.model_name, 'best')
    MODEL_LAST_DIR = os.path.join(MAIN_DIR, 'model_nrf_pdpa_ins', FLAGS.model_name, 'last')
    EVAL_DIR = os.path.join(MAIN_DIR, 'results_nrf_pdpa_ins', FLAGS.model_name)

    if not os.path.isdir(MODEL_LAST_DIR): os.makedirs(MODEL_LAST_DIR)
    if not os.path.isdir(EVAL_DIR):os.makedirs(EVAL_DIR)

    EVAL_SCORE_PATH = os.path.join(EVAL_DIR, '_eval_scores.xlsx')
    EVAL_DICT_PATH = os.path.join(EVAL_DIR, '_eval_details.pickle')

    logger.info(f'Models will be saved at: {MODEL_DIR}')
    logger.info(f'Best model will be saved at: {MODEL_BEST_DIR}')
    logger.info(f'Last trained model will be saved at {MODEL_LAST_DIR}')
    logger.info(f'Saving Eval_Score at: {EVAL_SCORE_PATH}')
    logger.info(f'Saving Eval_Dict at: {EVAL_DICT_PATH}')

    # Create training set based on chosen random seed
    logger.info("Generating training/ evaluation set")

    
    """
    LOAD MODEL
    """
    # Instantiate chosen model
    logger.info(f"Instantiating model: {FLAGS.model_name}")
    models = {
        "albert": GoldenRetriever_ALBERT,
        "bert": GoldenRetriever_BERT,
        "USE": GoldenRetriever
    }

    if FLAGS.model_name not in models:
        raise ValueError("Model not found: %s" % (FLAGS.model_name))

    model = models[FLAGS.model_name](max_seq_length=FLAGS.max_seq_length)
    logger.info(f"Model's max_seq_length: {model.max_seq_length}")

    # Set optimizer parameters
    model.opt_params = {'learning_rate': FLAGS.learning_rate,'beta_1': FLAGS.beta_1,'beta_2': FLAGS.beta_2,'epsilon': FLAGS.epsilon}



    """
    PULL AND PARSE KB FROM SQL
    """
    train_dict = dict()
    test_dict = dict()
    df_list = []

    # Get df using kb_handler
    kbh = kb_handler()
    kbs = kbh.load_sql_kb(
                        #   cnxn_path='db_cnxn_str.txt', 
                          cnxn_path='/polyaxon-data/goldenretriever/db_cnxn_str.txt', 
                        #   kb_names=['PDPA', 'nrf_16032020', 'covid19_16032020','annuities','home-insurance','Super_Bowl_50','Nikola_Tesla'])
                        #   kb_names=['PDPA', 'nrf_16032020', 'covid19_16032020']) 
                        #   kb_names=['PDPA', 'nrf_16032020']) # covid19_16032020 is to be excluded
                          kb_names=['nrf_16032020'])
        
    df = pd.concat([single_kb.create_df() for single_kb in kbs]).reset_index(drop='True')
    kb_names = df['kb_name'].unique()

    for kb_name in kb_names:
        kb_id = df[df['kb_name'] == kb_name].index.values
        train_idx, test_idx = train_test_split(kb_id, test_size=0.4,
                                            random_state=100)

        train_dict[kb_name] = train_idx
        test_dict[kb_name] = test_idx



    """
    FINETUNE
    """
    if FLAGS.task_type == 'train_eval':
        logger.info("Fine-tuning model")
            # Required for contrastive loss
            # label = tf.placeholder(tf.int32, [None], name='label')

        # see the performance of out of box model
        OOB_overall_eval, eval_dict = eval_model(model, df, test_dict)
        epoch_eval_score = OOB_overall_eval.loc['Across_all_kb','mrr_score']
        epoch_nrf_eval_score = OOB_overall_eval.loc['nrf_16032020','mrr_score']
        logger.info(f'Eval Score for OOB: {epoch_eval_score}')
        logger.info(f'Eval Score for OOB: {epoch_nrf_eval_score}')

        earlystopping_counter = 0
        for i in range(FLAGS.num_epochs):
            epoch_start_time = datetime.datetime.now()
            logger.info(f'Running Epoch #: {i}')

            cost_mean_total = 0
            batch_counter = 0
            epoch_start_time = datetime.datetime.now()

            # train_dataset_loader = random_triplet_generator(df, train_dict)
            train_dataset_loader = hard_triplet_generator(df, train_dict, model)

            for q, r, neg_r in train_dataset_loader:
                
                if random.randrange(100) <= 10:
                    logger.info(f'\nTRIPLET SPOT CHECK')
                    logger.info(f'{q[0]}')
                    logger.info(f'{r[0]}')
                    logger.info(f'{neg_r[0]}\n')
                
                batch_start_time = datetime.datetime.now()

                if batch_counter % 100 == 0:
                    logger.info(f'Running batch #{batch_counter}')
                cost_mean_batch = model.finetune(question=q, answer=r, context=r, \
                                                 neg_answer=neg_r, neg_answer_context=neg_r, \
                                                 margin=FLAGS.margin, loss=FLAGS.loss_type)

                cost_mean_total += cost_mean_batch

                batch_end_time = datetime.datetime.now()

                if batch_counter == 0 and i == 0:
                    len_training_triplets = sum([len(train_idxes) for kb, train_idxes in train_dict.items()])
                    num_batches = len_training_triplets // len(q)
                    logger.info(f'Training batches of size: {len(q)}')
                    logger.info(f'Number of batches per epoch: {num_batches}')
                    logger.info(f'Time taken for first batch: {batch_end_time - batch_start_time}')

                if batch_counter == num_batches:
                    break
                
                batch_counter += 1


            epoch_overall_eval, eval_dict = eval_model(model, df, test_dict)
            epoch_eval_score = epoch_overall_eval.loc['Across_all_kb','mrr_score']
            epoch_nrf_eval_score = epoch_overall_eval.loc['nrf_16032020','mrr_score']
            print(epoch_eval_score)

            logger.info(f'Number of batches trained: {batch_counter}')
            logger.info(f'Loss for Epoch #{i}: {cost_mean_total}')
            logger.info(f'Eval Score for Epoch #{i}: {epoch_eval_score}')
            logger.info(f'Eval Score for Epoch for nrf #{i}: {epoch_nrf_eval_score}')

            epoch_end_time = datetime.datetime.now()
            logger.info(f'Time taken for Epoch #{i}: {epoch_end_time - epoch_start_time}')

            # Save model for first epoch
            if i == 0:
                lowest_cost = cost_mean_total
                highest_epoch_eval_score = epoch_eval_score
                best_epoch = i
                earlystopping_counter = 0
                os.makedirs(os.path.join(MODEL_BEST_DIR, str(i)))
                model.export(os.path.join(MODEL_BEST_DIR, str(i)))
                _flags_to_file(FLAGS.get_key_flags_for_module(sys.argv[0]),
                                os.path.join(MODEL_BEST_DIR, str(i), 'train.cfg'))

            # Model checkpoint
            if epoch_eval_score > highest_epoch_eval_score:
            # if cost_mean_total < lowest_cost:
                best_epoch = i
                lowest_cost = cost_mean_total
                highest_epoch_eval_score = epoch_eval_score
                os.makedirs(os.path.join(MODEL_BEST_DIR, str(i)))
                model.export(os.path.join(MODEL_BEST_DIR, str(i)))
                _flags_to_file(FLAGS.get_key_flags_for_module(sys.argv[0]),
                               os.path.join(MODEL_BEST_DIR, str(i), 'train.cfg'))
                logger.info(f'Saved best model with cost of {lowest_cost} for Epoch #{i}')
                logger.info(f'Saved best model with cost of {highest_epoch_eval_score} for Epoch #{i}')
                earlystopping_counter = 0
            else:
                # Activate early stopping counter
                earlystopping_counter += 1

            experiment.log_metrics(steps=i, loss=cost_mean_total)

            # Early stopping
            if earlystopping_counter == FLAGS.early_stopping_steps:
                logger.info("Early stop executed")
                model.export(MODEL_LAST_DIR)
                _flags_to_file(FLAGS.get_key_flags_for_module(sys.argv[0]),
                                os.path.join(MODEL_LAST_DIR, 'train.cfg'))
                break
            
            epoch_end_time = datetime.datetime.now()
            logger.info(f'Time Taken for Epoch #{i}: {epoch_end_time - epoch_start_time}')
            logger.info(f'Average time Taken for each batch: {(epoch_end_time - epoch_start_time)/batch_counter}')

    # Restore best model. User will have to define path to model if only eval is done.
    logger.info("Restoring model")
    if FLAGS.task_type == 'train_eval':
        model.restore(os.path.join(MODEL_BEST_DIR, str(best_epoch)))
    else:
        if FLAGS.eval_model_dir:
            model.restore(FLAGS.eval_model_dir)
        else:
            logger.info("Using out-of-box model")
            pass


    """
    EVAL MODEL
    """
    logger.info("Evaluating model")
       
    overall_eval, eval_dict = eval_model(model, df, test_dict)
    print("="*10 + ' OOB ' + "="*10)
    print(OOB_overall_eval)
    print("="*10 + ' FINETUNED ' + "="*10)
    print(overall_eval)
    experiment.log_metrics(nrf_mrr=overall_eval.loc['nrf_16032020', 'mrr_score'])
    experiment.log_metrics(overall_mrr=overall_eval.loc['Across_all_kb', 'mrr_score'])

    # save the scores and details for later evaluation. WARNING: User will need to create the necessary directories to save df
    overall_eval.to_excel(EVAL_SCORE_PATH)
    with open(EVAL_DICT_PATH, 'wb') as handle:
        pickle.dump(eval_dict, handle)


if __name__ == "__main__":
    app.run(main)

