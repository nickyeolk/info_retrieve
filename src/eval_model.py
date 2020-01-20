"""
Description:
----------
This script evaluates the model as described in ReQA paper
https://arxiv.org/abs/1907.04780
The methodology centers around generating an "answer index" and "question index" 
from which the ranks are calculated and scored.

How to use:
-----------
When at root dir, enter in terminal
python -m src.eval_model -m GoldenRetriever -s fine_tune
"""

import os
import pickle
import datetime 

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.mstats import rankdata
import argparse

import sys
sys.path.append('..')
import src.model as models
from src.dataloader import kb_train_test_split

# metrics
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


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--modelname", default='GoldenRetriever', help="name of model class in GR src")
parser.add_argument("-s", "--savepath", default = 'fine_tune', help="directory of the model's saved weights")
args = parser.parse_args()
print(args.modelname)
print(args.savepath)

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    model_to_eval = getattr(models, args.modelname)()
    if args.savepath != '':
        model_to_eval.restore(args.savepath)

    df, train_dict, test_dict, train_idx_all, test_idx_all = kb_train_test_split(0.6, 100)

    eval_dict = {}
    # for kb_name in ['PDPA', 'nrf', 'critical-illness-insurance', 'other-insurance', 'Steam_engine', '1973_oil_crisis']:
    for kb_name in df.kb_name.unique():
        print(f"\n Evaluating on {kb_name} \n")

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

        # get encoded queries and responses
        encoded_queries = model_to_eval.predict(query_list, type='query')
        encoded_responses = model_to_eval.predict(response_list, type='response')

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
    print(overall_eval)

    # save the scores and details for later evaluation
    overall_eval.to_excel('GoldenRetrieval_eval_scores.xlsx')
    with open("GoldenRetrieval_eval_details.pickle", 'wb') as handle:
        pickle.dump(eval_dict, handle)

    end_time = datetime.datetime.now()
    print(f"Start      : {start_time}")
    print(f"End        : {end_time}")
    print(f"Time Taken : {end_time - start_time}")

"""
# expected output for 2 knowledge bases, PDPA and nrf

               mrr_score  r1_score  r2_score  r3_score
PDPA           0.579315  0.474576  0.542373  0.652542
nrf            0.097777         0  0.045977  0.091954
Across_all_kb  0.374955  0.273171  0.331707  0.414634
"""
