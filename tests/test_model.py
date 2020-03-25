import pytest
import pandas as pd
import os
import sys
import random
import numpy as np
import shutil
sys.path.append("../")

from src.model import GoldenRetriever
from src.kb_handler import kb, kb_handler
from sklearn.model_selection import train_test_split


def test_make_query():
    gr = GoldenRetriever()
    kbh = kb_handler() 
    nrf = kbh.load_sql_kb(cnxn_path="../db_cnxn_str.txt", kb_names=["nrf"])
    gr.load_kb(nrf)

    querystring = "Can I change funding source"
    actual = gr.make_query(querystring, top_k=5, index=False, predict_type="query", kb_name="nrf")

    assert isinstance(actual[0], list)
    assert isinstance(actual[0][0], str)
    assert len(actual[0]) == 5


def gen(batch_size, query, response, neg_response, shuffle_data=False):
    random.seed(42)
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
    random.seed(42)

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

def random_triplet_generator(df, train_dict):
    train_dict_with_neg = _generate_neg_ans(df, train_dict)
    train_pos_idxs = np.concatenate([v[0] for k,v in train_dict_with_neg.items()], axis=0)
    train_neg_idxs = np.concatenate([v[1] for k,v in train_dict_with_neg.items()], axis=0)

    train_query = df.iloc[train_pos_idxs].query_string.tolist()
    train_response = df.iloc[train_pos_idxs].processed_string.tolist()
    train_neg_response = df.iloc[train_neg_idxs].processed_string.tolist()
    
    train_dataset_loader = gen(32, train_query, train_response, train_neg_response, shuffle_data=True)
    
    return train_dataset_loader


@pytest.fixture
def create_delete_model_savepath():

    savepath = os.path.join(os.getcwd(), "finetune")

    yield savepath

    shutil.rmtree(savepath)


def test_finetune_export_restore(create_delete_model_savepath):
    gr = GoldenRetriever()

    train_dict = dict()
    test_dict = dict()

    # Get df using kb_handler
    kbh = kb_handler()
    kbs = kbh.load_sql_kb(cnxn_path="../db_cnxn_str.txt",
                          kb_names=['PDPA'])

    df = pd.concat([single_kb.create_df() for single_kb in kbs]).reset_index(drop='True')
    kb_names = df['kb_name'].unique()

    for kb_name in kb_names:
        kb_id = df[df['kb_name'] == kb_name].index.values
        train_idx, test_idx = train_test_split(kb_id, test_size=0.4,
                                            random_state=100)

        train_dict[kb_name] = train_idx
        test_dict[kb_name] = test_idx

    train_dataset_loader = random_triplet_generator(df, train_dict)

    batch_counter = 0

    for i in range(2):

        cost_mean_total = 0

        train_dataset_loader = random_triplet_generator(df, train_dict)

        for q, r, neg_r in train_dataset_loader:
            
            if batch_counter == 1:
                break

            cost_mean_batch = gr.finetune(question=q, answer=r, context=r, \
                                          neg_answer=neg_r, neg_answer_context=neg_r, \
                                          margin=0.3, loss="triplet")

            cost_mean_total += cost_mean_batch

            batch_counter += 1

    initial_pred = gr.predict("What is personal data?")

    savepath = create_delete_model_savepath
    gr.export(savepath)

    gr_new = GoldenRetriever()
    gr_new.restore(savepath)
    restored_pred = gr_new.predict("What is personal data?")

    assert isinstance(cost_mean_batch, np.floating)
    assert cost_mean_total != cost_mean_batch
    assert os.path.isdir(savepath)
    assert np.array_equal(initial_pred, restored_pred)


def test_load_kb():
    gr = GoldenRetriever()
    kbh = kb_handler()

    pdpa_df = pd.read_csv('./data/pdpa.csv')
    pdpa = kbh.parse_df('pdpa', pdpa_df, 'answer', 'question', 'meta')

    
    gr.load_kb(kb_=pdpa)

    assert isinstance(gr.kb["pdpa"], kb)
