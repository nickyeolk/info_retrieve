import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import read_txt, split_txt, aiap_qna_quickscore
from src.utils import question_cleaner, display_qn_and_ans, aiap_qna, ranker, scorer
import tensorflow as tf
import tensorflow_hub as hub
from src.model import GoldenRetriever
from pathlib import Path
import random
import pickle
import sys

if __name__=='__main__':

    learning_rates=[0.06]
    margins=[0.3]

    # Load and clean dataset
    datapath=Path('./data')
    df_query = pd.read_csv(datapath/'insuranceQA/V2/InsuranceQA.question.anslabel.raw.500.pool.solr.train.encoded', delimiter='\t', header=None)
    df_doc = pd.read_csv(datapath/'insuranceQA/V2/InsuranceQA.label2answer.raw.encoded', delimiter='\t', header=None)
    df_ind2word = pd.read_csv(datapath/'insuranceQA/V2/vocabulary', sep='\t', header=None, quotechar='', quoting=3, keep_default_na=False)
    dict_ind2word = pd.Series(df_ind2word[1].values,index=df_ind2word[0].values).to_dict()
    df_test = pd.read_csv(datapath/'insuranceQA/V2/InsuranceQA.question.anslabel.raw.100.pool.solr.test.encoded', delimiter='\t', header=None)
    df_test=question_cleaner(df_test)
    df_query=question_cleaner(df_query)
    df_doc=df_doc.set_index(0)
    def func(row):
        kb=[int(xx) for xx in (row[3]).split(' ')]
        gt = [int(xx) for xx in (row[2]).split(' ')]
        return random.sample([xx for xx in kb if xx not in gt], 100)
    df_query['neg_samples']=df_query.apply(lambda x: func(x), axis=1)
    def wordifier(tokes):
        return ' '.join([dict_ind2word[ind] for ind in tokes.strip().split(' ')])
    df_doc['text']=df_doc.apply(lambda x: wordifier(x[1]), axis=1)
    df_query['text']=df_query.apply(lambda x: wordifier(x[1]), axis=1)
    df_test['text']=df_test.apply(lambda x: wordifier(x[1]), axis=1)

    # Load processed train data
    with open("./data/tmp_trainset.pickle", "rb") as f:
        answers,questions,wrong_answers = pickle.load(f) 
    random.shuffle(wrong_answers)
    # expid = mlflow.set_experiment(sys.argv[1])
    for learning_rate in learning_rates:
        for margin in margins:
            model=GoldenRetriever(lr=learning_rate, margin=margin, loss='triplet')
            batch_size=100
            for epoch in range(3):
                # mlflow.start_run(experiment_id=expid)
                # mlflow.log_params({'learning_rate':learning_rate, 'margin':margin, 'epoch':epoch})
                for ii in range(0, len(answers), batch_size):
                    (qn, ans, neg_ans) = (questions[ii:ii+batch_size], answers[ii:ii+batch_size], wrong_answers[ii:ii+batch_size])
                    current_loss = model.finetune(qn, ans, ans, neg_ans, neg_ans)
                    # mlflow.log_metric('loss',current_loss)
                    # print(ii, current_loss)
                # score on test data
                question_vectors = model.predict(df_test['text'].tolist(), type='query')
                predictions, gts = ranker(model, question_vectors, df_test, df_doc)
                for k in range(5):
                    string='Accuracy_at_'
                    string+=str(k+1)
                    # mlflow.log_metric(string, scorer(predictions, gts, k+1))
                    # print('Score @{}: {:.4f}'.format(k+1, scorer(predictions, gts, k+1)))
                # mlflow.end_run()
            model.export(savepath='./google_use_qa_insuranceqa/variables')
            model.close()


