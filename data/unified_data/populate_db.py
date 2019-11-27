import os
print (os.getcwd())
from create_db import create_connection
from pathlib import Path
import pandas as pd
import sys
sys.path.append('.')
from src.utils import question_cleaner
import random



def load_insurance_qna_data():
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
        return random.sample([xx for xx in kb if xx not in gt], 90)
    df_query['neg_samples']=df_query.apply(lambda x: func(x), axis=1)
    df_test['neg_samples']=df_test.apply(lambda x: func(x), axis=1)

    def wordifier(tokes):
        return ' '.join([dict_ind2word[ind] for ind in tokes.strip().split(' ')])
    df_doc['text']=df_doc.apply(lambda x: wordifier(x[1]), axis=1)
    df_query['text']=df_query.apply(lambda x: wordifier(x[1]), axis=1)
    df_test['text']=df_test.apply(lambda x: wordifier(x[1]), axis=1)
    df_all = pd.concat([df_query, df_test])
    category_list = df_all[0].unique()
    df_all['pos_samples'] = df_all.apply(lambda x: [int(xx) for xx in (x[2]).split(' ')], axis=1)

    return df_all, df_doc, category_list



def create_kb_insure(conn):
    cur = conn.cursor()
    cur.execute("""SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';""")
    print(cur.fetchall())
    df_all, df_doc, category_list = load_insurance_qna_data()

    prefix = 'InsuranceQA_'
    for category in category_list:
        df_cat = df_all[df_all[0]==category]
        qa_pairs=[]
        for ii, row in df_cat.iterrows():
            question_text = row['text']
            for answer in row['pos_samples']:
                answer_text = df_doc.loc[answer, 'text']
                qa_pairs.append([question_text, answer_text])
        df_see = pd.DataFrame(qa_pairs, columns=['qn', 'ans'])
        print(len(df_see))
        x = df_see.groupby('ans')['ans'].count()
        print('repeats', len(x[x>1]))
        df_kb = pd.DataFrame(df_see['ans'].unique(), columns=['answer']) # this is answer index
        def func(row):
            return df_kb[df_kb['answer']==row['ans']].index.values[0]
        df_see['ans_ind'] = df_see.apply(lambda x: func(x), axis=1) # this is qn vs answer index

        kb_raw_statement="""INSERT INTO kb_raw (filepath, kb_name, type) VALUES (?, ?, ?)"""
        cur.execute(kb_raw_statement, ['None', category, 'qna'])
        kb_raw_uid = cur.lastrowid
        kb_clause_statement="""INSERT INTO kb_clauses (raw_id, clause_ind, raw_string, processed_string) VALUES (?, ?, ?, ?)"""
        for ii, row in df_kb.iterrows():
            cur.execute(kb_clause_statement, [kb_raw_uid, ii, row.values[0], row.values[0]])
            kb_clause_uid = cur.lastrowid

            df_relevant_ans = df_see[df_see['ans_ind']==ii]
            labeled_queries_statement="""INSERT INTO labeled_queries (query_string, clause_id) VALUES (?, ?)"""
            for jj, roww in df_relevant_ans.iterrows():
                cur.execute(labeled_queries_statement, [roww['qn'], kb_clause_uid])



def main():
    conn = create_connection('./data/unified_data/pythonsqlite.db')
    create_kb_insure(conn)
    conn.commit()

if __name__ == '__main__':
    main()