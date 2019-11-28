import os
print (os.getcwd())
from create_db import create_connection
from pathlib import Path
import pandas as pd
import sys
sys.path.append('.')
from src.utils import question_cleaner
import random
from datetime import datetime
import secrets



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



def create_kb_insure(conn, dir_uid):
    cur = conn.cursor()
    cur.execute("""SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';""")
    print(cur.fetchall())
    df_all, df_doc, category_list = load_insurance_qna_data()

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

        kb_raw_statement="""INSERT INTO kb_raw (filepath, kb_name, type, directory_id) VALUES (?, ?, ?, ?)"""
        cur.execute(kb_raw_statement, ['None', category, 'qna', dir_uid])
        kb_raw_uid = cur.lastrowid
        kb_clause_statement="""INSERT INTO kb_clauses (raw_id, clause_ind, raw_string, processed_string, created_at) VALUES (?, ?, ?, ?, ?)"""
        for ii, row in df_kb.iterrows():
            cur.execute(kb_clause_statement, [kb_raw_uid, ii, row.values[0], row.values[0], datetime.now()])
            kb_clause_uid = cur.lastrowid

            df_relevant_ans = df_see[df_see['ans_ind']==ii]
            labeled_queries_statement="""INSERT INTO labeled_queries (query_string, clause_id, created_at) VALUES (?, ?, ?)"""
            for _, roww in df_relevant_ans.iterrows():
                cur.execute(labeled_queries_statement, [roww['qn'], kb_clause_uid, datetime.now()])

def create_kb_pdpa(conn, dir_uid):
    cur = conn.cursor()
    def read_and_condition_csv(csv_path, meta_col='meta', answer_col='answer', query_col='question', answer_str_col='answer', cutoff=None):
        """Only read organization meta, not personal. index=196"""
        df_pdpa = pd.read_csv(csv_path)
        if cutoff:
            df_pdpa = df_pdpa.iloc[:cutoff]
        df_pdpa['kb'] = df_pdpa[meta_col]+df_pdpa[answer_col]
        df_pdpa.rename(columns={query_col:'queries', answer_str_col:'answer_str'}, inplace=True)
        df_pdpa[answer_col] = [[x] for x in df_pdpa.index]
        df_pdpa['kb']=df_pdpa['kb'].str.replace('\n', '. ').replace('.. ', '. ')
        return df_pdpa
    df_pdpa = read_and_condition_csv('./data/pdpa.csv', cutoff=196)

    kb_raw_statement="""INSERT INTO kb_raw (filepath, kb_name, type, directory_id) VALUES (?, ?, ?, ?)"""
    cur.execute(kb_raw_statement, ['./data/pdpa.csv', 'PDPA', 'qna', dir_uid])    
    kb_raw_uid = cur.lastrowid

    kb_clause_statement="""INSERT INTO kb_clauses (raw_id, clause_ind, context_string, raw_string, processed_string, created_at) VALUES (?, ?, ?, ?, ?, ?)"""
    labeled_queries_statement="""INSERT INTO labeled_queries (query_string, clause_id, created_at) VALUES (?, ?, ?)"""
    for ii, row in df_pdpa.iterrows():
        cur.execute(kb_clause_statement, [kb_raw_uid, ii, row['meta'], row['answer_str'], row['kb'], datetime.now()])
        kb_clause_uid = cur.lastrowid
        cur.execute(labeled_queries_statement, [row['queries'], kb_clause_uid, datetime.now()])
    



def create_user_id(conn):
    user_insert_statement="""INSERT INTO users (created_at, email, full_name, org_name, hashkey) VALUES (?, ?, ?, ?, ?)"""
    cur = conn.cursor()
    cur.execute(user_insert_statement, [datetime.now(), 'lik@aisingapore.org', 'likkhian', 'aisg', secrets.token_hex(16)])
    return cur.lastrowid

def create_kb_directory(conn, user_uid, dir_name):
    kb_directory_insert_statement = """INSERT INTO kb_directory (created_at, dir_name, user_id) VALUES (?, ?, ?)"""
    cur = conn.cursor()
    cur.execute(kb_directory_insert_statement, [datetime.now(), dir_name, user_uid])
    return cur.lastrowid

def load_my_kbs(conn):
    user_uid = create_user_id(conn)
    dir_uid = create_kb_directory(conn, user_uid, 'insuranceQA')
    create_kb_insure(conn, dir_uid)
    dir_uid = create_kb_directory(conn, user_uid, 'PDPA')
    create_kb_pdpa(conn, dir_uid)


def main():
    conn = create_connection('./data/unified_data/pythonsqlite.db')
    load_my_kbs(conn)
    conn.commit()

if __name__ == '__main__':
    main()