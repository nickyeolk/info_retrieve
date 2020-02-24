import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split

def kb_train_test_split(test_size, random_state):

    """Retrieve train-test data for evaluation.
    Parameters
    ----------
    test_size: float, int or None, optional (default=None)
    Random_state: int, RandomState instance or None, optional (default=None)

    Returns
    -------
    df: (dataframe): full dataframe from SQL database
    train_dict: (dictionary) kb_name: array of ids for training set
    test_dict: (dictionary) kb_name: array of ids for testing set
    train_idx_all: (list) ids for training set
    test_idx_all: (list) ids for testing set
    """

    cnxn_path = "/polyaxon-data/goldenretriever/db_cnxn_str.txt"
    conn = pyodbc.connect(open(cnxn_path, 'r').read())

    SQL_Query = pd.read_sql_query('''SELECT dbo.query_labels.id, dbo.query_db.query_string, \
                                 dbo.kb_clauses.processed_string, dbo.kb_raw.kb_name, dbo.kb_raw.type FROM dbo.query_labels \
                                 JOIN dbo.query_db ON dbo.query_labels.query_id = dbo.query_db.id \
                                 JOIN dbo.kb_clauses ON dbo.query_labels.clause_id = dbo.kb_clauses.id \
                                 JOIN dbo.kb_raw ON dbo.kb_clauses.raw_id = dbo.kb_raw.id''', conn)

    df = pd.DataFrame(SQL_Query).set_index('id')
    kb_names = df['kb_name'].unique()

    train_dict = dict()
    test_dict = dict()

    train_idx_all = []
    test_idx_all = []

    for kb_name in kb_names:
        kb_id = df[df['kb_name'] == kb_name].index.values
        train_idx, test_idx = train_test_split(kb_id, test_size=test_size,
                                               random_state=random_state)
        
        train_dict[kb_name] = train_idx
        test_dict[kb_name] = test_idx
        
    for k,v in train_dict.items():
        for idx in v:
            train_idx_all.append(idx)
            
    for k,v in test_dict.items():
        for idx in v:
            test_idx_all.append(idx)
    
    return df, train_dict, test_dict, train_idx_all, test_idx_all

def document_retrieval(db_string_path):
    conn = pyodbc.connect(open(db_string_path, 'r').read())
    SQL_Query = pd.read_sql_query('''SELECT dbo.kb_raw.id, kb_name, STRING_AGG(raw_string,' ') as raw_txt 
                                    FROM dbo.kb_raw
                                    inner join dbo.kb_clauses as kbc
                                    ON kbc.raw_id=dbo.kb_raw.id
                                    WHERE directory_id=11
                                    GROUP BY kb_name, dbo.kb_raw.id
                                    ''', conn)
    df_kb = pd.DataFrame(SQL_Query)

    SQL_Query = pd.read_sql_query('''SELECT dbo.kb_clauses.raw_id as doc_id, query_string 
                                    FROM dbo.query_db
                                    INNER JOIN dbo.query_labels
                                    ON dbo.query_db.id=dbo.query_labels.query_id
                                    INNER JOIN dbo.kb_clauses
                                    ON dbo.query_labels.clause_id=dbo.kb_clauses.id
                                    INNER JOIN dbo.kb_raw
                                    ON dbo.kb_raw.id=dbo.kb_clauses.raw_id
                                    WHERE dbo.kb_raw.directory_id=11
                                        ''', conn)
    df_query = pd.DataFrame(SQL_Query)
    return df_kb, df_query