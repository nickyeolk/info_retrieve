import pandas as pd
import numpy as np
import pyodbc

class kb:
    """
    kb object holds the responses, queries, mapping and vectorised_responses.
    
    This class is used for all finetuning and training purposes.
    Txt, csv and sql data can be loaded into this class with accompanying kb_handler class
    
    
    Example:
    -------    
    # PDPA csv file
    pdpa_df = pd.read_csv('../data/pdpa.csv')
    pdpa = kbh.parse_df('pdpa', pdpa_df, 'answer', 'question', 'meta')

    display(pdpa.responses)
    display(pdpa.queries)
    display(pdpa.mapping)

    
    #SQL kbs
    kbs = kbh.load_sql_kb()
    ii = -1
    display(kbs[ii].name)
    display(kbs[ii].responses)
    display(kbs[ii].queries)
    display(kbs[ii].mapping)
    """
    def __init__(self, name, responses, queries, mapping, vectorised_responses=None):
        self.name = name
        self.responses = responses
        self.queries = queries
        self.mapping = mapping
        self.vectorised_responses = vectorised_responses
    
    
class kb_handler():
    """
    kb_handler loads knowledge bases from text files
    """
    def preview(self, path, N=20):
        """
        Print the first N lines of the file in path
        """
        with open(path) as text_file:
            for i in range(N):
                print(next(text_file))
                
    
    def parse_df(self, kb_name, df, answer_col, query_col='', context_col='context_string'):
        """
        parses pandas DataFrame into responses, queries and mappings
        
        args:
        ------
            kb_name: (str) name of kb to be held in kb object
            df: (pd.DataFrame) contains the queries, responses and context strings
            answer_col: (str) column name string that points to responses
            query_col: (str) column name string that points to queries
            context_col: (str) column name string that points to context strings
        Return:
        ------
            kb object
        """
        df = df.assign(context_string_ = '') if context_col == 'context_string' else df 
        df = df.rename(columns = {
                                   answer_col: 'raw_string', 
                                   context_col: 'context_string',
                                   query_col: 'query_string'
                                  })
            
        queries, mappings = [], []
        unique_responses_df = df.loc[~df.duplicated(), ['raw_string', 'context_string']].drop_duplicates().reset_index(drop=True)
        
        if query_col=='':
            # if there are no query columns
            # there will be no query or mappings to return
            # simply return unique responses now
            return unique_responses_df, pd.DataFrame(), pd.DataFrame()
        
        """
        Handle many-to-many matching between the queries and responses
            1. Get list of unique queries and responses
            2. Index the given queries and responses 
            3. Create mappings from the indices of these non-unique queries and responses
        """
        contexted_answer = df.loc[:,'context_string'].fillna('') + ' ' + df.loc[:,'raw_string'].fillna('')
        response_list = contexted_answer.tolist()
        unique_response_list = contexted_answer.drop_duplicates().tolist()

        question_list = df.loc[:,'query_string'].tolist()
        unique_question_list = df.loc[:,'query_string'].drop_duplicates().tolist()

        response_idx = [unique_response_list.index(response) for response in response_list]
        query_idx = [unique_question_list.index(question) for question in question_list]
        
        # create mapping from query-response indices
        for one_query_idx, one_response_idx in zip(query_idx, response_idx):
            mappings.append([one_query_idx, one_response_idx])
        
        unique_query_df = df.loc[:,['query_string']].drop_duplicates().reset_index(drop=True)
        
        return kb(kb_name, unique_responses_df, unique_query_df, mappings)
    
    
    def parse_text(self, path, clause_sep='/n', inner_clause_sep='', 
                   query_idx=None, context_idx=None, 
                   kb_name=None):
        """
        Parse text file from kb path into query, response and mappings
        
        args:
        ----
            path: (str) path to txt file
            clause_sep: (str, default = '\n') Seperates the text file into their clauses
            inner_clause_sep: (str, default = '') In the case that either query or context 
                                                  string is encoded within the first few 
                                                  sentences, inner_clause_sep may separate 
                                                  the sentences and query_idx and context_idx
                                                  will select the query and context strings 
                                                  accordingly
            query_idx: (int, default = None)
            context_idx: (int, default = None)
            kb_name: (str, default = name of path file)
        """
        kb_name = kb_name if kb_name is not None else path.split('/')[-1].split('.')[0]
        
        """
        1. Parse the text into its fields
        """
        # read the text
        with open(path) as text_file:
            self.raw_text = text_file.read()
        
        clauses = [clause for clause in self.raw_text.split(clause_sep) if clause!='']
        clauses = pd.DataFrame(clauses, columns = ['raw_string']).assign(context_string='')
        query_list = []
        mappings = []
        
        """
        2. This logic settles inner clause parsing. 
           ie the first line is the query or the context string
        """
        if (inner_clause_sep != ''):
            
            assert ((query_idx is not None) | (context_idx is not None)), "either query_idx or context_idx must not be None"
            clause_idx = max([idx for idx in [query_idx, context_idx, 0] if idx is not None]) + 1
            
            new_clause_list = []
            for idx, clause in clauses.raw_string.iteritems():
                
                inner_clauses = clause.strip(inner_clause_sep).split(inner_clause_sep)
                
                if query_idx is not None: 
                    query_list.append(inner_clauses[:query_idx+1])
                    mappings.append([idx, idx])
                    
                
                context_string = inner_clause_sep.join(inner_clauses[:context_idx+1]) if context_idx is not None else ''
                new_clause_list.append( {
                                         "raw_string":inner_clause_sep.join(inner_clauses[clause_idx:]),
                                         "context_string": context_string
                                        })

            clauses = pd.DataFrame(new_clause_list)
                            
        return kb(kb_name, clauses, pd.DataFrame(query_list, columns=['query_string']), mappings)
            
    def parse_csv(self, path, answer_col, query_col='', context_col='context_string'):
        """
        Parse CSV file into kb format
        As pandas leverages csv.sniff to parse the csv, this function leverages pandas.
        
        args:
        ------
            kb_name: (str) name of kb to be held in kb object
            df: (pd.DataFrame) contains the queries, responses and context strings
            answer_col: (str) column name string that points to responses
            query_col: (str) column name string that points to queries
            context_col: (str) column name string that points to context strings
        """
        kb_name = kb_name if kb_name is not None else path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path)
        kb = self.parse_df(kb_name, df, answer_col, query_col, context_col)
        return kb
    
    def load_sql_kb(self, cnxn_path = "../db_cnxn_str.txt", kb_names=[]):
        """
        Load the knowledge bases from SQL.
        
        GoldenRetriever keeps the responses text in the 
        text and vectorized_knowledge attributes 
        as dictionaries indexed by their respective kb names
        
        TODO: load vectorized knowledge from precomputed weights

        args:
            cnxn_path: (str) string directory of the connection string 
                            that needs to be fed into pyodbc
            kb_names: (list, default=[]) to list specific kb_names to parse
                                         else if empty, parse all of them
        """
        conn = pyodbc.connect(open(cnxn_path, 'r').read())
        
        SQL_Query = pd.read_sql_query('''SELECT dbo.query_labels.id, dbo.query_db.query_string, \
                                     dbo.kb_clauses.context_string, dbo.kb_clauses.processed_string, dbo.kb_clauses.raw_string, dbo.kb_raw.kb_name, dbo.kb_raw.type FROM dbo.query_labels \
                                     JOIN dbo.query_db ON dbo.query_labels.query_id = dbo.query_db.id \
                                     JOIN dbo.kb_clauses ON dbo.query_labels.clause_id = dbo.kb_clauses.id \
                                     JOIN dbo.kb_raw ON dbo.kb_clauses.raw_id = dbo.kb_raw.id''', conn)

        conn.close()
        
        df = SQL_Query.set_index('id')
        kb_names = df['kb_name'].unique() if len(kb_names) == 0 else kb_names
        
        kbs = []
        for kb_name in kb_names:
            one_kb_df = df.loc[df.kb_name==kb_name]
            kb = self.parse_df(kb_name, one_kb_df, 'raw_string', query_col='query_string', context_col='context_string')
            kbs.append(kb)
        
        return kbs
            
            
        

