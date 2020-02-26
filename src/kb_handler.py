"""
kb_handler converts knowledge bases in various data types (txt, csv, sql) into a kb object.
This kb object may then be further used for finetuning and eval
"""

import pandas as pd
import numpy as np
import pyodbc

def unique_indexing(non_unique):
    """
    Convert a non_unique string pd.Series into a list of its indices of its unique list
    
    Args:
    -----
        non_unique: (pd.Series) containing non_unique values
    
    Return:
    ------
        idxed_non_uniques: (list) contains the index of non unique values
                                  indexed by the unique values
    """
    unique = non_unique.drop_duplicates().tolist()
    non_unique = non_unique.tolist()
    idxed_non_uniques = [unique.index(each_non_unique) for each_non_unique in non_unique]
    return idxed_non_uniques

def generate_mappings(responses, queries):
    """
    Generate a list of list mappings between responses and queries
    The length of responses and queries must be the same.
    
    
    To note, the argument takes Responses then Queries ...
    but the returned mappings list Queries then Responses 
    for convenient use in downstream scripts

    Args:
    ---
        responses: (pd.Series) contains query strings, may be non unique
        queries: (pd.Series) contains query strings, may be non unique

    Return:
    ------
        mappings: (list of list of ints) contains mappings between queries responses
    """
    assert len(responses) == len(queries), "length of responses and queries do not match!"

    # Generate the mappings from responses and queries
    response_idx = unique_indexing(responses)
    query_idx = unique_indexing(queries)

    # create mapping from query-response indices
    mappings = []
    for one_query_idx, one_response_idx in zip(query_idx, response_idx):
        mappings.append([one_query_idx, one_response_idx])

    return mappings


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
                
    def create_df(self):
        """
        Create pandas DataFrame in a similar format to dataloader.py
        which may be used for finetuning and evaluation.

        Importantly, if there is many-to-many matches between Queries and Responses
        the returned dataframe will have duplicates

        Return:
        ------
            df: (pd.DataFrame) contains the columns query_string, processed_string, kb_name
        """
        # get processed string
        processed_string_series = ( self.responses.context_string.fillna('') + ' ' + self.responses.raw_string.fillna('')).str.replace('\n','')
        processed_string_series.name = 'processed_string'

        # transpose list of lists 
        # https://stackoverflow.com/questions/6473679/transpose-list-of-lists
        q_r_idx = list(map(list, zip(*self.mapping)))

        # Concat and create kb_name
        df = pd.concat([self.queries.iloc[q_r_idx[0]].reset_index(drop=True), 
                        processed_string_series.iloc[q_r_idx[1]].reset_index(drop=True)], 
                        axis=1)
        df = df.assign(kb_name = self.name)

        return df
        
    def json(self, hashkey=None):
        """
        Create json dict to use with flask endpoint
        """
        json_dict = {}

        if hashkey is not None:
            json_dict['hashkey'] = hashkey
        json_dict['kb_name'] = self.name

        json_dict['kb'] = {}
        json_dict['kb']['responses'] = self.responses.raw_string.tolist()
        json_dict['kb']['contexts'] = self.responses.context_string.tolist()

        if (len(self.queries) > 0) & (len(self.mapping) > 0):
            json_dict['kb']['queries'] = self.queries.query_string.tolist()
            json_dict['kb']['mapping'] = self.mapping

        return json_dict


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
        df = df.assign(context_string = '') if context_col == 'context_string' else df 
        df = df.rename(columns = {
                                   answer_col: 'raw_string', 
                                   context_col: 'context_string',
                                   query_col: 'query_string'
                                  })
            
        queries = []
        unique_responses_df = df.loc[~df.duplicated(), ['raw_string', 'context_string']].drop_duplicates().reset_index(drop=True)
        
        if query_col=='':
            # if there are no query columns
            # there will be no query or mappings to return
            # simply return unique responses now
            return unique_responses_df, pd.DataFrame(), []
        
        """
        Handle many-to-many matching between the queries and responses
            1. Get list of unique queries and responses
            2. Index the given queries and responses 
            3. Create mappings from the indices of these non-unique queries and responses
        """
        contexted_answer = df.loc[:,'context_string'].fillna('') + ' ' + df.loc[:,'raw_string'].fillna('')
        query_string_series = df.loc[:,'query_string']

        mappings = generate_mappings(contexted_answer, query_string_series)

        # get unique query strings
        unique_query_df = query_string_series.drop_duplicates().reset_index(drop=True)
        
        return kb(kb_name, unique_responses_df, unique_query_df, mappings)
    
    
    def parse_text(self, path, clause_sep='/n', inner_clause_sep='', 
                   query_idx=None, context_idx=None, 
                   kb_name=None):
        """
        Parse text file from kb path into query, response and mappings
        
        args:
        ----
            path: (str) path to txt file, 
                        it may also be the raw text, if there is no txt. extention
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

        if path.endswith('txt'):
            kb_name = kb_name if kb_name is not None else path.split('/')[-1].split('.')[0]
            
            """
            1. Parse the text into its fields
            """
            # read the text
            with open(path) as text_file:
                self.raw_text = text_file.read()
            
        else:
            self.raw_text = path

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
            
    def parse_csv(self, path, answer_col, query_col='', context_col='context_string', kb_name=''):
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

    def pdf_converter(self, PDF_file_path, header="", NumOfAppendix=0, kb_name='pdf_kb'):
        """
        Function to convert PDFs to Dataframe with columns as index number & paragraphs.

        Parameters
        ----------

        PDF_file_path : string
            The filename and path of pdf 

        header: string
            To remove the header in each page

        NumOfAppendix: int
            To remove the Appendix after the main content
            
        kb_name: str
            Name of returned kb object

        Returns
        -------------
        kb : kb object
        """

        raw = parser.from_file(PDF_file_path)
        s=raw["content"].strip()

        s=re.sub(header,"",s)
        s=s+"this is end of document."
        s=re.split("\n\nAPPENDIX ",s)
        newS=s[:len(s)-NumOfAppendix]
        s=' '.join(newS)

        s = re.sub('(\d)+(\-(\d)+)+',' newparagraph ',s)
        paragraphs=re.split("newparagraph", s)
        list_par=[]
        
        # (considered as a line)
        for p in paragraphs:
            if p is not None:
                if not p.isspace():  # checking if paragraph is not only spaces
                    list_par.append(p.strip().replace("\n", "")) # appending paragraph p as is to list_par
        
        list_par.pop(0)

        # pd.set_option('display.max_colwidth', -1)
        clause_df=pd.DataFrame(list_par, columns=['clause'])
        
        # convert to kb object
        responses, queries, mapping = self.parse_df(kb_name, clause_df, 'clause')
        return kb(kb_name, responses, queries, mapping) 


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

        SQL_Query = pd.read_sql_query('''SELECT dbo.query_labels.clause_id, dbo.query_labels.query_id, dbo.query_db.query_string, \
                                    dbo.kb_clauses.context_string, dbo.kb_clauses.processed_string, dbo.kb_clauses.raw_string, dbo.kb_raw.kb_name, dbo.kb_raw.type FROM dbo.query_labels \
                                    JOIN dbo.query_db ON dbo.query_labels.query_id = dbo.query_db.id \
                                    JOIN dbo.kb_clauses ON dbo.query_labels.clause_id = dbo.kb_clauses.id \
                                    JOIN dbo.kb_raw ON dbo.kb_clauses.raw_id = dbo.kb_raw.id''', conn)
        conn.close()
        
        kb_names = SQL_Query['kb_name'].unique() if len(kb_names) == 0 else kb_names
        
        kbs = []
        for kb_name in kb_names:
            
            kb_df = SQL_Query.loc[SQL_Query.kb_name == kb_name]

            # load name, responses, queries, mapping into kb object
            indexed_responses = kb_df.loc[:,['clause_id', 'raw_string', 'context_string']].drop_duplicates(subset=['clause_id']).fillna('') # fillna: not all responses have a context_string
            indexed_queries = kb_df.loc[:,['query_id', 'query_string']].drop_duplicates(subset=['query_id']).fillna('')
            mappings = generate_mappings(kb_df.processed_string, kb_df.query_string)
            
            kb_ = kb(kb_name, indexed_responses, indexed_queries, mappings)

            kbs.append(kb_)
        
        return kbs
