"""
Flask app for GoldenRetriever
The script interfaces with a non-public db

The app may allow 2 methods:
    1. make_query
    2. save_feedback 

To use:
    python app_flask.py -db "db_cnxn_str.txt"
"""
import datetime
import pyodbc
import numpy as np
import pandas as pd
import pandas.io.sql as pds
from flask import Flask, jsonify, request
import argparse

from src.model import GoldenRetriever
from src.kb_handler import kb_handler


"""
Setup
"""
app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("-db", "--credentials", dest='dir',
                     default='db_cnxn_str.txt', 
                     help="directory of the pyodbc password string")
args = parser.parse_args()


class InvalidUsage(Exception):
    """
    Raises exception
    https://flask.palletsprojects.com/en/1.1.x/patterns/apierrors/
    """
    status_code = 400

    def __init__(self, message="query endpoint reqires three arguments: query, kbname", 
                 status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response




"""
SET UP
------
Caches the following:
    1. gr: model object to make query
    2. cursor: SQL connection
    3. get_kb_dir_id, get_kb_raw_id: dictionary to retrieve
                                     kb_dir_id and kb_raw_id 
                                     from kb_name (user provided string)
"""
# load the model and knowledge bases
gr = GoldenRetriever()
# gr.restore('./google_use_nrf_pdpa_tuned/variables-0')
kbh = kb_handler()
kbs = kbh.load_sql_kb(cnxn_path = args.dir, kb_names=['PDPA','nrf'])
print(kbs)
print(kbs[0])
gr.load_kb(kbs)
print(gr.kb.keys())

# make the SQL connection and cursor
conn = pyodbc.connect(open(args.dir, 'r').read())
cursor = conn.cursor()

# get kb_names to kb_id
kb_ref = pds.read_sql("""SELECT id, kb_name, directory_id  FROM dbo.kb_raw""", conn)
get_kb_dir_id = kb_ref.loc[:,['kb_name', 'directory_id']].set_index('kb_name').to_dict()['directory_id']
get_kb_raw_id = kb_ref.loc[:,['kb_name', 'id']].set_index('kb_name').to_dict()['id']

# get kb permissions
permissions = pds.read_sql("SELECT hashkey, kb_name FROM dbo.users \
                            LEFT JOIN dbo.kb_directory ON dbo.users.id = dbo.kb_directory.user_id \
                            LEFT JOIN kb_raw ON dbo.kb_directory.id = dbo.kb_raw.directory_id \
                           ", conn)
permissions = pd.DataFrame(np.array(permissions), columns = ['hashkey', 'kb_name']).set_index('hashkey')



"""
API endpoints:
--------------
    1. make_query
    2. feedback 
"""
@app.route("/query", methods=['POST'])
def make_query():
    """
    Main function for User to make requests to. 

    Args:
    -----
        user_id: (str, optional) identification; intended to be their hashkey 
                                 to manage exclusive knowledge base access.
        query: (str) query string contains their natural question
        kbname: (str) Name of knowledge base to query
        top_k: (int, default 5) Number of top responses to query. Currently kept at 5

    Return:
    -------
        Reply: (list) contains top_k string responses
        user_id: (int) contains id of the request to be used for when they give feedback
    """
    # 1. parse the request and get timestamp
    request_timestamp = datetime.datetime.now()
    request_dict = request.get_json()
    
    if not all([key in ['query', 'kb_name'] for key in request_dict.keys()]):
    #if not all([key in ['query', 'kbname', 'top_k'] for key in request_dict.keys()]):
        raise InvalidUsage()

    user_id = request_dict.get('user_id')
    query_string = request_dict["query"]
    kb_name = request_dict["kb_name"]
    # top_k = request_dict["top_k"]
    
    # # Manage KB access
    # try:
    #     if kb_name in permissions.loc[user_id].kb_name:
    #         pass
    #     else:
    #         raise InvalidUsage(f"Unauthorised or unfound kb: {user_id} tried to access {kb_name}")
    # except:
    #     raise InvalidUsage(f"Unrecognized hashkey: {user_id}")



    # 2. model inference
    reply, reply_index = gr.make_query(query_string, 
                                       # top_k=int(top_k), 
                                       top_k = 5,
                                       index=True, kb_name=kb_name)


    # 3. log the request in SQL
    # id, created_at, query_string, user_id, kb_dir_id, kb_raw_id, Answer1, Answer2, Answer3, Answer4, Answer5
    rowinfo = [request_timestamp, query_string] 
    # append user_id
    rowinfo.append(user_id) 
    # append kb_dir_id
    rowinfo.append(get_kb_dir_id[kb_name])   
    # append kb_raw_id
    rowinfo.append(get_kb_raw_id[kb_name])
    # returned answers clause_id
    rowinfo.extend(gr.kb[kb_name].responses.clause_id.iloc[reply_index].tolist())

    conn.execute('INSERT INTO query_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rowinfo)
    conn.commit()

    # 4. Return response to user
    # return id of latest log request to user for when they give feedback
    cursor.execute("SELECT id from dbo.query_log WHERE query_string = ?", query_string)
    query_log_table = cursor.fetchall()
    # oldest_request_id = query_log_table[-1][0] if query_log_table[-1][0] is not None else -1
    # current_request_id = oldest_request_id + 1
    current_request_id = query_log_table[-1][0]

    return jsonify(responses=reply, query_id=current_request_id)


@app.route("/feedback", methods=['POST'])
def save_feedback():
    """
    Retrieve feedback from end users

    args:
    ----
        query_id: (int) specifies the query to raise feedback for
        is_correct: (list) list fo booleans for true or false
    """
    request_timestamp = datetime.datetime.now()
    request_dict = request.get_json()

    if not all([key in ['query_id', 'is_correct'] for key in request_dict.keys()]):
        raise InvalidUsage("request requires 'query_id', 'is_correct")

    # 1. parse the request
    query_id = request.get_json()["query_id"]
    is_correct = request.get_json()["is_correct"]
    is_correct = is_correct+[False]*(5-len(is_correct)) if len(is_correct) < 5 else is_correct # ensures 5 entries

    # log the request in SQL
    rowinfo = [request_timestamp]
    rowinfo.append(query_id)
    rowinfo.extend(is_correct[:5]) # ensures only 5 values are logged

    conn.execute('INSERT INTO feedback_log VALUES (?, ?, ?, ?, ?, ?, ?)', rowinfo)
    conn.commit()

    return jsonify(message="Success")




if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
