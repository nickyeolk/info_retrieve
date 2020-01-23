"""

"""
import datetime
import pyodbc
import pandas as pd
from flask import Flask, jsonify, request

from src.model import GoldenRetriever

app = Flask(__name__)


"""
1. SET UP
"""
# load the model
gr = GoldenRetriever()
# gr.restore('./google_use_nrf_pdpa_tuned/variables-0')
gr.load_sql_kb("db_cnxn_str.txt", kb_names=['PDPA','nrf'])

# make the SQL connection and cursor
cnxn_path = "db_cnxn_str.txt"
conn = pyodbc.connect(open(cnxn_path, 'r').read())
cursor = conn.cursor()



"""
2. make_query and feedback 
"""
@app.route("/make_query", methods=['POST'])
def train_model():

    request_dict = request.get_json()
    request_timestamp = datetime.datetime.now()

    if not all([key in ['query', 'kbname', 'top_k'] for key in request_dict.keys()]):
        return jsonify(message='request requires "query", "kbname", "top_k"', query_id=1)

    # 1. parse the request
    data = request.get_json()["query"]
    kb = request.get_json()["kbname"]
    top_k = request.get_json()["top_k"]

    # 2. model inference
    reply, score = gr.make_query(data, top_k=int(top_k), kb_name=kb)

    # find id of latest log request
    # this id is returned to the user
    # for when they give feedback
    cursor.execute("SELECT MAX(id) from dbo.query_log")
    query_log_table = cursor.fetchall()
    if query_log_table[0][0] is not None:
        oldest_request_id = query_log_table[0][0]
        current_request_id = oldest_request_id + 1
    else:
        oldest_request_id=-1
        current_request_id = oldest_request_id + 1

    # log the request in SQL
    

    return jsonify(responses=reply, query_id=current_request_id)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
