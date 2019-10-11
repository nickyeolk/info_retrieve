from flask import Flask, jsonify, request
from src.model import GoldenRetriever
app = Flask(__name__)

# test_ans={'answer':'This is the answer', 'confidence':0.5}
gr = GoldenRetriever()
gr.restore('./google_use_nrf_tuned/variables-0')
gr.load_kb(path_to_kb='./data/aiap.txt', is_faq=True)

@app.route('/', methods=['POST', 'GET'])
def get_prediction():
    # data = request.form.get("question")
    data=request.args.get("question")
    reply, _ = gr.make_query(data, top_k=1)
    print('request received!')
    return reply[0]

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
