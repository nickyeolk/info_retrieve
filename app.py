from flask import Flask, jsonify, request, render_template
from src.model import GoldenRetriever
app = Flask(__name__)

gr = GoldenRetriever()
gr.restore('./google_use_nrf_tuned/variables-0')
# gr.load_kb(path_to_kb='./data/aiap.txt', is_faq=True)

@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def get_prediction():
    # data = request.form.get("question")
    kb=request.args.get("kbname")
    data=request.args.get("question")
    if kb=='aiap':
        gr.load_kb(path_to_kb='./data/aiap.txt', is_faq=True)
    if kb=='resale_tnc':
        gr.load_kb(path_to_kb='./data/resale_tnc.txt')
    reply, score = gr.make_query(data, top_k=1)
    print('request received!')
    return render_template('index.html', prediction = reply[0], rating = score[0], question = data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
