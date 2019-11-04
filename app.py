from flask import Flask, jsonify, request, render_template
from src.model import GoldenRetriever
app = Flask(__name__)

gr = GoldenRetriever()
gr.restore('./google_use_nrf_pdpa_tuned/variables-0')
gr.load_kb(path_to_kb='./data/aiap.txt', is_faq=True, kb_name='aiap')
gr.load_kb(path_to_kb='./data/resale_tnc.txt', kb_name='resale_tnc')
gr.load_kb(path_to_kb='./data/fund_guide_tnc_full.txt', kb_name='nrf')
gr.load_csv_kb(path_to_kb='./data/pdpa.csv', cutoff=196, kb_name='pdpa')

@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def get_prediction():
    # data = request.form.get("question")
    kb = request.args.get("kbname")
    data = request.args.get("question")
    top_k = request.args.get("top_k")
    reply, score = gr.make_query(data, top_k=int(top_k), kb_name=kb)
    print('request received!')
    return render_template('index.html', prediction = reply, rating = score, question = data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
