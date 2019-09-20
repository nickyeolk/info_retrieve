#/bin/bash
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
mkdir fastText
curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip fastText/crawl-300d-2M.vec.zip -d fastText/
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
python -m nltk.downloader punkt
git clone https://github.com/facebookresearch/InferSent.git
cd data
git clone https://github.com/shuzi/insuranceQA.git
gzip -r ./insuranceQA/V2/
mkdir google_use_qa
curl -L "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1?tf-hub-format=compressed" | tar -zxvC ./google_use_qa