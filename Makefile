setup:
	conda env create -f environment.yml
	mkdir google_use_qa
	curl -L "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1?tf-hub-format=compressed" | tar -zxvC ./google_use_qa

experiment:
	conda env create -f environment.yml
	./init.sh