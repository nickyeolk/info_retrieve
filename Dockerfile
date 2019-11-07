FROM tensorflow/tensorflow:1.13.1-py3
ADD . /code
WORKDIR /code
RUN pip install flask \
&& pip install tensorflow-hub \
&& pip install pandas \
&& pip install scikit-learn \
&& pip install tf-sentencepiece \
&& pip install streamlit \
&& apt-get -y install wget \
&& wget https://finetunedweights.blob.core.windows.net/finetuned01/google_use_nrf_pdpa_tuned.tar.gz \
&& tar -zxvf google_use_nrf_pdpa_tuned.tar.gz
CMD ["streamlit", "run", "--server.port", "5000", "app2.py"]

