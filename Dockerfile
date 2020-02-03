FROM tensorflow/tensorflow:latest-py3
ADD . /code
WORKDIR /code
RUN pip install flask \
&& pip install tensorflow-hub \
&& pip install pandas \
&& pip install scikit-learn \
&& pip install tensorflow==2.0.0   \
&& pip install tensorflow-estimator==2.0.1  \
&& pip install tensorflow-text==2.0.1  \
&& pip install tensorflow-hub==0.7.0   \
&& pip install tensorflow-gpu==2.0.0 \
&& pip install tensorflow-addons==0.6.0 \
&& pip install streamlit \
&& pip install pyodbc
&& apt-get -y install wget \
&& wget https://finetunedweights.blob.core.windows.net/finetuned01/google_use_nrf_pdpa_tuned.tar.gz \
&& tar -zxvf google_use_nrf_pdpa_tuned.tar.gz
CMD ["streamlit", "run", "--server.port", "5000","--server.headless","true", "--browser.serverAddress","0.0.0.0", "--server.enableCORS", "false",  "app2.py"]