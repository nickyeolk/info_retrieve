FROM tensorflow/tensorflow:latest-py3
ADD . /code
WORKDIR /code
RUN pip install tensorflow-hub \
&& pip install pandas \
&& pip install scikit-learn \
&& pip install tensorflow==2.0.0 \
# && pip install tensorflow-gpu==2.0.0 \
&& pip install tensorflow-estimator==2.0.1 \
&& pip install tensorflow-text==2.0.1  \
# && pip install tensorflow-addons==0.6.0 \
&& pip install transformers \
&& curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
&& curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
&& apt-get update \
&& ACCEPT_EULA=Y apt-get -y install msodbcsql17 \
&& apt-get -y install unixodbc unixodbc-dev \
&& pip install pyodbc \
&& pip install tika \
&& pip install flask \
&& pip install waitress \
&& pip install azure-storage-blob \
&& apt-get -y install wget \
&& wget https://finetunedweights.blob.core.windows.net/finetuned02/variables.tar.gz \
&& tar -zxvf variables.tar.gz
CMD ["python", "app_flask.py", "-db", "db_cnxn_str.txt"]