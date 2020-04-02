FROM tensorflow/tensorflow:latest-py3

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
&& curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
&& apt-get update \
&& ACCEPT_EULA=Y apt-get -y install msodbcsql17 unixodbc unixodbc-dev 

COPY api_requirements.txt ./api_requirements.txt
RUN pip install -r requirements.txt

RUN wget https://finetunedweights.blob.core.windows.net/finetuned02/variables.tar.gz \
&& tar -zxvf variables.tar.gz

CMD ["python", "app_flask.py", "-db", "db_cnxn_str.txt"]