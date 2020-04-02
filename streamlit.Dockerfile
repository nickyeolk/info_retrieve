FROM tensorflow/tensorflow:latest-py3

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
&& curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
&& apt-get update \
&& ACCEPT_EULA=Y apt-get -y install msodbcsql17 unixodbc unixodbc-dev wget

COPY streamlit_requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ADD . /code 
WORKDIR /code 

RUN wget https://finetunedweights.blob.core.windows.net/finetuned02/variables.tar.gz \
&& tar -zxvf variables.tar.gz

CMD ["streamlit", "run", "--server.port", "5000","--server.headless","true", "--browser.serverAddress","0.0.0.0", "--server.enableCORS", "false",  "app2.py"]