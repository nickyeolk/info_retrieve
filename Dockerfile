FROM tensorflow/tensorflow:1.13.1-py3
ADD . /code
WORKDIR /code
RUN pip install flask \
&& pip install tensorflow-hub \
&& pip install pandas \
&& pip install scikit-learn \
&& pip install tf-sentencepiece
CMD ["python", "app.py"]

