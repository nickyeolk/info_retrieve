# info_retrieve
Framework for a information retrieval engine (QnA, knowledge base query, etc)  
Step 1: The knowledge base has to be separated into documents. Each document is an indexed unit of information e.g. a clause, a sentence, a paragraph.  
Step 2: The clauses (and query) are encoded with the same encoder (Infersent, Google USE, Google USE-QA)  
Step 3: A similarity score is calculated (cosine dist, arccos dist, dot product, nearest neighbors)  
Step 4: Clauses with the highest score (or nearest neighbors) are returned as the retrieved document  
  
The current use case it is being optimized for is the retrieval of clauses in a contract/terms and conditions document given some natural language query.

There is a potential for fine tuning following Yang et. al's (2018) paper on [learning textual similarity from conversations](https://arxiv.org/abs/1804.07754). A fully connected layer is inserted after the clauses are encoded to maximize the dot product between the transformed clauses and the encoded query.

# Testing
Currently, 3 sentence encoding models are compared against the test set of the [InsuranceQA corpus](https://github.com/shuzi/insuranceQA). Each test case consists of a question, and 100 possible answers, of which the correct answer is one or more of the 100 possible answers. Model evaluation metric is accuracy@k, where the score is 1 if the top k matches contain the target answer, and 0 otherwise.
  
|Model|acc@1|acc@2|acc@3|acc@4|acc@5|
|---|---|---|---|---|---|
|InferSent|0.083|0.134|0.1814|0.226|0.268|
|Google USE|0.251|0.346|0.427|0.481|0.534|
|Google USE-QA|**0.387**|**0.519**|**0.590**|**0.648**|**0.698**|
