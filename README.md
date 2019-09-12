# info_retrieve
Framework for a information retrieval engine (QnA, knowledge base query, etc)  
Step 1: The knowledge base has to be separated into documents. Each document is an indexed unit of information e.g. a clause, a sentence, a paragraph.  
Step 2: The clauses (and query) are encoded with the same encoder (Infersent, Google USE, Google USE-QA)  
Step 3: A similarity score is calculated (cosine dist, arccos dist, dot product, nearest neighbors)  
Step 4: Clauses with the highest score (or nearest neighbors) are returned as the retrieved document  
  
The current use case it is being optimized for is the retrieval of clauses in a contract/terms and conditions document given some natural language query.

There is a potential for fine tuning following Yang et. al's (2018) paper on (learning textual similarity from conversations)[https://arxiv.org/abs/1804.07754]. A fully connected layer is inserted after the clauses are encoded to maximize the dot product between the transformed clauses and the encoded query.
