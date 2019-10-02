import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def question_cleaner(df_query):
    kb=([int(xx) for xx in (df_query[3].iloc[0]).split(' ')])
    gt = [int(xx) for xx in (df_query[2].iloc[0]).split(' ')]
    ct=0
    negg=0
    withans=[]
    for ii in range(len(df_query)):
        kb=[int(xx) for xx in (df_query[3].iloc[ii]).split(' ')]
        gt = [int(xx) for xx in (df_query[2].iloc[ii]).split(' ')]
        if bool(set(gt) & set(kb)):
            withans.append(ii)
        else:
            negg+=1
    print('total:{}, removed:{}, remainder:{}'.format(len(df_query), negg, len(withans)))
    return df_query.iloc[withans]

def display_qn_and_ans(df_query, df_doc, index=0):
    kb=[int(xx) for xx in (df_query[3].iloc[index]).split(' ')]
    gt = [int(xx) for xx in (df_query[2].iloc[index]).split(' ')]
    print('Question is: {}'.format(df_query['text'].iloc[index]))
    print('Answer index: ', gt)
    print('Answers: ', df_doc.loc[gt, 'text'].values)

def read_txt(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.readlines()
    return text

def split_txt(text, qa=False):
    '''Splits a text document into clauses based on whitespaces. 
    Additionally, reads a faq document by assuming that the first line is a question 
    between each whitespaced group'''
    condition_terms = []
    stringg=''
    for tex in text:
        if (tex=='\n'):
            if (stringg != ''):
                condition_terms.append(stringg)
                stringg=''
            else: pass
        else: stringg+=tex
    if qa:
        condition_context = [x.split('\n')[0] for x in condition_terms]
        condition_terms = [' '.join(x.split('\n')[1:]) for x in condition_terms]
    condition_terms=[x.replace('\n', '. ') for x in condition_terms]
    if qa:
        return condition_terms, condition_context
    else: return condition_terms

def aiap_qna(question, answer_array, aiap_qa, model, k=1):
    similarity_score=cosine_similarity(answer_array, model.predict([question], type='query'))
    sortargs=np.flip(similarity_score.argsort(axis=0))
    sortargs=[x[0] for x in sortargs]
    sorted_ans=[]
    for indx in range(k):
        sorted_ans.append(aiap_qa[sortargs[indx]])
    return sorted_ans, sortargs, similarity_score

def aiap_qna_quickscore(aiap_context, answer_array, aiap_qa, model, k=1):
    '''Quickly scores the model against the aiap qna dataset. 
    This function works because the order of questions and answers are synched in the list'''
    score=0
    for ii, qn in enumerate(aiap_context):
        _, sortargs, simscore = aiap_qna(qn, answer_array, aiap_qa, model, k)
        if bool(set([ii]) & set(sortargs[:k])):
            score+=1
    return score/len(aiap_context)

def ranker(model, question_vectors, df_query, df_doc):
    '''for model evaluation on InsuranceQA datset'''
    predictions=[]
    gts=[]
    for ii, question_vector in enumerate(question_vectors):
        kb=[int(xx) for xx in (df_query[3].iloc[ii]).split(' ')]
        gt = [int(xx) for xx in (df_query[2].iloc[ii]).split(' ')]
        doc_vectors = model.predict(df_doc.loc[kb]['text'].tolist())
        cossim = cosine_similarity(doc_vectors, question_vector.reshape(1, -1))
        sortargs=np.flip(cossim.argsort(axis=0))
        returnedans = [kb[jj[0]] for jj in sortargs]
        predictions.append(returnedans)
        gts.append(gt)
    return predictions, gts
        
def scorer(predictions, gts, k=3):
    '''for model evaluation on InsuranceQA datset'''
    'returns score@k'
    score=0
    total=0
    for gt, prediction in zip(gts, predictions):
        if bool(set(gt) & set(prediction[:k])):
            score+=1
        total+=1
    return score/total