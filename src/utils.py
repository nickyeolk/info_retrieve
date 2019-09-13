import pandas as pd
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