"""
Help visualize word importances in GoldenRetriever

Methodology adopted from:
    Li, Jiwei, Will Monroe, and Dan Jurafsky. 
    "Understanding neural networks through representation erasure." 
    arXiv preprint arXiv:1612.08220 (2016).

Sample code in ipython kernel:
-----------
    from IPython.core.display import display, HTML
    from ipywidgets import interact

    from src.importance import importance_by_erasure, partial_highlight
    from src.model import GoldenRetriever

    gr = GoldenRetriever()

    response = "Coronaviruses are a large family of viruses causing illnesses ranging from the common cold to pneumonia (a more severe lung infection). A new coronavirus strain has been identified in Wuhan, China. The Coronavirus Disease 19 (COVID-19) has caused cases of severe pneumonia in China and cases have been exported to other countries and cities."
    query = "what is coronavirus?"
    print(query)

    word_importance = importance_by_erasure(gr, 
                                            response,
                                            query,
                                        )
    def f_(k):
        partially_highlighted_html = partial_highlight(word_importance, k=k)
        display(HTML(partially_highlighted_html))
    interact( f_, k=(1, len(word_importance)) );
"""

import re
import string

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def add_period(word):
    if word in string.punctuation:
        return word
    else:
        return ' ' + word

def add_period_html(word):
    if ('>' in word) & ('<' in word):
        if re.findall(">(.*?)<", word)[0] in string.punctuation:
            return word
        else:
            return ' '+word
    else:
        return add_period(word)

def parse_string(string):
    """
    Parse sentence string into:
        1. its words and punctuations
        2. List of idx
    """
    list_of_words_and_puncs = re.findall(r'''[\w']+|[.,!?;-~-{}`´_<=>:/@*()&'$%#"]''', string) 
    list_of_word_idx = [i for i, word in enumerate(list_of_words_and_puncs)]
    
    return list_of_words_and_puncs

def reconstruct(list_of_substring):
    """
    Reconstruct a list of substrings back together
    """
    reconstructed = ''
    
    for substring in list_of_substring:
        reconstructed += substring if substring in string.punctuation else ' ' + substring
    
    return reconstructed.strip()
    
def importance_by_erasure(model, response_string, query_string, verbose=False):
    """
    Calculates the importance of each word by erasure
    
    Args:
    ----
        gr: Model object with predict() method
        response_string: (str) contains the response to visualize
        query_string: (str) contains the query to visualize
    
    Return:
    ------
        word_importance: (pd.Series) contains the relative importance 
                                     of each word, 
                                     indexed by word position.
        
        
    Example pd.Series output for 3 word response:
    -------------
        0     0.01083826
        1    -0.09676583
        2    -0.30163735
        dtype: object
    """
    # get original encodings and similarities
    encoded_query = model.predict(query_string, type='query')
    encoded_original_response = model.predict(response_string, type='response')
    original_similarity = cosine_similarity(encoded_original_response, encoded_query)[0]

    # Iterate through each word in the response
    # measuring the difference in similarity upon erasure
    list_of_words_and_puncs = parse_string(response_string)
    word_importance = {}
    for i, word in enumerate( list_of_words_and_puncs ):

        # create perturbed string
        perturbed = list_of_words_and_puncs.copy()
        del perturbed[i]
        perturbed = reconstruct(perturbed)

        # calculate similarity of perturbed string to ques
        encoded_perturbed = model.predict(perturbed, type='response')
        perturbed_similarity = cosine_similarity(encoded_perturbed, encoded_query)[0]
        
        # calculate similarity
        importance = (original_similarity - perturbed_similarity) / original_similarity
        word_importance[i] = importance[0]
        
        # log
        if verbose:
            print('\n')
            print(f"{i}: Erasing {word} changes similarity from {original_similarity[0]}" \
            f"to {perturbed_similarity[0]}")
            print(perturbed)
        
    # return dataframe of the words and their importances
    word_importance_df = {}
    word_importance_df['word'] = {idx:word for idx,word in enumerate(list_of_words_and_puncs)}
    word_importance_df['importance'] = word_importance
    word_importance_df = pd.DataFrame(word_importance_df).round(4)
    
    # add space to front of every 
    # word_importance_df.word =  word_importance_df.word.apply(add_period)
        
    return word_importance_df

def partial_highlight(word_importance_, k=-1, sns_palette_str = "YlOrBr"):
    """
    Partially highlight some text
    
    args:
    -----
        word_importance: (pd.DataFrame) contains words and their respective importances
        k: (int) specify the number top most important words to highlight
        sns_palette_str: (str, default="YlOrBr") seaborn palette to use 
    
    Return:
    ------
        partially_highlighted_html: (str) text of highlighted sentence 
    """
    if k<0:
        k = len(word_importance_)
    
    # generate hexes
    no_of_intervals = len(word_importance_)
    palette = sns.color_palette(sns_palette_str, no_of_intervals).as_hex()
    hexes = palette[:k][::-1]

    # create new columns
    word_importance_ = word_importance_.assign(working_string = word_importance_.word)
    # word_importance_.working_string = word_importance_.working_string.apply(add_period)
    
    # get idx of most importance words, 
    # Quite impt, this idx will also determine the order of the color
    idx_of_words_to_color = np.argsort(-1*word_importance_.importance)[:k]
    
    # add highlight to words
    word_importance_.loc[idx_of_words_to_color, "working_string"] = ["<font style='background-color:" + hex_ + "'>" for hex_ in hexes] \
                                                                    + word_importance_.loc[idx_of_words_to_color, "working_string"] + '</font>'
                                                                    
    # If it is a mere punctuation, 
    word_importance_.working_string = word_importance_.working_string.apply(add_period_html)
    partially_highlighted_html = ''.join(word_importance_.working_string.tolist())

    return partially_highlighted_html


def gen_hex_for_word_importances(word_importance, sns_palette_str = "YlOrBr"):
    """
    Generates hex codes sorted according to word importances
    """
    # get palette hex codes
    no_of_intervals = len(word_importance)
    palette = sns.color_palette(sns_palette_str, no_of_intervals).as_hex()
    
    # sort hex according to word importances
    word_importance_idx = np.argsort(word_importance.importance).tolist()
    sorted_hex = [x for _,x in sorted(zip(word_importance_idx,palette))]
    
    return sorted_hex

def gen_html_of_highlighted_text(word_importance, sns_palette_str = "YlOrBr"):
    """
    Generates a html text that highlights words based on their importance
    """
    # get hex strings
    sorted_hex = gen_hex_for_word_importances(word_importance, sns_palette_str)
    
    # construct html with styled fonts
    highlighted_string = ["<font style='background-color:" + hex_ + "'>" + add_period(word) + "</font>"  for hex_, word in  zip(sorted_hex, word_importance.word)]
    highlighted_string = ' '.join(highlighted_string)

    return highlighted_string