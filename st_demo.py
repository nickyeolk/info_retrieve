import streamlit as st
import pandas as pd
import time

from src.model import GoldenRetriever
from src.kb_handler import kb_handler
from src.importance import importance_by_erasure, partial_highlight
from src import SessionState

# Init per-session persistent state variabless
state = SessionState.get(fetch=False, 
                        prediction=[], 
                        interpret0=False, interpret1=False, interpret2=False, interpret3=False, interpret4=False, 
                        importance_frame_0=None,importance_frame_1=None,importance_frame_2=None,importance_frame_3=None,importance_frame_4=None, 
                        k0=5,k1=5,k2=5,k3=5,k4=5,
)

# 0. CACHED FUNCTIONS FOR INIT AND INTERPRET
@st.cache(allow_output_mutation=True)
def init():
    retriever = GoldenRetriever()
    # retriever.restore('./2/')

    # parse text into kb
    kbh = kb_handler()
    resale_tnc = kbh.parse_text('data/resale_tnc.txt', 
                                   clause_sep='\n\n', inner_clause_sep="\n", 
                                   context_idx=0,
                                   kb_name = 'resale_tnc'
                                  )
    aiap = kbh.parse_text('data/aiap.txt', clause_sep='\n\n', inner_clause_sep="\n", query_idx=0)
    pdpa = kbh.parse_csv('data/pdpa.csv', 
                         answer_col='answer', query_col='question', context_col='meta', 
                         kb_name='pdpa')
    covid19 = kbh.parse_csv('data/covid19.csv',
                        answer_col='answer', query_col='question', context_col='meta', 
                        kb_name='covid19')

    # load kbs
    retriever.load_kb(resale_tnc)
    retriever.load_kb(aiap)
    retriever.load_kb(pdpa)
    retriever.load_kb(covid19)

    # load SQL db
    # kbs = kbh.load_sql_kb(cnxn_path = "db_cnxn_str.txt", kb_names=['PDPA','nrf'])
    # retriever.load_kb(kbs)
    return retriever

gr = init()


# 1. INTRODUCTION AND INFO
st.title('GoldenRetriever')
st.header('This Information Retrieval demo allows you to query FAQs, T&Cs, or your own knowledge base in natural language.')
st.markdown('View the source code [here](https://github.com/nickyeolk/info_retrieve)!')
st.markdown('Visit our [community](https://makerspace.aisingapore.org/community/ai-makerspace/) and ask us a question!')
kb_to_starqn = {'pdpa':"Can an organization retain the physical NRIC?",
                'resale_tnc':"How much is the option fee?",
                'aiap':"Do I need to pay for the program?",
                'covid19':'what is coronavirus?',
                # 'nrf':"Can I vire from EOM into travel?",
                'raw_kb':"What do you not love?"}

# 2. USER INPUT FOR KNOWLEDGE BASES AND QUERY
def format_func(kb_name):
    namedicts={'covid19':'COVID-19',
                'pdpa':'PDPA',
                'resale_tnc':'HDB Resale',
                'aiap':'AIAP',
                # 'nrf':'NRF',
                'raw_kb':'Paste Raw Text'}
    return namedicts[kb_name]
kb = st.selectbox('Select Knowledge Base', options=['covid19', 'pdpa', 'resale_tnc', 'aiap', 'raw_kb'],
                    format_func=format_func)
if kb=='raw_kb':
    kb_raw = st.text_area(label='Paste raw text (terms separated by empty line)', 
                        value="""I love my chew toy!\n\nI hate Mondays.\n""")
top_k = st.radio('Number of Results', options=[1,2,3], index=2)
data = st.text_input(label='Input query here', value=kb_to_starqn[kb])


# 3. PRESENTING RESULTS
if st.button('Fetch', key='fetch'):
    
    # 1. Sets fetch to True
    # 2. Resets all interpret states to False
    state.fetch = True
    for i in range(5):
        setattr(state, f"interpret{i}", False)

# if state.fetch == True or (data != kb_to_starqn[kb]): 
#     # first condition is separate from st.button('Fetch'), to keep the state persistent
#     # Second condition ensures the answer will not appear right away

    if kb=='raw_kb':
        # load raw text kb
        kbh = kb_handler()
        raw_kb = kbh.parse_text(kb_raw,
                                clause_sep='\n\n',
                                kb_name = 'raw_kb'
                                )
        gr.load_kb(raw_kb)
        
    prediction, scores = gr.make_query(data, top_k=int(top_k), kb_name=kb)
    state.prediction = prediction
    time.sleep(2)
    
qn_string="""<h3><text>Question: </text>{}</h3>""".format(data)
st.markdown(qn_string, unsafe_allow_html=True)

if len(state.prediction)>0:
    for ansnum, result in enumerate(state.prediction):
        
        anshead_string = """<h3><text>Answer {}</text></h3>""".format(ansnum+1)

        markdown_context = st.empty()
        markdown_answer = st.empty()

        if st.button('interpret', key=f"key_{ansnum}"):
            setattr(state, f"interpret{ansnum}", True) 
            setattr(state, f"importance_frame_{ansnum}", importance_by_erasure(gr, result, data))

        if state.fetch:
            if getattr(state, f"interpret{ansnum}"):
                setattr(state, f"k{ansnum}", st.slider("Most important words", 0, len(result.split())) )
                markdown_prediction = partial_highlight(getattr(state, f"importance_frame_{ansnum}"), k=getattr(state, f"k{ansnum}"))
            else:
                markdown_prediction = result

            # set the markdown object
            markdown_context.markdown(anshead_string, unsafe_allow_html=True)
            reply_string="""<table>"""
            lines = [line for line in markdown_prediction.split('\n') if line]
            for line in lines:
                reply_string += """<tr>"""
                tabledatas = line.split(';;')
                for tabledata in tabledatas:
                    if len(tabledatas)>1:
                        line_string = """<td>{}</td>""".format(tabledata)
                    else:
                        line_string = """<td colspan=42>{}</td>""".format(tabledata)
                    reply_string += line_string
                reply_string += """</tr>"""
            reply_string+="""</table><br>"""

        markdown_answer.markdown(reply_string, unsafe_allow_html=True)

st.markdown(
"""
<details><summary>Sample sentences</summary>
<strong>COVID-19</strong>
<p>Why are schools still continuing with CCAs and PE lessons?</p>
<strong>PDPA</strong>
<p>How long can an organisation retain its customers' personal data?</p>
<strong>HDB resale terms and conditions</strong>
<p>Do I need to pay back CPF?</p>
<strong>AIAP</strong>
<p>What will be covered during the program?</p>
<strong>Raw text </strong><a href="https://www.straitstimes.com/asia/east-asia/china-wants-centralised-digital-currency-after-bitcoin-crackdown" target="_blank">China Digital Currency</a><i> (Select all, copy, and paste into raw text box)</i>
<p>Which electronic payment gateways support the currency?</p>
</details>"""
, unsafe_allow_html=True)
