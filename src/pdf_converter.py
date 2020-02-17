import os
import re
import pandas as pd
from tika import parser
import argparse

def pdf_converter(PDF_file_path, header="", NumOfAppendix=0):
    """
    Function to convert PDFs to Dataframe with columns as index number & paragraphs.

    Parameters
    ----------

    PDF_file_path : string
        The filename and path of pdf 

    header: string
        To remove the header in each page

    NumOfAppendix: int
        To remove the Appendix after the main content

    Returns
    -------------
    df : Dataframe
    """

    raw = parser.from_file(PDF_file_path)
    s=raw["content"].strip()

    s=re.sub(header,"",s)
    s=s+"this is end of document."
    s=re.split("\n\nAPPENDIX ",s)
    newS=s[:len(s)-NumOfAppendix]
    s=' '.join(newS)

    s = re.sub('(\d)+(\-(\d)+)+',' newparagraph ',s)
    paragraphs=re.split("newparagraph", s)
    list_par=[]
    
    # (considered as a line)
    for p in paragraphs:
        if p is not None:
            if not p.isspace():  # checking if paragraph is not only spaces
                list_par.append(p.strip().replace("\n", "")) # appending paragraph p as is to list_par
    
    list_par.pop(0)

    # pd.set_option('display.max_colwidth', -1)
    clause_df=pd.DataFrame(list_par, columns=['clause'])

    return clause_df

# if __name__ == "__main__":
#     main()

#python pdf_converter.py "./PDF_files/Guidelines to MAS Notice 626  April 2015.pdf" "GUIDELINES TO MAS NOTICE 626 ON PREVENTION OF MONEY LAUNDERING AND \nCOUNTERING THE FINANCING OF TERRORISM" "2"

