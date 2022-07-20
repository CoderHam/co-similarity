import cohere
import pandas as pd
import os
from nltk import tokenize
from typing import List
from tqdm import tqdm
import numpy as np

# nltk.download('punkt')

# initialize the Cohere Client with an API Key
API_KEY = os.getenv('COHERE_API_KEY')
co = cohere.Client(API_KEY)

def extract_embedding(sentences: List[str]):
    response = co.embed(model='small', texts=sentences)
    
    return response.embeddings

# Extract sentence embeddings and store them in .npy files
records = pd.read_csv("bgg_2000.csv")
progress_bar = tqdm(records.iterrows())
for idx, record in progress_bar:
    progress_bar.set_description("Extracting embedding for game %s" % idx)
    description = record['description']
    sentences = tokenize.sent_tokenize(description)
    embeddings = extract_embedding(sentences)
    embeddings = np.save('%s.npy' % idx, embeddings)
