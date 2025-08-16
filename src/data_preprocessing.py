import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import spacy

_nlp = None

#1. removing the HTML & Bracketed text
def noiseremoval_text(text):
  soup = BeautifulSoup(text, "html.parser")
  text = soup.get_text()
  text = re.sub('\[[^]]*\]', '',text)
  return text

def lowercase_and_clean(text):
    # Lowercase
    text = text.lower()
    # Remove numbers and special characters except spaces
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def handle_negations(text):
    # Simple negation handling by joining 'not' + next word with underscore
    pattern = re.compile(r'\b(not|no|never|none|n\'t)\s+(\w+)')
    text = pattern.sub(lambda x: x.group(1) + '_' + x.group(2), text)
    return text

def remove_white_space(text):
    text = text.strip()    
    # Replace multiple spaces, tabs, or newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

def get_spacy_model():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp

#Define a faster lemmatization function using nlp.pipe
def lemmatize_texts(texts):
    nlp = get_spacy_model()
    lemmatized_texts = []
    for doc in nlp.pipe(texts, batch_size=500, n_process=2):
        lemmatized_texts.append(" ".join([token.lemma_ for token in doc]))
    return lemmatized_texts


def remove_stopwords(text):
    tokenizer = ToktokTokenizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_short_tokens(text, min_length=3):
    tokens = text.split()
    filtered_tokens = [token for token in tokens if len(token) >= min_length]
    return ' '.join(filtered_tokens)

def stem_text(text):
    ps = PorterStemmer()
    tokens = text.split()
    stemmed = ' '.join([ps.stem(token) for token in tokens])
    return stemmed

def preprocess_pipeline(text,use_stemming=False):
    text = noiseremoval_text(text)
    text = lowercase_and_clean(text)
    text = handle_negations(text)
    #text = remove_white_space(text)
    text = lemmatize_texts(text)
    text = remove_stopwords(text)
    text = remove_short_tokens(text)
    if use_stemming:
        text= stem_text(text)
    
    return text 

