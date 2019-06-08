import nltk
import numpy as np
import random
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from SpellChecker import final
f = pd.read_csv('qa_friendly.csv')
final_dict = dict(zip(f.Questions, f.Answers))

final_dict = dict((k.lower().strip(), v.lower().strip()) for k,v in final_dict.items())

raw = final_dict.keys()
raw = list(raw)
raw = [x.lower() for x in raw]
raw = '. '.join(raw)

# f=open('test.txt','r',errors = 'ignore')
# raw=f.read()
#
# raw=raw.lower()# converts to lowercase
# print(raw)


nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words


lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        # robo_response = 'do you think you could ever love me?'
        final_response = final_dict[robo_response[:-1]]
        return final_response

flag=True
print("GROOT: I am GROOT. How may I help you? If you want to exit, type Rocket!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    #print(spellchecker.(user_response))
    if(user_response!='Rocket'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("GROOT: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("GROOT: "+greeting(user_response))
            else:
                print("GROOT: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("GROOT: I am GROOT")
