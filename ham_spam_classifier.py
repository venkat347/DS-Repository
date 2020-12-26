#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd

gmail_message = pd.read_csv(r'C:\Users\Venkat\Desktop\DataScience_Sep_3rd\krish NLP files\smsspamcollection\SMSSpamCollection', sep='\t',
                           names=["label", "gmail_message"])

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
WL= WordNetLemmatizer()
corpus = []
for i in range(0, len(gmail_message)):
    review = re.sub('[^a-zA-Z]', ' ', gmail_message['gmail_message'][i])
    review = review.lower()
    review = review.split()
    
    review = [WL.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the TF- IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv = TfidfVectorizer(max_features=2500)
X = tfidfv.fit_transform(corpus).toarray()

Y=pd.get_dummies(gmail_message['label'])

Y=Y.iloc[:,1].values


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

Y_pred=spam_detect_model.predict(X_test)















# In[28]:


corpus


# In[31]:


X


# In[40]:


Y


# In[43]:


X_train, X_test, Y_train, Y_test 


# In[46]:


Y_pred


# In[47]:


from sklearn.metrics import confusion_matrix
conf = confusion_matrix(Y_test,Y_pred)


# In[48]:


conf


# In[49]:


from sklearn.metrics import accuracy_score
accuracy =accuracy_score(Y_test,Y_pred)


# In[50]:


accuracy


# In[ ]:




