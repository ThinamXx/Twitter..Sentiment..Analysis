# **Twitter Sentiment Analysis**

### **Problem Statement and Business Case**
  - Understanding the Natural Language Processing (NLP) works by converting text or words into numbers and are trained to Machine Learning Model to make predictions.
  - Predictions could be Sentiment inferred from social media posts and product reviews.
  - Machine Learning based Sentiment analysis is crucial for companies to automatically predict whether the customers are satisfied or not.
  - Understanding that the process could be done automatically without manually reviewing thousands of tweets and customers reviews by humans.
  - In this Project, I will prepare a Model that will analyze thousands of Twitter tweets to predict people's sentiment.
  
[Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598323284/1_ztmvin.jpg)

### **Importing the Dependencies**

```javascript
# Downloading and loading the Libraries and Dependencies
# !pip install jupyterthemes
# !pip install WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from IPython.display import display
from jupyterthemes import jtplot
jtplot.style(theme = "monokai", context = "notebook", ticks = True, grid = False)
from wordcloud import WordCloud

import warnings
warnings.filterwarnings("ignore")
import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
```
