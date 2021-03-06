# **Twitter Sentiment Analysis**

### **Problem Statement and Business Case**
  - Understanding the Natural Language Processing (NLP) works by converting text or words into numbers and are trained to Machine Learning Model to make predictions.
  - Predictions could be Sentiment inferred from social media posts and product reviews.
  - Machine Learning based Sentiment analysis is crucial for companies to automatically predict whether the customers are satisfied or not.
  - Understanding that the process could be done automatically without manually reviewing thousands of tweets and customers reviews by humans.
  - In this Project, I will prepare a Model that will analyze thousands of Twitter tweets to predict people's sentiment.
  
![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598323284/1_ztmvin.jpg)

### **Exploratory Data Analysis**
- Performed various Data Exploration such as Statistical Exploration, Data Visualization, Imputation, Feature Preprocessing, Count Vectorization, Plotting Word Cloud and so on.

**Snapshot of Data Visualization**

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598323937/2_x8jtqz.png)
![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598324017/3_wf4v2j.png)

**Snapshot of the Word Cloud**

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598324126/4_bzxlqr.png)

### **Preparing the Model**

**Naive Bayes**
- In statistics, Naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong independence assumptions between the features. They are among the simplest Bayesian network models.

**Training Naive Bayes Classifier**

```javascript
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
```

**Snapshot of Model Evaluation**

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598324479/5_jjmzcg.png)
