#!/usr/bin/env python
# coding: utf-8
# #  NoWorkflow - Summer of Reproducibility
# This is the exploratory notebook for the noworkflow project in Summer of Reproducibility - 2023.
# In[1]:

import pandas as pd

# ## Data ingestion
# 
# #### Some highlights about the data ingestion process:
# 
# Dataset originally from Kaggle challenge [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

# In[2]:


df = pd.read_csv('/Users/jesselima/noworkflow/sor_noworkflow/dataset/creditcard.csv',  encoding='utf-8')
print("Reading dataset")
# ### Feature engineering

# 3. The dataset shows a clas inbalanced and the issue needs to be addressed with some rebalancing technique. One approach to the problem is oversampling the ones until classes are 1:1 proportional. Here I did all the work with SMOTE package

# In[23]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

features = df.columns.drop(['Time', 'Class'])
df_train, df_test = train_test_split(df, test_size=0.30, stratify=df['Class'], random_state=654321)

smote = SMOTE(random_state=654321, sampling_strategy='not majority')
X_resampled, y_resampled = smote.fit_resample(df_train[features], df_train['Class'])

X_train = X_resampled
y_train = y_resampled

#X_train = df_train
#y_train = df_test

y_test = df_test.Class.astype('int')
X_test = df_test[features]



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


clf = LogisticRegression(random_state=654321, solver='liblinear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("ROC = %f, F1 = %f " % (roc_auc_score(y_test, y_pred), f1_score(y_test, y_pred)))


# #### Testing different scoring methods


