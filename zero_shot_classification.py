#https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
#!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv

#!pip install transformers

from transformers import pipeline
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import textwrap
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

classifier = pipeline("zero-shot-classification", device=0)

classifier("This is a great movie", candidate_labels=["positive", "negative"])

text = "Due to the presence of isoforms of its components, there are 12 versions of AMPK in mammals, each of which can have different tissue localizations, and different functions under different conditions. AMPK is ragulated allosterically and by post-translational modification, which work together"
classifier(text, candidate_labels=["biology", "math", "geology"])

data_frame = pd.read_csv('bbc_text_cls.csv')
len(data_frame)

data_frame.sample(frac=1).head()

labels = list(set(data_frame['labels']))
labels

print(textwrap.fill(data_frame.iloc[1024]['text']))

data_frame.iloc[1024]['labels']

classifier(data_frame.iloc[1024]['text'], candidate_labels=labels)

# Takes about 55 min
predictions = classifier(data_frame['text'].tolist(), candidate_labels=labels)

predicted_labels = [d['labels'][0] for d in predictions]
data_frame['predicted_labels'] = predicted_labels

print("Accurancy: ", np.mean(data_frame['predicted_labels'] == data_frame['labels']))

# Convert prediction probs into an NxK matrix according to original label order
N = len(data_frame)
K = len(labels)
label2idx = {v:k for k, v in enumerate(labels)}

probs = np.zeros((N, K))
for i in range(N):
  # loop through labels and scores in corresponding order
  d = predictions[i]
  for label, score in zip(d['labels'], d['scores']):
    k = label2idx[label]
    probs[i, k] = score


int_labels = [label2idx[x] for x in data_frame['labels']]

int_preds = np.argmax(probs, axis=1)
cm = confusion_matrix(int_labels, int_preds, normalize='true')

# Scikit-learn is transitioning to V1 but its not available on colab
# The changes modify how confusion matrixes are plotted
def plot_cm(cm):
  df_cm = pd.DataFrame(cm, index=labels, clumns=labels)
  ax = sn.heatmap(df_cm, annot=True, fmt='.2g')
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Target")

plot_cm(cm)



