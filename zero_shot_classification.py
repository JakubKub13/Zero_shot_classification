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

