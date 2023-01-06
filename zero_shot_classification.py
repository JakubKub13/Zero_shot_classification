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