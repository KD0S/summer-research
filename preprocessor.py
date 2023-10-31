from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('law-ai/InLegalBERT')

class Preprocessing():
	def __init__(self, path):
		self.data = path
		self.val_data = './data/train_data.csv'
		self.test_data = './data/test_data.csv'							
		self.max_len = 256
		
	def load_data(self):
		df_train = pd.read_csv(self.data)
		df_test = pd.read_csv(self.test_data)
		df_val = pd.read_csv(self.val_data)
		X_train = df_train['TEXT'].values
		X_test = df_test['TEXT'].values
		X_val =  df_val['TEXT'].values
		self.y_train = df_train['IMPORTANT'].values
		self.y_test = df_test['IMPORTANT'].values
		self.y_val = df_val['IMPORTANT'].values
		self.x_train = self.process(X_train)
		self.x_test = self.process(X_test)
		self.x_val = self.process(X_val)
		# self.x_train = X_train
		# self.x_test = X_test

		
	def process(self, x):
		X = [(tokenizer(text, padding='max_length', max_length = self.max_len, truncation=True, return_tensors="tf")['input_ids']) for text in x]
		X = np.asarray(X)
		return X
		


	
