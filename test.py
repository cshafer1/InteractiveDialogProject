from ast_attendgru_xtra import AstAttentionGRUModel as xtra
import json
import io
import os
import re
import random
import numpy as np
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfTransformer

from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding, Conv1D, Masking, Flatten
convocabsize = 1000
ansvocabsize = 1000
quesvocabsize = 1000

queslen = 100
conlen = 50
anslen = 100

traincon = []
trainques = []
trainans = []


with open("contextFinal.txt") as f:
	traincon = f.readlines()

with open("questionsFinal.txt") as f:
	trainques = f.readlines()

with open("answersFinal.txt") as f:
	trainans = f.readlines()


#tokenizers

contok = Tokenizer(filters='', lower=False, num_words=convocabsize, oov_token="UNK")
contok.fit_on_texts(traincon)

questok = Tokenizer(filters='', lower=False, num_words=quesvocabsize, oov_token="UNK")
questok.fit_on_texts(trainques)

anstok = Tokenizer(filters='', lower=False, num_words=ansvocabsize, oov_token="UNK")
anstok.fit_on_texts(trainans)

#tokenize sequences

trainquest = questok.texts_to_sequences(trainques)
traincont = contok.texts_to_sequences(traincon)

#pad sequences

trainquest = pad_sequences(trainquest, padding="post", truncating="post", maxlen=queslen)
traincont = pad_sequences(traincont, padding="post", truncating="post", maxlen=conlen)

config = dict()

config['convocabsize'] = 1000
config['ansvocabsize'] = 1000
config['quesvocabsize'] = 1000
config['anslen'] = anslen
config['conlen'] = conlen
config['queslen'] = queslen
config['multigpu'] = False
config['batch_size'] = 5

astmodel = xtra(config)

config, model = astmodel.create_model()

print(model.summary())
print(len(trainques), len(traincon))
i = 0

#cons = []
#quess = []
#anss = []
#wordss = []



for line in trainans:
	splitline = line.split(" ")
	words = splitline[0]
	for wordi in range(len(splitline)-1):
		twords = anstok.texts_to_sequences([words])
		twords = pad_sequences(twords, padding="post", truncating="post", maxlen=anslen)	

		con = contok.texts_to_sequences([traincon[i]])
		ques = questok.texts_to_sequences([trainques[i]])

		con = pad_sequences(con, padding="post", truncating="post", maxlen=conlen)
		ques = pad_sequences(ques, padding="post", truncating="post", maxlen=queslen)

		twordn = anstok.texts_to_sequences([splitline[wordi + 1]])
		ans = [0] * 1000
		ans[twordn[0][0]] = 1
		#print(len(twords), len(con), words)
		#cons.append(con)
		#quess.append(ques)
		#anss.append(ans)
		#wordss.append(twords)
		model.fit([con, twords, ques], np.array([ans]), batch_size=config['batch_size'], epochs=5, verbose=1)
		words += " " + splitline[wordi+1]
	i += 1


q = "how many points did andy duncan get in the 1951 season"
c = "357  357  1951.0  Andy Duncan  F-C  28.0  BOS  14.0 NaN NaN NaN  0.292 NaN  0.55 NaN NaN NaN NaN NaN NaN NaN NaN NaN -0.4  0.1 -0.3 NaN NaN NaN NaN NaN NaN  7.0  40.0  0.175 NaN NaN NaN  7.0  40.0  0.175  0.175  15.0  22.0  0.682 NaN NaN  30.0  8.0 NaN NaN NaN  32.0  29.0"

a = ""

tq = questok.texts_to_sequences([q])
tc = contok.texts_to_sequences([c])

tq = pad_sequences(tq, padding="post", truncating="post", maxlen=queslen)
tc = pad_sequences(tc, padding="post", truncating="post", maxlen=conlen)

ta = anstok.texts_to_sequences([a])
ta = pad_sequences(ta, padding="post", truncating="post", maxlen=anslen)

for i in range(anslen):
	yhat = model.predict([tc, ta, tq])
	ta[0][i] = np.argmax(yhat, axis = 1)[0]

text = anstok.sequences_to_texts(ta)
text = re.findall('(<s>.*</s>)', text[0])[0]
print("\n\n\n\n\n")
print(text)
