#Sean Healy
import sys
import json
import re
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


qlen = 100
clen = 400
alen = 100


args = sys.argv[1:]
single = False
if len(args) < 4:
	print("missing arguments")
	sys.exit(0)
while len(args) and args[0].startswith('-') and len(args[0]) > 1:
	arg = args.pop(0)
	if arg == '-n':
		lineNumber = int(args.pop(0))
	elif arg == '-q':
		question = args.pop(0)
	else:
		print("incorrect arguments")
		sys.exit(0)

print("\n\n",question, "\n\n")
# load in tokenizers
f = open("data/ques_tok.json")
json_string = json.dumps(json.load(f))
qtokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
f.close()

f = open("data/con_tok.json")
json_string = json.dumps(json.load(f))
ctokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
f.close()

f = open("data/ans_tok.json")
json_string = json.dumps(json.load(f))
atokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
f.close()

model = tf.keras.models.load_model("qamodel.h5")

f = open("contextFinal.txt")

context = f.readlines()[lineNumber]

f.close()
print(context)
tquestion = qtokenizer.texts_to_sequences([question])
tcontext = ctokenizer.texts_to_sequences([context])

tquestion = pad_sequences(tquestion, padding="post", truncating="post", maxlen=qlen)
tcontext = pad_sequences(tcontext, padding="post", truncating="post", maxlen=clen)


answer = ""
tanswer = atokenizer.texts_to_sequences([answer])
tanswer = pad_sequences(tanswer, padding="post", truncating="post", maxlen=alen)
print(tanswer)
print()
for i in range(alen):
	yhat = model.predict([tcontext, tanswer, tquestion])
	tanswer[0][i] = np.argmax(yhat, axis = 1)[0]

text = atokenizer.sequences_to_texts(tanswer)[0]
#print(lineNumber, text)
if lineNumber < 500:
	text = re.sub("<year>", str(int(float(context.split("  ")[2]))), text)
elif lineNumer < 1000:
	text = re.sub("<year>", str(int(float(re.findall("[0-9]+", text)[0]))), text)
else:
	text = re.sub("<year>", str(int(float(context.split("  ")[1]))), text)
text = re.findall('([^<]*</s>)', text)[0]
print("\n\n\n\n\n")
print(text)
