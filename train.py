from ast_attendgru_xtra import AstAttentionGRUModel as xtra
import json
import io
import os
import re
import random
import time
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

# tokenizers

contok = Tokenizer(filters='', lower=False, num_words=convocabsize, oov_token="UNK")
contok.fit_on_texts(traincon)

questok = Tokenizer(filters='', lower=False, num_words=quesvocabsize, oov_token="UNK")
questok.fit_on_texts(trainques)

anstok = Tokenizer(filters='', lower=False, num_words=ansvocabsize, oov_token="UNK")
anstok.fit_on_texts(trainans)

# tokenize sequences

trainquest = questok.texts_to_sequences(trainques)
traincont = contok.texts_to_sequences(traincon)

# pad sequences

trainquest = pad_sequences(trainquest, padding="post", truncating="post", maxlen=queslen)
traincont = pad_sequences(traincont, padding="post", truncating="post", maxlen=conlen)

config = dict()

i = 0

cons = []
quess = []
anss = []
wordss = []
startTime = time.time()
wordsDone = 0
avgTime = time.time()
for line in trainans:
    wordsDone += 1
    avgTime = (time.time() - startTime) / wordsDone
    toCompletion = (len(trainans) - wordsDone) * avgTime
    print("Estimated Time to Completion: " + str(toCompletion))
    splitline = line.split(" ")
    words = splitline[0]
    for wordi in range(len(splitline) - 1):
        twords = anstok.texts_to_sequences([words])
        twords = pad_sequences(twords, padding="post", truncating="post", maxlen=anslen)
        con = contok.texts_to_sequences([traincon[i]])
        ques = questok.texts_to_sequences([trainques[i]])

        con = pad_sequences(con, padding="post", truncating="post", maxlen=conlen)
        ques = pad_sequences(ques, padding="post", truncating="post", maxlen=queslen)

        twordn = anstok.texts_to_sequences([splitline[wordi + 1]])
        ans = [0] * 1000
        ans[twordn[0][0]] = 1
        wordss.append(ans)
        anss.append(twords)
        quess.append(ques)
        cons.append(con)
        words += " " + splitline[wordi + 1]
    i += 1
# cons = np.array(cons)
print(len(cons))

config['convocabsize'] = 1000
config['ansvocabsize'] = 1000
config['quesvocabsize'] = 1000
config['anslen'] = anslen
config['conlen'] = conlen
config['queslen'] = queslen
config['multigpu'] = False
config['batch_size'] = 32
config['consamount'] = len(cons)
config['quesamount'] = len(quess)
config['ansamount'] = len(anss)
astmodel = xtra(config)
config, model = astmodel.create_model()

print(model.summary())
print(len(anss))


def gen_call():
    def arg_free_gen():
        for index, q in enumerate(cons):
            yield {"input_con": cons[index], "input_ques": quess[index], "input_ans": anss[index]}, np.asarray(
                wordss[index]).reshape(1,-1)

    return arg_free_gen





generator = gen_call()
data = tf.data.Dataset.from_generator(generator, output_types=({"input_con": tf.int32, "input_ques": tf.int32,
                                                                "input_ans": tf.int32}, tf.int32),
                                     output_shapes=({"input_con":(1, conlen), "input_ques":(1, queslen),
													 "input_ans":(1, anslen)}, (1,1000)))
model.fit(data, epochs=5, batch_size=config['batch_size'], workers=2, use_multiprocessing=True)

q = "how many points did scottie pippen get in the <year> season"
c = "15872  15872  2002.0  Scottie Pippen*  SF  36.0  POR  62.0  60.0  1996.0  14.9  0.497  0.295  0.244  4.5  14.5  9.5  28.7  2.7  1.3  20.5  19.1 NaN  1.2  2.6  3.7  0.09 NaN  0.5  1.7  2.2  2.1  246.0  599.0  0.411  54.0  177.0  0.305  192.0  422.0  0.455  0.456  113.0  146.0  0.774  77.0  244.0  321.0  363.0  101.0  35.0  171.0  162.0  659.0"
a = ""
model.save("qamodel.h5")
tq = questok.texts_to_sequences([q])
tc = contok.texts_to_sequences([c])

tq = pad_sequences(tq, padding="post", truncating="post", maxlen=queslen)
tc = pad_sequences(tc, padding="post", truncating="post", maxlen=conlen)

ta = anstok.texts_to_sequences([a])
ta = pad_sequences(ta, padding="post", truncating="post", maxlen=anslen)

for i in range(anslen):
    yhat = model.predict([tc, ta, tq])
    ta[0][i] = np.argmax(yhat, axis=1)[0]
print(ta)
text = anstok.sequences_to_texts(ta)
print(text)
text = re.findall('(<s>.*</s>)', text[0])[0]
print("\n\n\n\n\n")
print(text)
