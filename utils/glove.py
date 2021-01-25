from tensorflow import keras
import numpy as np
from tensorflow.python.keras.initializers.initializers_v2 import Constant


def readGloveFile(gloveFile):
    with open(gloveFile, 'r', encoding="utf8") as f:
        wordToGlove = {}  # map from a token (word) to a Glove embedding vector
        wordToIndex = {}  # map from a token to an index
        indexToWord = {}  # map from an index to a token
        count_fail = 0
        for line in f:
            record = line.strip().split()
            token = record[0]  # take the token (word) from the text line
            try:
                wordToGlove[token] = np.array(record[1:],
                                              dtype=np.float32)  # associate the Glove embedding vector to a that token (word)
            except:
                print('line skipped : ', count_fail)
                count_fail += 1
                pass

        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  # 0 is reserved for masking in Keras (see above)
            wordToIndex[tok] = kerasIdx  # associate an index to a token (word)
            indexToWord[kerasIdx] = tok  # associate a word to a token (word). Note: inverse of dictionary above

    return wordToIndex, indexToWord, wordToGlove


# Create Pretrained Keras Embedding Layer
def createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, isTrainable=False):
    vocabLen = len(wordToIndex)  # adding 1 to account for masking
    embDim = next(iter(wordToGlove.values())).shape[0]  # 300 in our case
    print('createPretrainedEmbeddingLayer \nembDim : {}, wordToIndex : {}'.format(len(wordToIndex), embDim))
    embeddingMatrix = np.zeros((vocabLen, embDim))  # initialize with zeros
    for word, index in wordToIndex.items():
        try:
            embeddingMatrix[index, :] = wordToGlove[word]  # create embedding: word index to Glove word embedding
        except IndexError:
            pass
    print('for out')

    embeddingLayer = keras.layers.Embedding(len(wordToIndex), embDim, embeddings_initializer=Constant(embeddingMatrix),
                                            trainable=isTrainable)
    return embeddingLayer