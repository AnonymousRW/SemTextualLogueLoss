import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from transformers import *
import argparse
import logging
from torchtext.legacy.data import BucketIterator, Field, TabularDataset
from torchtext.data.metrics import bleu_score
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import math
import time
import random
from tqdm import tqdm
import os
import csv
from rouge import Rouge

import tensorflow as tf
from tensorflow.keras.layers import Dense
from os.path import join
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# import sentencepiece as spm
# import torch.nn.functional as F
# import spacy
# from torchtext.datasets import Multi30k


# Global variables and settings
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="[%d-%b-%y %H:%M:%S]",
    level=logging.INFO,
)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
logging.info(f"Device: {device}\n")

# seed
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Split sentences for detokenizing
def split_sentences(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

limit=2

#train loop 
def train(model, iterator, optimizer, criterion, clip, epoch, total, trg_eos_token,SRC,TRG,embed_model):

    model.train()
    epoch_loss = 0
    start_time = time.time()
    for i, batch in enumerate(iterator):
        src = batch.src
        # src: [src_len, batch_size]
        trg = batch.trg
        # trg: [trg_len, batch_size]


        max_len = model.encoder.max_len

        if trg.shape[0] > max_len:
            # logging.info(f"{trg_1.shape}")
            trg = trg[:max_len-1, :]
            # logging.info(f"{trg_1.shape}")
            pad_token = trg_eos_token
            pad_tensor = torch.IntTensor([pad_token]).repeat(1, trg.shape[1]).to(device)
            trg = torch.vstack((trg, pad_tensor))
            # logging.info(f"{trg_1.shape, pad_tensor.shape}")
        

        optimizer.zero_grad()

        output = model(src, trg[:-1, :])  # [trg_len-1, batch_size]

        batchSize = output.shape[1]  
        #print(batchSize)
        
        new_trg = trg.t()

        trg = trg[1:].reshape(-1)
        output=output.reshape(-1,output.shape[2])

        output=torch.chunk(output,32,dim=0)
        trg=torch.chunk(trg,32,dim=0)

        loss=0
        for cur_trg,cur_new_trg,cur_output in zip(trg,new_trg,output):

            # output: [trg_len-1 * batch_size, trg_vocab_size]
            output_argmax = torch.argmax(cur_output, dim=1)

            #print('OA ',output_argmax)
            output_detokenized = [TRG.vocab.itos[i] for i in output_argmax]
            output_detokenized = " ".join(output_detokenized)
            # print('predicted-',output_detokenized)
            output_embedding = embed_model.encode(output_detokenized, show_progress_bar=False)

            trg_detokenized = [TRG.vocab.itos[i] for i in cur_new_trg]
            trg_detokenized = " ".join(trg_detokenized)
            trg_sentences = list(split_sentences(trg_detokenized, batchSize))
            # print('actual-',trg_detokenized)
            trg_embedding = embed_model.encode(trg_detokenized, show_progress_bar=False)
            # trg_embedding: [batch_size * embedding size(768)]
            
            cosine_similarity = np.dot(output_embedding, trg_embedding)/(norm(output_embedding)*norm(trg_embedding))
    
            loss += (1-cosine_similarity)*criterion(cur_output, cur_trg)

    
    
        loss=loss/batchSize
        loss.backward()
        if i % 1000 == 0 or i == len(iterator) - 1:
            logging.info(
                f"Training Epoch: {epoch}/{total}\tBatch: {i+1}/{len(iterator)}\tLoss: {loss :.3f}\tPPL: {math.exp(loss) :,.3f}\tTime Elapsed (mins): {(time.time()-start_time)/60 :,.3f}"
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# evaluate loop
def evaluate(model, iterator, criterion, trg_eos_token,SRC,TRG,embed_model):

    model.eval()
    epoch_loss = 0
    start_time = time.time()
    for i, batch in enumerate(iterator):
        src = batch.src
        # src: [src_len, batch_size]
        trg = batch.trg
        # trg: [trg_len, batch_size]
        

        max_len = model.encoder.max_len

        if trg.shape[0] > max_len:
            # logging.info(f"{trg_1.shape}")
            trg = trg[:max_len-1, :]
            # logging.info(f"{trg_1.shape}")
            pad_token = trg_eos_token
            pad_tensor = torch.IntTensor([pad_token]).repeat(1, trg.shape[1]).to(device)
            trg = torch.vstack((trg, pad_tensor))
            # logging.info(f"{trg_1.shape, pad_tensor.shape}")

        output = model(src, trg[:-1, :])  # [trg_len-1, batch_size]
        batchSize = output.shape[1]  
        #print(batchSize)
        
        new_trg = trg.t()

        trg = trg[1:].reshape(-1)
        output=output.reshape(-1,output.shape[2])

        output=torch.chunk(output,32,dim=0)
        trg=torch.chunk(trg,32,dim=0)

        loss=0

        for cur_trg,cur_new_trg,cur_output in zip(trg,new_trg,output):

            # output: [trg_len-1 * batch_size, trg_vocab_size]
            output_argmax = torch.argmax(cur_output, dim=1)

            #print('OA ',output_argmax)
            output_detokenized = [TRG.vocab.itos[i] for i in output_argmax]
            output_detokenized = " ".join(output_detokenized)
            # print('predicted-',output_detokenized)
            output_embedding = embed_model.encode(output_detokenized, show_progress_bar=False)

            trg_detokenized = [TRG.vocab.itos[i] for i in cur_new_trg]
            trg_detokenized = " ".join(trg_detokenized)
            trg_sentences = list(split_sentences(trg_detokenized, batchSize))
            # print('actual-',trg_detokenized)
            trg_embedding = embed_model.encode(trg_detokenized, show_progress_bar=False)
            # trg_embedding: [batch_size * embedding size(768)]
            
            cosine_similarity = np.dot(output_embedding, trg_embedding)/(norm(output_embedding)*norm(trg_embedding))
            # print(cosine_similarity)
            # print("hello")
            # loss += criterion(cur_output, cur_trg)*(-cosine_similarity)
            loss += (1-cosine_similarity)*criterion(cur_output, cur_trg)

    
        # loss=1/cosine_similarity
        #print(loss)
        loss=loss/batchSize
        if i % 1000 == 0 or i == len(iterator) - 1:
            logging.info(
                f"Validation:\tBatch: {i+1}/{len(iterator)}\tLoss: {loss :.3f}\tPPL: {math.exp(loss) :,.3f}\tTime Elapsed (mins): {(time.time()-start_time)/60 :,.3f}"
            )

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
