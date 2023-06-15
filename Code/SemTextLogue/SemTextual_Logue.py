import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# tokenizers
def tokenize_src(text):
    return [tok for tok in text.split()]


def tokenize_trg(text):
    return [tok for tok in text.split()]


# fields
SRC = Field(
    tokenize=tokenize_src,
    use_vocab=True,
    init_token="<sos>",
    eos_token="<eos>",
    lower=False,
)
TRG = Field(
    tokenize=tokenize_trg,
    use_vocab=True,
    init_token="<sos>",
    eos_token="<eos>",
    lower=False,
)

#for finding the embedding
embed_model = SentenceTransformer('bert-base-nli-mean-tokens')


# Split sentences for detokenizing
def split_sentences(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
limit=2

# train loop
def train(model,bse_model, iterator,optimizer, bse_optimizer, criterion, clip, epoch, total, trg_eos_token,SRC,TRG,embed_model):
    #add: bse train 

    model.train()
    bse_model.train()

    epoch_loss = 0
    epoch_bse_loss=0
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
        
        #add: add seperate optimizer for bse and nlg tokyo university 
        optimizer.zero_grad()
        bse_optimizer.zero_grad()

        output = model(src, trg[:-1, :])  # [trg_len-1, batch_size]

        batchSize = output.shape[1]  
        #print(batchSize)
        
        new_trg = trg.t()
        new_src = src.t()

        trg = trg[1:].reshape(-1)
        output=output.reshape(-1,output.shape[2])

        output=torch.chunk(output,batchSize,dim=0)
        trg=torch.chunk(trg,batchSize,dim=0)

        #creation of tensors loss and reward dif    
        batch_loss_ce = torch.FloatTensor(batchSize).zero_().to(device)
        batch_reward_diff= torch.FloatTensor(batchSize).zero_().to(device)
        batch_loss_diff= torch.FloatTensor(batchSize).zero_().to(device)


        loss=0
        for iter,(cur_trg,cur_new_trg,cur_new_src,cur_output) in enumerate(zip(trg,new_trg,new_src,output)):
            
            # predicted_bert_score=0
            bse_input = cur_output.clone().to(device)
            bse_input = F.pad(bse_input, (0,0,0,80-cur_output.shape[0]), mode='constant', value=0.0)

            predicted_bert_score= bse_model(bse_input)

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

            src_detokenized = [SRC.vocab.itos[i] for i in cur_new_src]
            src_detokenized = " ".join(src_detokenized)
            src_embedding = embed_model.encode(src_detokenized, show_progress_bar=False)

            
            cosine_similarity = np.dot(output_embedding, trg_embedding)/(norm(output_embedding)*norm(trg_embedding))
            cosine_context_similiarity = np.dot(output_embedding,src_embedding)/(norm(output_embedding)*norm(src_embedding))

            bert_score = cosine_similarity
            bert_context_score= cosine_context_similiarity

            bert_score = 0.6*bert_score+0.4*bert_context_score

            cur_loss = criterion(cur_output, cur_trg)
            loss += cur_loss
            batch_loss_ce[iter] = cur_loss

            predicted_bert_score= torch.sigmoid(predicted_bert_score)
            reward_diff = bert_score-predicted_bert_score
            loss_diff = 1.0-predicted_bert_score
            batch_reward_diff[iter]=reward_diff
            batch_loss_diff[iter]= loss_diff

        bse_loss = torch.mul(batch_reward_diff,batch_reward_diff).sum()/batchSize
        ce_loss = torch.sum(batch_loss_ce)/batchSize
        rl_loss = torch.mul(batch_loss_diff,batch_loss_ce).sum()/batchSize

        # if epoch less than 10 ce_loss and bse_loss seperate training
        if  epoch<=10:
            loss=bse_loss+ce_loss
        else:
            lam = 0.005
            loss= lam*ce_loss + (1-lam)*rl_loss + bse_loss
        
        loss.backward()

        if i % 1000 == 0 or i == len(iterator) - 1:
            logging.info(
                f"Training Epoch: {epoch}/{total}\tBatch: {i+1}/{len(iterator)}\tLoss: {loss :.3f}\tPPL: {math.exp(loss) :,.3f}\tTime Elapsed (mins): {(time.time()-start_time)/60 :,.3f}"
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)


        optimizer.step()
        bse_optimizer.step()
        epoch_loss += loss.item()
        epoch_bse_loss += bse_loss.item()
    
    return epoch_loss / len(iterator), epoch_bse_loss / len(iterator)


# evaluate loop
def evaluate(model,bse_model, iterator, criterion, trg_eos_token,SRC,TRG,embed_model):
    model.eval()
    
    epoch_loss = 0
    epoch_bse_loss=0
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
        new_src = src.t()

        trg = trg[1:].reshape(-1)
        output=output.reshape(-1,output.shape[2])

        output=torch.chunk(output,batchSize,dim=0)
        trg=torch.chunk(trg,batchSize,dim=0)

        #creation of tensors loss and reward dif    
        batch_loss_ce = torch.FloatTensor(batchSize).zero_().to(device)
        batch_reward_diff= torch.FloatTensor(batchSize).zero_().to(device)
        batch_loss_diff= torch.FloatTensor(batchSize).zero_().to(device)

        loss=0
        for iter,(cur_trg,cur_new_trg,cur_new_src,cur_output) in enumerate(zip(trg,new_trg,new_src,output)):
            
            # predicted_bert_score=0
            bse_input = cur_output.clone().to(device)
            bse_input = F.pad(bse_input, (0,0,0,80-cur_output.shape[0]), mode='constant', value=0.0)

            predicted_bert_score= bse_model(bse_input)

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
            
            src_detokenized = [SRC.vocab.itos[i] for i in cur_new_src]
            src_detokenized = " ".join(src_detokenized)
            src_embedding = embed_model.encode(src_detokenized, show_progress_bar=False)

            
            cosine_similarity = np.dot(output_embedding, trg_embedding)/(norm(output_embedding)*norm(trg_embedding))
            cosine_context_similiarity = np.dot(output_embedding,src_embedding)/(norm(output_embedding)*norm(src_embedding))

            bert_score = cosine_similarity
            bert_context_score= cosine_context_similiarity

            bert_score = 0.6*bert_score+0.4*bert_context_score

            cur_loss = criterion(cur_output, cur_trg)
            loss += cur_loss
            batch_loss_ce[iter] = cur_loss

            predicted_bert_score= torch.sigmoid(predicted_bert_score)

            reward_diff = bert_score-predicted_bert_score
            loss_diff = 1.0 - predicted_bert_score
            batch_reward_diff[iter]=reward_diff
            batch_loss_diff[iter]= loss_diff

        bse_loss = torch.mul(batch_reward_diff,batch_reward_diff).sum()/batchSize
        ce_loss = torch.sum(batch_loss_ce)/batchSize
        rl_loss = torch.mul(batch_loss_diff,batch_loss_ce).sum()/batchSize

        #if epoch less than 10 ce_loss and bse_loss seperate training
        lam = 0.005
        loss= lam*ce_loss + (1-lam)*rl_loss + bse_loss

        if i % 1000 == 0 or i == len(iterator) - 1:
            logging.info(
                f"Validation:\tBatch: {i+1}/{len(iterator)}\tLoss: {loss :.3f}\tPPL: {math.exp(loss) :,.3f}\tTime Elapsed (mins): {(time.time()-start_time)/60 :,.3f}"
            )

        epoch_loss += loss.item()
        epoch_bse_loss += bse_loss.item()

    return epoch_loss / len(iterator) , epoch_bse_loss / len(iterator)

