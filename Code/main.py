import importlib
import itertools
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


# Transformer Encoder
class Transformer_Encoder(nn.Module):
    def __init__(
        self, input_dim, emb_dim, max_len, ff_dim, n_heads, n_layers, dropout, device
    ):
        super(Transformer_Encoder, self).__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)
        self.max_len=max_len

        # token and position embeddings
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_len, emb_dim)

        # TransformerEncoderLayer and TransformerEncoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.transformer_encode = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer, num_layers=n_layers
        )

    # forward
    def forward(self, src, src_mask):
        # src: [src_len, batch_size]
        # src_mask: [batch_size, src_len]
        src_len = src.shape[0]
        batch_size = src.shape[1]

        # creating pos tensor
        pos = (
            torch.arange(0, src_len).unsqueeze(1).repeat(1, batch_size).to(self.device)
        )
        # pos: [src_len, batch_size]

        src_embedded = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )
        # src_embedded: [src_len, batch_size, emb_dim]

        output = self.transformer_encode(
            src=src_embedded, src_key_padding_mask=src_mask
        )
        # output: [src_len, batch_size, emb_dim]

        return output

# Transformer Decoder
class Transformer_Decoder(nn.Module):
    def __init__(
        self, input_dim, emb_dim, max_len, ff_dim, n_heads, n_layers, dropout, device
    ):
        super(Transformer_Decoder, self).__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)
        self.fc_out = nn.Linear(emb_dim, input_dim)

        # token and position embeddings
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_len, emb_dim)

        # TransformerDecoderLayer and TransformerDecoder
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.transformer_decode = nn.TransformerDecoder(
            decoder_layer=transformer_decoder_layer, num_layers=n_layers
        )

    # forward
    def forward(self, trg, encoder_output, trg_mask, src_mask):
        # trg: [trg_len, batch_size]
        # encoder_output: [src_len, batch_size, emb_dim]
        # trg_mask: [trg_len, trg_len]
        # src_mask: [batch_size, src_len]
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]

        # creating pos tensor
        pos = (
            torch.arange(0, trg_len).unsqueeze(1).repeat(1, batch_size).to(self.device)
        )
        # pos: [trg_len, batch_size]

        trg_embedded = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        )
        # trg_embedded: [trg_len, batch_size, emb_dim]

        output = self.transformer_decode(
            tgt=trg_embedded,
            memory=encoder_output,
            tgt_mask=trg_mask,
            memory_key_padding_mask=src_mask,
        )
        # output: [trg_len, batch_size, emb_dim]
        # print(output.shape)

        output = self.fc_out(output)
        # output: [trg_len, batch_size, input_dim]

        return output


# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    # we are making src_key_padding_mask: [batch_size, src_len]
    # we are creating a bool tensor where the value True indicates position not allowed to attend
    # so we are making bool values to false where the actual information is
    def make_src_mask(self, src):
        src_mask = (src == self.src_pad_idx).permute(1, 0)
        # permute(1, 0) because src_key_padding_mask expect the tensor in shape: [batch_size, src_len]
        return src_mask.to(self.device)

    # we are making trg_mask: [trg_len, trg_len]
    # copied from 'Transformer.generate_square_subsequent_mask(sz)' code
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(
            0, 1
        )  # triu returns upper triangualr part of matrix
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask.to(self.device)

    # forward
    def forward(self, src, trg):
        # src: [src_len ,batch_size]
        # trg: [trg_len, batch_size]
        trg_len = trg.shape[0]

        # generate masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.generate_square_subsequent_mask(trg_len)
        # print(trg_mask, trg_mask.shape)

        # encoder
        encoder_output = self.encoder(src, src_mask)
        # encoder_output: [src_len, batch_size, emb_dim]
        # print(encoder_output.shape)

        # decoder
        decoder_output = self.decoder(trg, encoder_output, trg_mask, src_mask)
        # decoder_output: [trg_len, batch_size, output_dim]
        # print(decoder_output.shape)

        return decoder_output


# parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# weight initialization with Xavier uniform
def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

#  inference
def translate_sentence(sentence, src_field, trg_field, model, device, max_len):
    # 1. set the model to eval
    model.eval()

    # 2. check whether sentence is str or not then tokenize
    if isinstance(sentence, str):
        # tokens = tokenize_src(sentence)
        tokens = [tok for tok in tokenize_src(sentence)]
        # logging.info(f'Tokenized: {tokens}')
    else:
        tokens = [tok for tok in sentence]
        # logging.info(f'Tokenized: {tokens}')

    # 3. add <sos> and <eos> tokens
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    # 4. numericalize the sentence
    src_indexes = [src_field.vocab.stoi[tok] for tok in tokens]

    # 5. create tensor
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    # logging.info(src_tensor.shape)

    # 6. create src mask
    src_mask = model.make_src_mask(src_tensor)

    # 7. encoder
    with torch.no_grad():
        encoder_output = model.encoder(src_tensor, src_mask)

    # 8. create target tensor
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    # 9. run decoder to predict next token
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)
        trg_mask = model.generate_square_subsequent_mask(trg_tensor.shape[0])

        with torch.no_grad():
            output = model.decoder(trg_tensor, encoder_output, trg_mask, src_mask)

        # print(output.shape)
        # print(output.argmax(2))
        pred_token = output.argmax(2)[-1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # print(trg_indexes)
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    # trg_tokens = spm_model.decode(trg_tokens[:-1])
    # print(f"Translation: {' '.join(tok for tok in trg_tokens[1:-1])}")
    # print(f'Translation: {trg_tokens}')
    return trg_tokens[1:]


# calculation of BLEU score
def calculate_scores(data, src_field, trg_field, model, device, max_len):
    trgs = []
    srcs = []
    pred_trgs = []
    i = 0
    logging.info("Calculating BLEU\n")

    for item in data:
        src = vars(item)["src"]
        trg = vars(item)["trg"]
        i += 1

        pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len)
        pred_trg = pred_trg[:-1]

        if i % 1000 == 0 or i == len(data):
            logging.info(f"Processed {i}/{len(data)} sentences")

        # for tokenized bleu
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        srcs.append(src)
        
    pred_sentences=[" ".join(text) for text in pred_trgs]
    trg_sentences=[" ".join(text[0]) for text in trgs]
    src_sentences=[" ".join(text) for text in srcs]

    scores=imported_calculate_scores(pred_sentences,trg_sentences,src_sentences)
    scores['tokenized_bleu']=bleu_score(pred_trgs, trgs)*100
    return scores,pred_trgs

# timeit
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def output_generate(sentence,actual_output,model):
    input=' '.join(tok for tok in sentence)
    generated=' '.join(translate_sentence(sentence, SRC, TRG, model, device, MAX_LEN))
    reference=' '.join(tok for tok in actual_output)
    return input,generated,reference

# main:
if __name__ == "__main__":
    # arg parser
    arg_parser = argparse.ArgumentParser("Transformer NMT")
    arg_parser.add_argument(
        "--data", help="Path to data folder", required=True, type=str, default=None
    )
    arg_parser.add_argument(
        "--train", help="Train file name", required=True, type=str, default=None
    )
    arg_parser.add_argument(
        "--valid", help="Valid file name", required=True, type=str, default=None
    )
    arg_parser.add_argument(
        "--test", help="Test file name", required=True, type=str, default=None
    )
    arg_parser.add_argument(
        "--save", help="Path to save model", required=True, type=str, default=None
    )
    arg_parser.add_argument(
        "--predict_dir", help="Path to save results", required=True, type=str, default=None
    )
	
    arg_parser.add_argument(
        "--batch_size", help="Batch Size", required=False, type=int, default=16
    )
    arg_parser.add_argument(
        "--emb_dim", help="Embedding Size", required=False, type=int, default=512
    )
    arg_parser.add_argument(
        "--ff_dim", help="FF Size", required=False, type=int, default=2048
    )
    arg_parser.add_argument(
        "--nheads", help="No. of attention heads", required=False, type=int, default=8
    )
    arg_parser.add_argument(
        "--nlayers",
        help="No. of Encoder and Decoder layers",
        required=False,
        type=int,
        default=6,
    )
    arg_parser.add_argument(
        "--dropout", help="Dropout", required=False, type=float, default=0.1
    )
    arg_parser.add_argument(
        "--epoch", help="No. of epochs", required=False, type=int, default=2
    )
    arg_parser.add_argument(
        "--max_len",
        help="Maximum length of sentence",
        required=False,
        type=int,
        default=250,
    )
    arg_parser.add_argument(
        "--test_only", help="Testing only", required=False, type=str, default="no"
    )
    arg_parser.add_argument(
        "--continue_train",
        help="Continue Training",
        required=False,
        type=str,
        default="no",
    )

    args = arg_parser.parse_args()

    # loading data
    logging.info("Loading data\n")
    train_set, valid_set, test_set = TabularDataset.splits(
        path=args.data,
        train=args.train,
        validation=args.valid,
        test=args.test,
        format="tsv",
        fields=[("src", SRC), ("trg", TRG)],
    )
    
    # building vocab
    logging.info("Building vocab\n")
    SRC.build_vocab(train_set, min_freq=1)
    TRG.build_vocab(train_set, min_freq=1)

    # batching
    logging.info("Batching\n")
    BATCH_SIZE = args.batch_size
    train_iter, valid_iter, test_iter = BucketIterator.splits(
        (train_set, valid_set, test_set),
        batch_size=BATCH_SIZE,
        sort=False,
        device=device,
    )

    # statistics
    logging.info(f"SRC vocab size: {len(SRC.vocab)}\n")
    logging.info(f"TRG vocab size: {len(TRG.vocab)}\n")
    logging.info(
        f"Data sizes:\nTrain:{len(train_set):,}\nDev:{len(valid_set):,}\nTest:{len(test_set):,}\n"
    )
    logging.info(
        f"Iterator sizes:\ntrain_iter: {len(train_iter):,}\nvalid_iter: {len(valid_iter):,}\ntest_iter: {len(test_iter):,}\n"
    )

    # model parameters
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    EMB_DIM = args.emb_dim
    MAX_LEN = args.max_len
    FF_DIM = args.ff_dim
    N_HEADS = args.nheads
    N_LAYERS = args.nlayers
    DROPOUT = args.dropout
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model_log_save_path = 'Logs/'+args.save + '.transformer.log'
    
    
    f_log = open(model_log_save_path, 'a')

    f_log.write(f'SRC vocab size: {len(SRC.vocab)}\n')
    f_log.write(f'TRG vocab size: {len(TRG.vocab)}\n')
    f_log.write(f'Data sizes:\nTrain:{len(train_set)}\nDev:{len(valid_set)}\nTest:{len(test_set)}\n')
    f_log.write(f'train_iter: {len(train_iter)}\nvalid_iter: {len(valid_iter)}\ntest_iter: {len(test_iter)}\n\n')

    # initializing encoder, decoder and seq2seq
    encoder = Transformer_Encoder(
        INPUT_DIM, EMB_DIM, MAX_LEN, FF_DIM, N_HEADS, N_LAYERS, DROPOUT, device
    )
    decoder = Transformer_Decoder(
        OUTPUT_DIM, EMB_DIM, MAX_LEN, FF_DIM, N_HEADS, N_LAYERS, DROPOUT, device
    )
    model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, device).to(device)

    # parameter count
    logging.info(f"The model has {count_parameters(model):,} trainable parameters\n")
    f_log.write(f'The model has {count_parameters(model):,} trainable parameters\n\n')
    f_log.write(f'{model}')

    # applying weights
    model.apply(initialize_weights)
    # logging.info(model)

    # optimizer
    LR = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # running
    N_EPOCHS = args.epoch
    CLIP = 1

    best_valid_loss = float("inf")
    best_tokenized_bleu = float("-inf")

    #have to move file
    model_save_path = "Model/"+args.save + ".transformer.best.pt"
    print('Model path:', model_save_path)

    #loading train and evaluate function
    train_evaluate_module = importlib.import_module(args.save)
    train= getattr(train_evaluate_module,"train")
    evaluate=getattr(train_evaluate_module,"evaluate")

    #loading score function 
    module_name = 'scores'
    module_path = '/scores'
    spec = importlib.util.spec_from_file_location(module_name, f"{module_path}/{module_name}.py")
    score_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_module)
    imported_calculate_scores=score_module.imported_calculate_scores

    # continuing the training
    if args.continue_train.lower() == "yes":
        model.load_state_dict(torch.load(model_save_path))
        logging.info(f"Loaded best model parameters from: {model_save_path}")
        logging.info(f"Continuing training")

    if args.test_only.lower() == "no":
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss = train(
                model, train_iter, optimizer, criterion, CLIP, epoch + 1, N_EPOCHS,TRG.vocab.stoi[TRG.eos_token],
                SRC,TRG,embed_model
            )
            valid_loss = evaluate(
                model, valid_iter, criterion,TRG.vocab.stoi[TRG.eos_token],
                SRC,TRG,embed_model
            )



            # if valid_loss < best_valid_loss:
            #     best_valid_loss = valid_loss
            #     torch.save(model.state_dict(), model_save_path)


            score_dictionary_valid,_ = calculate_scores(valid_set, SRC, TRG, model, device, MAX_LEN)
            score_dictionary_test,_ = calculate_scores(test_set, SRC, TRG, model, device, MAX_LEN)

            if score_dictionary_valid["tokenized_bleu"] > best_tokenized_bleu:
                best_tokenized_bleu = score_dictionary_valid["tokenized_bleu"]
                torch.save(model.state_dict(), model_save_path)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            logging.info(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            logging.info(
                f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
            )
            logging.info(
                f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
            )

            for key,val in score_dictionary_valid.items():
                logging.info(f'\t Val. {key}: {val :.5f}')
            for key,val in score_dictionary_test.items():
                logging.info(f'\t Test. {key}: {val :.5f}')

            f_log.write(f'\n Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            f_log.write(f'\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            f_log.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            for key,val in score_dictionary_valid.items():
                f_log.write(f'\t Val. {key}: {val :.5f}')
            for key,val in score_dictionary_test.items():
                f_log.write(f'\t Test. {key}: {val :.5f}')

            sentences = [vars(iter)["src"] for iter in test_set.examples ]
            actual_outputs = [vars(iter)["trg"] for iter in test_set.examples ]

            output_path = 'Output_csv/'+args.save + '.csv'

            with open(output_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["context", "gold", "prediction"])
                for sentence,actual_output in zip(sentences,actual_outputs):
                    input,generated,reference=output_generate(sentence,actual_output,model)
                    writer.writerow([input,reference,generated])

    # testing
    os.path.join(args.save, model_save_path)
    model.load_state_dict(torch.load(model_save_path))
    print('model loaded')
    logging.info(f"\nLoaded best model parameters from: {model_save_path}\n")
    test_loss = evaluate(model, test_iter, criterion,TRG.vocab.stoi[TRG.eos_token],SRC,TRG,embed_model)
    logging.info("Testing:\n")
    logging.info(
        f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |\n"
    )
    # f_log.write(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |\n')

    sentence = vars(test_set.examples[0])["src"]
    actual_output = vars(test_set.examples[0])["trg"]
    # sentence = ' '.join(tok for tok in sentence)
    logging.info(f"Input: {' '.join(tok for tok in sentence)}\n")
    logging.info(
        f"Generated: {' '.join(translate_sentence(sentence, SRC, TRG, model, device, MAX_LEN))}\n"
    )
    logging.info(f"Reference: {' '.join(tok for tok in actual_output)}\n")


    score_dictionary_test,preds=calculate_scores(test_set, SRC, TRG, model, device, MAX_LEN)
    for key,val in score_dictionary_test.items():
        logging.info(f'\t Test. {key}: {val :.5f}')
        f_log.write(f'\t Test. {key}: {val :.5f}\n')

    model_pred_save_path = 'Pred_tok/'+ args.save + "." + args.test + ".pred.tok"
    os.path.join(args.save, model_pred_save_path)
    f_preds = open(model_pred_save_path, "w")

    for item in preds:
        sent = " ".join(tok for tok in item)
        f_preds.write(f"{sent}\n")

    result_path = 'Results/'+args.save + '.txt'

    with open(result_path, 'w') as file:
        for key,val in score_dictionary_test.items():
            file.write(f'\t Test. {key}: {val :.5fd}\n')
   
