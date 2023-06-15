#requirements presonality.csv only

#Dataset format creation 

import csv

#######################################################d
#define window size and train test dev ratio here 
windowsize=3
train_ratio,dev_ratio,test_ratio=0.8,0.1,0.1

#######################################################

#src_file.txt trg_file.txt creation
def format(s,symbol):
    return symbol+" "+s+" "+symbol

def format_chat(persona,history,windowsize,cur_utterance):
    
    #creating choosing window size 
    if windowsize!=-1:
        window_history="".join(history[-(2*windowsize-1):-1])
    else:
        window_history="".join(history[:-1])
    window_history=format(window_history,'<history>')
    chat_data=f"{persona}{window_history}{cur_utterance}"
    return chat_data

with open("personality.csv",'r') as file:
    reader=csv.reader(file)
    with open("src.txt",'w') as src_file,open("trg.txt",'w') as trg_file:
        for i,row in enumerate(reader):
            if i==0:
                continue
            chats=row[2].split('\n')
            persona=format(row[1],'<persona>')
            history=[]
            cur_utterance=""
            for i,chat in enumerate(chats):
                if i%2==0:
                    u1=format(chat,'<u1>')
                    cur_utterance=format(chat,'<cur_utterance>')
                    history.append(u1)
                else:
                    u2=format(chat,'<u2>')
                    cur_src=format_chat(persona,history,windowsize,cur_utterance)
                    cur_trg=chat
                    src_file.write(cur_src+'\n')
                    trg_file.write(cur_trg+'\n')
                    history.append(u2)

#code for train test dev 
#caution check window size 
import math
def f(x):
    return math.floor(x) 

src_trg_pairs=[]
with open("src.txt", "r") as src_file, open("trg.txt", "r") as trg_file:
    # Iterate over corresponding lines from both files
    for src, trg in zip(src_file,trg_file):
        src_trg_pairs.append([src,trg])

total=len(src_trg_pairs)

a1=f(total*0.8)
a2=f(total*0.9)

train_src_trg=src_trg_pairs[0:a1]
test_src_trg=src_trg_pairs[a1:a2]
dev_src_trg=src_trg_pairs[a2:total]

files=['train','test','dev']
list_dic={'train':train_src_trg,'test':test_src_trg,'dev':dev_src_trg}
for file in files:
    save_src_file=file+'_src.txt'
    save_trg_file=file+'_trg.txt'
    with open(save_src_file,'w') as src_file,open(save_trg_file,'w') as trg_file:
        for src,trg in list_dic[file]:
            src_file.write(src)
            trg_file.write(trg)


####################################################################################################

# Preprocess parallel corpus and remove sentences above specified length
import os 
files=['train','dev','test']
# paste -d'\t' train_src.txt train_trg.txt > train.tsv
for file in files:
    os.system(f"paste -d'\t' {file}_src.txt {file}_trg.txt > {file}.tsv")

import sys
from tqdm import tqdm

def prepocess(filename):
    input_file = filename
    max_len = 200
    fout_path = input_file + '.pre'
    count = 0
    with open(filename, 'r', encoding='utf-8') as fin, open(fout_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.rstrip('\n')
            src = line.split('\t')[0]
            trg = line.split('\t')[1]
            # label = line.split('\t')[2]

            len_src = len(src.split())
            len_trg = len(trg.split())

            if len_src <= max_len and len_trg<= max_len and len_src > 0 and len_trg > 0:
                # print(f'{src}\t{trg}')
                fout.write(f'{src}\t{trg}\n')
                # fout.write(f'{src}\t{trg}\t{label}\n')
            else:
                # print(f'{len_src, len_trg}')
                src = ' '.join(src.split()[:max_len])
                trg = ' '.join(trg.split()[:max_len])
                len_src=len(src.split())
                len_trg=len(trg.split())
                if len_trg>0 and len_src>0:
                    fout.write(f'{src}\t{trg}\n')
                    # fout.write(f'{src}\t{trg}\t{label}\n')
                    count += 1

    print(f'Preprocessed file written: {fout_path}')
    print(f'No. of lines truncated: {count}')

files=['train.tsv','dev.tsv','test.tsv']
for file in files:
    prepocess(file)
