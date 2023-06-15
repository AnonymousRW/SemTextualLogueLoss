# Preprocess parallel corpus and remove sentences above specified length

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