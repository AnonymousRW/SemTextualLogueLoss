#new evaluation metrics
from sacrebleu.metrics import BLEU
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from numpy.linalg import norm
import numpy as np 

embed_model = SentenceTransformer('bert-base-nli-mean-tokens')

def calculate_sacre_bleu(hyps,refs):
    bleu = BLEU()

    result = bleu.corpus_score(hyps, [refs])
    temp=result.precisions
    dic_result={"Bleu":result.score,"Blue-1":temp[0],"Blue-2":temp[1],"Blue-3":temp[2],"Blue-4":temp[3]}
    return dic_result

def calculate_meteor(hyps,refs):
    refs= [item.split(' ') for item in refs]
    hyps= [item.split(' ') for item in hyps]
    total=0.0
    for cur_refs,cur_hyps in zip(refs,hyps):
        total+=meteor_score([cur_refs],cur_hyps)
    total=total/len(refs)
    return total

def cos_sim(text1,text2):
    e1=embed_model.encode(text1,show_progress_bar=False)
    e2=embed_model.encode(text2,show_progress_bar=False)
    return np.dot(e1, e2)/(norm(e1)*norm(e2))

def calculate_bert_similarity(sentences1,sentences2):
    cos_sim_list=[cos_sim(text1,text2) for text1,text2 in zip(sentences1,sentences2)]
    return sum(cos_sim_list)/len(cos_sim_list)

def calculatge_rouge(predicted_sentences,target_sentences):
    
    new_predicted_sentences=[]
    new_target_sentences=[]
    for prd,trg in zip(predicted_sentences,target_sentences):
        if len(prd)!=0 and len(trg)!=0:
            new_predicted_sentences.append(prd)
            new_target_sentences.append(trg)
    
    rouge=Rouge()
    result=rouge.get_scores(new_predicted_sentences,new_target_sentences,avg=True)
    dic_result={}
    list=["rouge-1","rouge-2","rouge-l"]
    for key in list:
        dic_result[key]=result[key]['f']*100
    return dic_result

def imported_calculate_scores(predicted_sentences,target_sentences,source_sentences):
    results={}
    bleu_score=calculate_sacre_bleu(predicted_sentences,target_sentences)
    meteor_score=calculate_meteor(predicted_sentences,target_sentences)
    rouge_score=calculatge_rouge(predicted_sentences,target_sentences)
    bert_score=calculate_bert_similarity(predicted_sentences,target_sentences)
    bert_context_score= calculate_bert_similarity(predicted_sentences,source_sentences)
    results.update(bleu_score)
    results["meteor"]=meteor_score*100
    results.update(rouge_score)
    results["bert_similarity"]=bert_score*100
    results["bert_context_similarity"]=bert_context_score*100
    return results