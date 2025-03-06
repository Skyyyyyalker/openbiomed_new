import json
import os
from transformers import LlamaTokenizer
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

llama_tokenizer_path = "/data_storage/niezk/model/llama/llama-2-7b-chat"

def calculate(outputs, labels):
    tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_path)
    output_tokens = []
    gt_tokens = []
    meteor_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    print("Calculate meteor and rouge...")    
    for i in tqdm(range(len(outputs))):
        output_tokens.append(tokenizer.tokenize(outputs[i]))
        gt_tokens.append([tokenizer.tokenize(labels[i])])
        
        meteor_scores.append(meteor_score(gt_tokens[-1], output_tokens[-1]))
        rouge_scores.append(scorer.score(outputs[i], labels[i]))
    
    print("Calculate BLEU2...")    
    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
    print("Calculate BLEU4...")    
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    res = {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]),
    }
    return res


def getTSVScore():
    score_saving_path = f"score_easy.json"     # output file
    df = pd.read_csv("output.tsv", sep='\t').fillna("")
    df_filtered = df[df['Pred_Effect'] != 'error']
    labels_function = list(df_filtered["Label_Func"])
    outputs_effect = list(df_filtered["Pred_Effect"])
    outputs_function = list(df_filtered["Pred_Func"])
    labels_effect = list(df_filtered["Label_Effect"])
    sites = list(df_filtered["Site"])
    print("total sites:", len(df))
    print("valid sites:", len(sites))
    print("error sites:", len(df)-len(sites))
    
    function_res = calculate(outputs_function, labels_function)
    effect_res = calculate(outputs_effect, labels_effect)
    print("function result:", str(function_res))
    print("effect result:", str(effect_res))
    print(f"Save result at {score_saving_path}.")
    with open(score_saving_path, 'a+') as f:
        json.dump({
            "sites": len(df),
            "label": "",
            "function result": function_res,
            "effect result": effect_res
        }, f)


getTSVScore()
