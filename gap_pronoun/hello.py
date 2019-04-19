from flask import Flask,render_template, request,json
import os
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from helperbot import BaseBot, TriangularLR
from Utils import *

app = Flask(__name__)

def predict(loader, model):
    model.eval()
    outputs = []
    with torch.set_grad_enabled(False):
        for *input_tensors, y_local in (loader):
            input_tensors = [x.to("cuda:0") for x in input_tensors]
            tmp = model(*input_tensors)
            outputs.append(tmp)
        outputs = torch.cat(outputs, dim=0)
    return outputs

BERT_MODEL = 'bert-large-uncased'
CASED = False

tokenizer = BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=CASED,
    never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]"))
tokenizer.vocab["[A]"] = -1
tokenizer.vocab["[B]"] = -1
tokenizer.vocab["[P]"] = -1

model = GAPModel(BERT_MODEL, torch.device("cuda:0"))
model.load_state_dict(torch.load("./cache/model_cache/best.pth"))
model.eval()

# df_inference = pd.read_csv("inference.tsv", delimiter="\t")
# inference_ds = GAPDataset(df_inference, tokenizer)
# inference_loader = DataLoader(
#     inference_ds,
#     collate_fn = collate_examples,
#     batch_size=128,
#     num_workers=2,
#     pin_memory=True,
#     shuffle=False)

# preds = predict(inference_loader, model)
# _, labels = torch.max(preds, 1)
# print("************************************************************************")
# print(labels.item())

@app.route('/')
def hello_world():
    return render_template('input.html')

@app.route('/get_input', methods=['GET'])
def get_input():
    df_test = pd.read_csv("gap-development.tsv", delimiter="\t")
    df_elements = df_test.sample(n=1)
    df_dict = df_elements.to_dict()
    ind = df_elements.index.values[0]
    return json.dumps({'status':'OK',
                       'content': df_dict['Text'][ind],
                       "target_a":df_dict['A'][ind],
                       "target_b":df_dict['B'][ind],
                       "pronoun":df_dict['Pronoun'][ind],
                       "target_a_pos":df_dict['A-offset'][ind],
                       "target_b_pos":df_dict['B-offset'][ind],
                       "pronoun_pos":df_dict['Pronoun-offset'][ind]})

@app.route('/predict', methods=['POST'])
def predict_model():
    
    ID = "Inference_Sample"
    URL = "https://www.google.com"
    content      =  request.form['content']
    target_a =  request.form['target_a']
    target_b =  request.form['target_b']
    pronoun  =  request.form['pronoun']
    target_a_pos =  request.form['target_a_pos']
    target_b_pos =  request.form['target_b_pos']
    pronoun_pos  =  request.form['pronoun_pos']
    line = "\n"+ID+"\t"+content+"\t"+pronoun+"\t"+pronoun_pos+"\t"+target_a+"\t"+target_a_pos+"\t"
    line += "FALSE\t"+target_b+"\t"+target_b_pos+"\tFALSE\t"+URL
    with open("inference.tsv", "w") as f: 
        f.write("ID\tText\tPronoun\tPronoun-offset\tA\tA-offset\tA-coref\tB\tB-offset\tB-coref\tURL")
        f.write(line)
    df_inference = pd.read_csv("inference.tsv", delimiter="\t")
    inference_ds = GAPDataset(df_inference, tokenizer)
    inference_loader = DataLoader(
        inference_ds,
        collate_fn = collate_examples,
        batch_size=128,
        num_workers=2,
        pin_memory=True,
        shuffle=False)   
    preds = predict(inference_loader, model)
    _, labels =  torch.max(preds, 1)
    print("======================================================================")
    print(labels.item())
    return json.dumps({'status':'OK','value':labels.item()})


if __name__ == "__main__":
    #init()
    app.run(threaded=True, host='0.0.0.0')
