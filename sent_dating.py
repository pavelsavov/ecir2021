import nltk
import numpy as np
import os
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

min_year = 1994
max_year = 2020

max_sent_length = 64
batch_size = 32
model_base = "/Users/ja/Documents/PycharmProjects/ecir/model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(yr):
    model_path = os.path.join(model_base, str(yr))
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        output_attentions=True,
        output_hidden_states=True,
    )
    model.to(device)
    return model, tokenizer


def sent_date_proba(sentences):
    proba = []
    for y in range(min_year, max_year):
        model, tokenizer = load_model(y)
        model.eval()

        input_ids = []
        attention_masks = []
        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=max_sent_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True,
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        dataset = TensorDataset(torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0))
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

        n_batch = 0
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)
                logits = outputs.logits

            logits = logits.detach().cpu().numpy()

            # Initialize while processing first year
            if y == min_year:
                proba.append(np.ones((max_year - min_year + 1, len(logits))))

            pred = np.transpose(softmax(logits, axis=1)[:, :2])
            for iy in range(max_year - min_year + 1):
                proba[n_batch][iy] *= pred[0] if iy + min_year <= y else pred[1]

            n_batch += 1

    return np.transpose(proba)


def date_document(doc):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(doc)
    proba = sent_date_proba(sentences)

    pred_year = np.mean([np.argmax(proba[i]) for i in range(len(proba))]) + min_year
    sent_years = [np.sum([pr[j] * (j + min_year) for j in range(len(pr))]) / np.sum(pr) for pr in proba]

    return pred_year, [(sentences[i], sent_years[i]) for i in range(len(sentences))]
