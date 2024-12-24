
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import utils.preprocessing as pp
import utils.data_helper as dh
from utils import modeling, evaluation, model_utils
import json
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertweetTokenizer, BartTokenizer
import torch.nn as nn
import os
from transformers import BartForSequenceClassification

# Paths
# model_path = "C:\\Users\\CSE RGUKT\\Downloads\\TTS\\checkpoint.pt"

model_path = "C:\\Users\\CSE RGUKT\\Downloads\\TTS\\checkpoint.pt"

test_data_path = "C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\data\\raw_test_all_onecol.csv"

noslang_dict_path = "C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\noslang_data.json"

emnlp_dict_path = "C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\emnlp_dict.txt"



# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load normalization dictionaries
with open(noslang_dict_path, "r") as f:
    data1 = json.load(f)
with open(emnlp_dict_path, "r") as f:
    data2 = {line.split("\t")[0]: line.split("\t")[1].strip() for line in f.readlines()}
norm_dict = {**data1, **data2}

# Load tokenizer
model_select = "Bart"
Dataset_type = "vast"
output_path = f"analysis_{model_select}_from_{Dataset_type}_test.csv"

if model_select == "Bertweet":
    tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

elif model_select == "Bart":
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli", normalization=True, clean_up_tokenization_spaces=True)
    # model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli", num_labels = 3).to(device)
    model = modeling.bart_classifier(3, model_select, 1, 0.1, 0.7).to(device)


elif model_select == "Bert":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, clean_up_tokenization_spaces=True)
    model = modeling.bert_classifier(num_labels=3, gen=0, dropout=0.1, dropoutrest=0.7).to(device)

# Preprocess data

x, y, x_target = pp.clean_all(test_data_path, norm_dict)

def convert_data_to_ids(tokenizer, target, text, label):
    concat_sent = []
    
    for tar, sent in zip(target, text):
        concat_sent.append([' '.join(map(str, sent)),' '.join(map(str, tar))])

    encoded_dict = tokenizer.batch_encode_plus(
                    concat_sent,
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 300, # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,   # Construct attn. masks.
                    truncation = True,
               )
    encoded_dict['gt_label'] = label
    return encoded_dict

encoded_dict = convert_data_to_ids(tokenizer, x_target, x, y)

def data_loader(x_all, batch_size, model_select, mode):
    """
    Create a DataLoader for the dataset.
    """
    x_input_ids = torch.tensor(x_all["input_ids"], dtype=torch.long)
    x_attention_masks = torch.tensor(x_all["attention_mask"], dtype=torch.long)
    y = torch.tensor(x_all["gt_label"], dtype=torch.long)

    if model_select == "Bert":
        x_token_type_ids = torch.tensor(x_all["token_type_ids"], dtype=torch.long)
        dataset = TensorDataset(x_input_ids, x_attention_masks, x_token_type_ids, y)
    else:
        dataset = TensorDataset(x_input_ids, x_attention_masks, y)

    shuffle = mode == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), y

# Create DataLoader
loader, y = data_loader(encoded_dict, batch_size=64, model_select=model_select, mode="test")

# Load model


model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

model.eval()

# Performance computation
loss_function = nn.CrossEntropyLoss()

def model_preds(loader, model, device, loss_function):
    preds = []
    valtest_loss = []
    
    # Inference without gradients to save memory
    with torch.no_grad():
        for b_id, sample_batch in enumerate(loader):
            dict_batch = batch_fn(sample_batch)
            inputs = {k: v.to(device) for k, v in dict_batch.items()}

            # Forward pass
            outputs = model(**inputs)
            preds.append(outputs)
            loss = loss_function(outputs, inputs['gt_label'])

            # outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['gt_label'])
            # loss = outputs.loss
            # preds.append(outputs.logits)

            valtest_loss.append(loss.item())

    return torch.cat(preds, 0), valtest_loss

def batch_fn(sample_batch):
    
    dict_batch = {}
    dict_batch['input_ids'] = sample_batch[0]
    dict_batch['attention_mask'] = sample_batch[1]
    dict_batch['gt_label'] = sample_batch[-1]
    if len(sample_batch) > 3:
        dict_batch['token_type_ids'] = sample_batch[-2]
    
    return dict_batch

preds, loss = model_preds(loader, model, device, loss_function)

def compute_performance(preds, y, trainvaltest, args, seed, epoch, output_analysis_file):
    """
    Compute performance metrics and save analysis + step-wise details.
    """
    preds_np = preds.cpu().numpy()
    preds_np = np.argmax(preds_np, axis=1)
    y_np = y.cpu().numpy()

    results_two_class = precision_recall_fscore_support(y_np, preds_np, average=None)
    results_weighted = precision_recall_fscore_support(y_np, preds_np, average="macro")

    metrics = {
        "Dataset Type": trainvaltest,
        "Epoch": epoch,
        "Seed": seed,
        "Generation": args["gen"],
        "Dropout": args["dropout"],
        "Dropout Rest": args["dropoutrest"],
        "Against Precision": results_two_class[0][0],
        "Against Recall": results_two_class[1][0],
        "Against F1": results_two_class[2][0],
        "Favor Precision": results_two_class[0][1],
        "Favor Recall": results_two_class[1][1],
        "Favor F1": results_two_class[2][1],
        "Neutral Precision": results_two_class[0][2],
        "Neutral Recall": results_two_class[1][2],
        "Neutral F1": results_two_class[2][2],
        "Overall Precision": results_weighted[0],
        "Overall Recall": results_weighted[1],
        "Overall F1": results_weighted[2],
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_analysis_file, mode="a", index=False, header=not os.path.exists(output_analysis_file))

    return results_weighted[2]

f1macro = compute_performance(
    preds, y,
    trainvaltest="test",
    args={"gen": "0", "dropout": 0.1, "dropoutrest": 0.7},
    seed=42,
    epoch="Last",
    output_analysis_file=output_path,
)

print(f"F1 Macro: {f1macro}")
print(f"Results saved to {output_path}")
