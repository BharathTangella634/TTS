import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import argparse
import json
import os
import gc
import gspread
import utils.preprocessing as pp
import utils.data_helper as dh
from transformers import AdamW
from utils import modeling, evaluation, model_utils
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,classification_report, confusion_matrix

from torch.utils.tensorboard import SummaryWriter   
from pytorchtools import EarlyStopping

# from sklearn import metrics
# import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# def compute_stepwise_results(step_number, epoch, tweets, targets, stances, predicted_stances, metrics, dataset_type, output_file):
#     """Save step-wise results to a CSV file."""
#     results = []
#     for i in range(len(tweets)):
#         results.append([
#             dataset_type,  # Dataset Type
#             metrics['generation'],
#             step_number,  # Step
#             metrics['seed'],
#             metrics['dropout'],
#             metrics['dropout_rest'],
#             metrics['against_precision'],
#             metrics['against_recall'],
#             metrics['against_f1'],
#             metrics['favor_precision'],
#             metrics['favor_recall'],
#             metrics['favor_f1'],
#             metrics['neutral_precision'],
#             metrics['neutral_recall'],
#             metrics['neutral_f1'],
#             metrics['overall_precision'],
#             metrics['overall_recall'],
#             metrics['overall_f1'],
#             step_number,
#             tweets[i],
#             targets[i],
#             stances[i],
#             predicted_stances[i],
#             epoch
#         ])

#     column_names = [
#         'Dataset Type', 'Generation', 'Step', 'Seed', 'Dropout', 'Dropout Rest',
#         'Against Precision', 'Against Recall', 'Against F1',
#         'Favor Precision', 'Favor Recall', 'Favor F1',
#         'Neutral Precision', 'Neutral Recall', 'Neutral F1',
#         'Overall Precision', 'Overall Recall', 'Overall F1',
#         'Step Number', 'Tweet', 'Target', 'Stance', 'Predicted Stance', 'Epoch'
#     ]

#     results_df = pd.DataFrame(results, columns=column_names)
#     results_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

# def compute_performance(preds, y, tweets, targets, trainvaltest, step, args, seed, epoch, output_file):
#     """Compute performance and save step-wise details."""
#     preds_np = preds.cpu().numpy()
#     preds_np = np.argmax(preds_np, axis=1)
#     y_np = y.cpu().numpy()

#     # Compute metrics
#     results_two_class = precision_recall_fscore_support(y_np, preds_np, average=None)
#     results_weighted = precision_recall_fscore_support(y_np, preds_np, average='macro')

#     metrics = {
#         'generation': args['gen'],
#         'seed': seed,
#         'dropout': args['dropout'],
#         'dropout_rest': args['dropoutrest'],
#         'against_precision': results_two_class[0][0],
#         'against_recall': results_two_class[1][0],
#         'against_f1': results_two_class[2][0],
#         'favor_precision': results_two_class[0][1],
#         'favor_recall': results_two_class[1][1],
#         'favor_f1': results_two_class[2][1],
#         'neutral_precision': results_two_class[0][2],
#         'neutral_recall': results_two_class[1][2],
#         'neutral_f1': results_two_class[2][2],
#         'overall_precision': results_weighted[0],
#         'overall_recall': results_weighted[1],
#         'overall_f1': results_weighted[2],
#     }

#     # Save step-wise details
#     predicted_stances = ["AGAINST" if x == 0 else "FAVOR" if x == 1 else "NONE" for x in preds_np]
#     actual_stances = ["AGAINST" if x == 0 else "FAVOR" if x == 1 else "NONE" for x in y_np]
#     compute_stepwise_results(step, epoch, tweets, targets, actual_stances, predicted_stances, metrics, trainvaltest, output_file)

#     return results_weighted[2]


# def compute_stepwise_results(step_number, epoch, stances, predicted_stances, metrics, dataset_type):
#     """Save step-wise results to a CSV file."""
#     results = []
#     for i in range(len(stances)):
#         results.append([
#             dataset_type,  # Dataset Type
#             metrics['generation'],
#             step_number,  # Step
#             metrics['seed'],
#             metrics['dropout'],
#             metrics['dropout_rest'],
#             metrics['against_precision'],
#             metrics['against_recall'],
#             metrics['against_f1'],
#             metrics['favor_precision'],
#             metrics['favor_recall'],
#             metrics['favor_f1'],
#             metrics['neutral_precision'],
#             metrics['neutral_recall'],
#             metrics['neutral_f1'],
#             metrics['overall_precision'],
#             metrics['overall_recall'],
#             metrics['overall_f1'],
#             step_number,
#             stances[i],
#             predicted_stances[i],
#             epoch
#         ])

#     column_names = [
#         'Dataset Type', 'Generation', 'Step', 'Seed', 'Dropout', 'Dropout Rest',
#         'Against Precision', 'Against Recall', 'Against F1',
#         'Favor Precision', 'Favor Recall', 'Favor F1',
#         'Neutral Precision', 'Neutral Recall', 'Neutral F1',
#         'Overall Precision', 'Overall Recall', 'Overall F1',
#         'Step Number', 'Stance', 'Predicted Stance', 'Epoch'
#     ]

#     results_df = pd.DataFrame(results, columns=column_names)
#     # results_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
#     results_df.to_csv('./results_'+dataset_type+'_df.csv',index=False, mode='a', header=True) 

# def compute_performance(preds, y, trainvaltest, step, args, seed, epoch):
#     """Compute performance and save step-wise details."""
#     preds_np = preds.cpu().numpy()
#     preds_np = np.argmax(preds_np, axis=1)
#     y_np = y.cpu().numpy()

#     print("predicted stance: ", preds_np[:10])
#     print("Actual stance : ",y_np[:50])

#     stances_df = pd.DataFrame({
#       'predicted stance': preds_np,
#       'original stance': y_np
#     })

#     path = f"C:\\Users\\CSE RGUKT\\Downloads\\TTS\\step_wise_results\\{trainvaltest}_result_{step}step_{epoch}epoch.csv"

#     stances_df.rename(columns={'predicted stance': 'Predicted stance', 'original stance': 'Original stance'}, inplace=True)
#     stances_df.to_csv(path, index=False, columns=['Predicted stance', 'Original stance'])

#     # stances_df.to_csv(path, index=False, columns = ['Predicted stance', 'Original stance'])

#     # Compute metrics
#     results_two_class = precision_recall_fscore_support(y_np, preds_np, average=None)
#     results_weighted = precision_recall_fscore_support(y_np, preds_np, average='macro')

#     metrics = {
#         'generation': args['gen'],
#         'seed': seed,
#         'dropout': args['dropout'],
#         'dropout_rest': args['dropoutrest'],
#         'against_precision': results_two_class[0][0],
#         'against_recall': results_two_class[1][0],
#         'against_f1': results_two_class[2][0],
#         'favor_precision': results_two_class[0][1],
#         'favor_recall': results_two_class[1][1],
#         'favor_f1': results_two_class[2][1],
#         'neutral_precision': results_two_class[0][2],
#         'neutral_recall': results_two_class[1][2],
#         'neutral_f1': results_two_class[2][2],
#         'overall_precision': results_weighted[0],
#         'overall_recall': results_weighted[1],
#         'overall_f1': results_weighted[2],
#     }
#     result_overall = [results_weighted[0],results_weighted[1],results_weighted[2]]
#     result_against = [results_two_class[0][0],results_two_class[1][0],results_two_class[2][0]]
#     result_favor = [results_two_class[0][1],results_two_class[1][1],results_two_class[2][1]]
#     result_neutral = [results_two_class[0][2],results_two_class[1][2],results_two_class[2][2]]

#     result_id = ['train', args['gen'], step, seed, args['dropout'],args['dropoutrest']]
#     result_one_sample = result_id + result_against + result_favor + result_neutral + result_overall
#     result_one_sample = [result_one_sample]

#     # Save step-wise details
#     predicted_stances = ["AGAINST" if x == 0 else "FAVOR" if x == 1 else "NONE" for x in preds_np]
#     actual_stances = ["AGAINST" if x == 0 else "FAVOR" if x == 1 else "NONE" for x in y_np]
#     # output_file = f"C:\\Users\\CSE RGUKT\\Downloads\\TTS\\analysis_results_stepwise\\{trainvaltest}_result_{step}step_{epoch}epoch.csv"
#     compute_stepwise_results(step, epoch, actual_stances, predicted_stances, metrics, trainvaltest)
#     print(results_weighted[2])
#     return results_weighted[2], result_one_sample

# def compute_stepwise_results(step_number, epoch, stances, predicted_stances, dataset_type, output_file):
#     """
#     Save predicted stances step-wise with a single Actual column and
#     dynamically add new Predicted columns for every step and epoch.
#     """
#     column_name = f"Predicted_step_{step_number}_epoch_{epoch}"  # New column name

#     # Check if the file exists
#     if os.path.exists(output_file):
#         results_df = pd.read_csv(output_file)  # Read the existing file
#     else:
#         # Create a new DataFrame with 'Actual' column
#         results_df = pd.DataFrame({'Actual': stances})

#     # Ensure 'Actual' column exists; overwrite if needed
#     if 'Actual' not in results_df.columns:
#         results_df['Actual'] = stances

#     # Add the predicted column for the current step and epoch
#     results_df[column_name] = predicted_stances

#     # Save the updated DataFrame back to the file
#     results_df.to_csv(output_file, index=False)

import os
import pandas as pd

def compute_stepwise_results(step_number, epoch, stances, predicted_stances, dataset_type, output_file):
    """
    Save predicted stances step-wise with a single Actual column and
    dynamically add new Predicted columns for every step and epoch.
    """
    column_name = f"Predicted_step_{step_number}_epoch_{epoch}"  # New column name

    # Check if the file exists
    if os.path.exists(output_file):
        results_df = pd.read_csv(output_file)  # Read the existing file
    else:
        # Create a new DataFrame with 'Actual' column
        results_df = pd.DataFrame({'Actual': stances})

    # Ensure 'Actual' column exists; overwrite if needed
    if 'Actual' not in results_df.columns:
        results_df['Actual'] = stances

    # Add the predicted column for the current step and epoch
    results_df[column_name] = predicted_stances

    # Save the updated DataFrame back to the file
    results_df.to_csv(output_file, index=False)


def compute_performance(preds, y, trainvaltest, step, args, seed, epoch, output_analysis_file, output_results_file):
    """
    Compute performance metrics and save analysis + step-wise details.
    Returns overall F1 score and a single-row analysis list (result_one_sample).
    """
    # Convert predictions and targets to numpy arrays
    preds_np = preds.cpu().numpy()
    preds_np = np.argmax(preds_np, axis=1)
    y_np = y.cpu().numpy()

    # Map stances to labels
    predicted_stances = ["AGAINST" if x == 0 else "FAVOR" if x == 1 else "NONE" for x in preds_np]
    actual_stances = ["AGAINST" if x == 0 else "FAVOR" if x == 1 else "NONE" for x in y_np]

    # Save step-wise predictions and actual values
    compute_stepwise_results(step, epoch, actual_stances, predicted_stances, trainvaltest, output_results_file)

    # Compute precision, recall, f1 metrics
    results_two_class = precision_recall_fscore_support(y_np, preds_np, average=None)
    results_weighted = precision_recall_fscore_support(y_np, preds_np, average='macro')

    # Organize metrics
    result_overall = [results_weighted[0], results_weighted[1], results_weighted[2]]
    result_against = [results_two_class[0][0], results_two_class[1][0], results_two_class[2][0]]
    result_favor = [results_two_class[0][1], results_two_class[1][1], results_two_class[2][1]]
    result_neutral = [results_two_class[0][2], results_two_class[1][2], results_two_class[2][2]]

    result_id = [trainvaltest, args['gen'], step, seed, args['dropout'], args['dropoutrest']]
    result_one_sample = result_id + result_against + result_favor + result_neutral + result_overall

    # Metrics to save in a DataFrame
    metrics = {
        'Dataset Type': trainvaltest,
        'Step': step,
        'Epoch': epoch,
        'Seed': seed,
        'Generation': args['gen'],
        'Dropout': args['dropout'],
        'Dropout Rest': args['dropoutrest'],
        'Against Precision': results_two_class[0][0],
        'Against Recall': results_two_class[1][0],
        'Against F1': results_two_class[2][0],
        'Favor Precision': results_two_class[0][1],
        'Favor Recall': results_two_class[1][1],
        'Favor F1': results_two_class[2][1],
        'Neutral Precision': results_two_class[0][2],
        'Neutral Recall': results_two_class[1][2],
        'Neutral F1': results_two_class[2][2],
        'Overall Precision': results_weighted[0],
        'Overall Recall': results_weighted[1],
        'Overall F1': results_weighted[2],
    }

    # Append metrics to the analysis file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_analysis_file, mode='a', index=False, header=not os.path.exists(output_analysis_file))

    return results_weighted[2], result_one_sample  # Return overall F1 and single analysis row


def run_classifier():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-g', '--gen', help='Generation number of student model', required=False)
    parser.add_argument('-s', '--seed', help='Random seed', required=False)
    parser.add_argument('-d', '--dropout', help='Dropout rate', required=False)
    parser.add_argument('-d2', '--dropoutrest', help='Dropout rate for rest generations', required=False)
    parser.add_argument('-train', '--train_data', help='Name of the training data file', required=False)
    parser.add_argument('-dev', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-test', '--test_data', help='Name of the test data file', default=None, required=False)
    parser.add_argument('-kg', '--kg_data', help='Name of the kg test data file', default=None, required=False)
    parser.add_argument('-clipgrad', '--clipgradient', type=str, default='True', help='whether clip gradient when over 2', required=False)
    parser.add_argument('-step', '--savestep', type=int, default=1, help='whether clip gradient when over 2', required=False)
    parser.add_argument('-p', '--percent', type=int, default=1, help='whether clip gradient when over 2', required=False)
    parser.add_argument('-es_step', '--earlystopping_step', type=int, default=1, help='whether clip gradient when over 2', required=False)

    args = vars(parser.parse_args())



    # writer = SummaryWriter('./tensorboard/')

    sheet_num = 4  # Google sheet number
    num_labels = 3  # Favor, Against and None
#     random_seeds = [0,1,2,3,4,42]
    random_seeds = []
    random_seeds.append(int(args['seed']))
    
    # create normalization dictionary for preprocessing
    with open("C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    norm_dict = {**data1,**data2}
    
    # load config file
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    model_select = config['model_select']
    
    # Use GPU or not
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    best_result, best_against, best_favor, best_val, best_val_against, best_val_favor,  = [], [], [], [], [], []
    for seed in random_seeds:    
        print("current random seed: ", seed)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'train')
        train_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'val')
        val_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'test')
        test_writer = SummaryWriter(log_dir=log_dir)

        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
        
        x_train, y_train, x_train_target = pp.clean_all(args['train_data'], norm_dict)
        x_val, y_val, x_val_target = pp.clean_all(args['dev_data'], norm_dict)
        x_test, y_test, x_test_target = pp.clean_all(args['test_data'], norm_dict)
        x_test_kg, y_test_kg, x_test_target_kg = pp.clean_all(args['kg_data'], norm_dict)
        stance_mapping = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}

        # y_train = [stance_mapping[label] for label in y_train if label in stance_mapping]
        # y_val = [stance_mapping[label] for label in y_val if label in stance_mapping]
        # y_test = [stance_mapping[label] for label in y_test if label in stance_mapping]



        x_train_all = [x_train,y_train,x_train_target]
        x_val_all = [x_val,y_val,x_val_target]
        x_test_all = [x_test,y_test,x_test_target]
        x_test_kg_all = [x_test_kg,y_test_kg,x_test_target_kg]
        # print(x_train_all[0][0])
        # print(x_test_kg_all[0][-1])
        if int(args['gen']) >= 1:
            print("Current generation is: ", args['gen'])
            x_train_all = [a+b for a,b in zip(x_train_all, x_test_kg_all)]
        print(x_test_all[0][0], x_test_all[1][0], x_test_all[2][0])
        
        # print('-------')
        # print(x_train_all[0][-1])
        # raise Exception

        # prepare for model
        loader, gt_label = dh.data_helper_bert(x_train_all, x_val_all, x_test_all, x_test_kg_all, model_select, config)
        trainloader, valloader, testloader, trainloader2, kg_testloader = loader[0], loader[1], loader[2], loader[3], loader[4]
        y_train, y_val, y_test, y_train2 = gt_label[0], gt_label[1], gt_label[2], gt_label[3]
        y_val, y_test, y_train2 = y_val.to(device), y_test.to(device), y_train2.to(device)
        
        # train setup
        model, optimizer = model_utils.model_setup(num_labels, model_select, device, config, int(args['gen']), float(args['dropout']),float(args['dropoutrest']))
        loss_function = nn.CrossEntropyLoss()
        sum_loss = []
        val_f1_average, val_f1_against, val_f1_favor = [], [], []
        test_f1_average, test_f1_against, test_f1_favor, test_kg = [], [], [], []

        # early stopping

        # print(model)

        # for name, layer in model.named_modules():
        #     print(f"Layer name: {name}, Layer type: {layer}")
        # raise Exception

        es_intermediate_step = len(trainloader)//args['savestep']
        patience = args['earlystopping_step']   # the number of iterations that loss does not further decrease
        # patience = es_intermediate_step   # the number of iterations that loss does not further decrease        
        early_stopping = EarlyStopping(patience, verbose=True)
        print(100*"#")
        # print("len(trainloader):",len(trainloader))
        # print("args['savestep']:",args['savestep'])
        print("early stopping occurs when the loss does not decrease after {} steps.".format(patience))
        print(100*"#")
        # print(bk)
        # init best val/test results
        best_train_f1macro = 0
        best_train_result = []
        best_val_f1macro = 0
        best_val_result = []
        best_test_f1macro = 0
        best_test_result = []

        best_val_loss = 100000
        best_val_loss_result = []
        best_test_loss = 100000
        best_test_loss_result = []
        # start training
        print(100*"#")
        print("clipgradient:",args['clipgradient']=='True')
        print(100*"#")

        # model.eval()
        # with torch.no_grad():
        #     preds, loss_train = model_utils.model_preds(trainloader, model, device, loss_function)
        #     train_writer.add_scalar('loss', sum(loss_train) / len(loss_train), 0)
        #     preds, loss_val = model_utils.model_preds(valloader, model, device, loss_function)
        #     val_writer.add_scalar('loss', sum(loss_val) / len(loss_val), 0)
        #     preds, loss_test = model_utils.model_preds(testloader, model, device, loss_function)
        #     test_writer.add_scalar('loss', sum(loss_test) / len(loss_test), 0)
        step = 0

        # start training
        for epoch in range(0, int(config['total_epochs'])):
            print('Epoch:', epoch)
            train_loss = []  
            model.train()
            for b_id, sample_batch in enumerate(trainloader):
                model.train()
                optimizer.zero_grad()
                dict_batch = model_utils.batch_fn(sample_batch)

                inputs = {k: v.to(device) for k, v in dict_batch.items()}
                

                outputs = model(**inputs)
                loss = loss_function(outputs, inputs['gt_label'])


                # outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['gt_label'])
                # loss = outputs.loss
                # outputs = outputs.logits
                

                # accumulation_steps = 2  # Simulate batch size 32
                # optimizer.zero_grad()
                # for step in range(accumulation_steps):
                #     outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['gt_label'])
                #     loss = outputs.loss
                #     outputs = outputs.logits


                loss.backward()

                # nn.utils.clip_grad_norm_(model.parameters(), 2)

                if args['clipgradient']=='True':
                    nn.utils.clip_grad_norm_(model.parameters(), 2)

                optimizer.step()
                step+=1
                train_loss.append(loss.item())

                # print("len(trainloader):",len(trainloader))
                split_step = len(trainloader)//args['savestep']
                # print("savestep:",savestep)
                # print(bk)
                # if step%args['savestep']==0:





                # Printing confusion matrix at step

                # preds_train_class = torch.argmax(outputs, dim=1)  # Predictions after applying argmax to logits
                # y_train2_1 = inputs['gt_label']  # Ground truth labels

                # # print(f"Step: {step}")
                # # print(f"Shape of preds_train_class: {preds_train_class.shape}")
                # # print(f"Shape of y_train2_1: {y_train2_1.shape}")

                # Check if batch sizes match
                # assert preds_train_class.shape[0] == y_train2_1.shape[0], f"Shape mismatch: {preds_train_class.shape[0]} vs {y_train2_1.shape[0]}"

                # # Compute confusion matrix
                # preds_train_class = preds_train_class.cpu()
                # y_train2_1 = y_train2_1.cpu()

                # confusion_matrix_ = confusion_matrix(y_train2_1, preds_train_class)

                # Adjust the display_labels according to the number of classes
                # print(confusion_matrix_)





                if step%split_step==0:


                    # for name, param in model.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"{name}: {param.grad.norm()}")

                    print(outputs)
                    
                    model.eval()
                    with torch.no_grad():
                        preds_train, loss_train_inval_mode = model_utils.model_preds(trainloader2, model, device, loss_function)
                        preds_val, loss_val = model_utils.model_preds(valloader, model, device, loss_function)
                        preds_test, loss_test = model_utils.model_preds(testloader, model, device, loss_function)
                        print(100*"#")
                        print("at step: {}".format(step))
                        print("train_loss",train_loss,len(train_loss), sum(train_loss)/len(train_loss))
                        print("loss_val",loss_val,len(loss_val), sum(loss_val) / len(loss_val))
                        print("loss_test",loss_test,len(loss_test), sum(loss_test) / len(loss_test))

                        # print(bk)

                        train_writer.add_scalar('loss', sum(train_loss)/len(train_loss), step)
                        val_writer.add_scalar('loss', sum(loss_val) / len(loss_val), step)
                        test_writer.add_scalar('loss', sum(loss_test) / len(loss_test), step)


                        f1macro_train, result_one_sample_train = compute_performance(
                            preds_train, y_train2,
                            trainvaltest='train',  # Use 'test' or 'val' accordingly
                            step=step,
                            args={'gen': args['gen'], 'dropout': args['dropout'], 'dropoutrest': args['dropoutrest']},
                            seed=args['seed'],
                            epoch=epoch,
                            output_analysis_file='./analysis_results_train.csv',  # Analysis results
                            output_results_file='./results_train.csv'  # Stepwise results
                            )
                        f1macro_val, result_one_sample_val = compute_performance(
                            preds_val, y_val,
                            trainvaltest='validation',  # Use 'test' or 'val' accordingly
                            step=step,
                            args={'gen': args['gen'], 'dropout': args['dropout'], 'dropoutrest': args['dropoutrest']},
                            seed=args['seed'],
                            epoch=epoch,
                            output_analysis_file='./analysis_results_validation.csv',  # Analysis results
                            output_results_file='./results_validation.csv'  # Stepwise results
                        )
                        f1macro_test, result_one_sample_test = compute_performance(
                            preds_test, y_test,
                            trainvaltest='test',  # Use 'test' or 'val' accordingly
                            step=step,
                            args={'gen': args['gen'], 'dropout': args['dropout'], 'dropoutrest': args['dropoutrest']},
                            seed=args['seed'],
                            epoch=epoch,
                            output_analysis_file='./analysis_results_test.csv',  # Analysis results
                            output_results_file='./results_test.csv'  # Stepwise results
                        )
                        # f1macro_test, result_one_sample_test = compute_performance(preds_test,y_test,'test',step, args, seed, epoch)
                        print(preds_test)
                        print(y_train2)
                        train_writer.add_scalar('f1macro', f1macro_train, step)
                        val_writer.add_scalar('f1macro', f1macro_val, step)
                        test_writer.add_scalar('f1macro', f1macro_test, step)


                        if f1macro_val>best_val_f1macro:
                            best_val_f1macro = f1macro_val
                            best_val_result = result_one_sample_val
                            print(100*"#")
                            print("best f1-macro validation updated at epoch :{}, to: {}".format(epoch, best_val_f1macro))
                            best_test_f1macro = f1macro_test
                            best_test_result = result_one_sample_test
                            print("best f1-macro test updated at epoch :{}, to: {}".format(epoch, best_test_f1macro))
                            print(100*"#")

                        avg_val_loss = sum(loss_val) / len(loss_val)
                        avg_test_loss = sum(loss_test) / len(loss_test)
                        if avg_val_loss<best_val_loss:
                            best_val_loss = avg_val_loss
                            # best_val_loss_result = result_one_sample_val
                            print(100*"#")
                            print("best loss validation updated at epoch :{}, to: {}".format(epoch, best_val_loss))
                            best_test_loss = avg_test_loss
                            
                            print("best loss test updated at epoch :{}, to: {}".format(epoch, best_test_loss))
                            print(100*"#")


                        _, f1_average, f1_against, f1_favor = evaluation.compute_f1(preds_val, y_val)
                        val_f1_against.append(f1_against)
                        val_f1_favor.append(f1_favor)
                        val_f1_average.append(f1_average)
                        _, f1_average, f1_against, f1_favor = evaluation.compute_f1(preds_test, y_test)
                        test_f1_against.append(f1_against)
                        test_f1_favor.append(f1_favor)
                        test_f1_average.append(f1_average)

                        # kg eval
                        preds, loss_kg = model_utils.model_preds(kg_testloader, model, device, loss_function)
                        rounded_preds = F.softmax(preds, dim=1)
                        _, indices = torch.max(rounded_preds, dim=1)
                        y_preds_kg = np.array(indices.cpu().numpy())
                        test_kg.append(y_preds_kg)

                        # early stopping
                        print("loss_val:",loss_val,"average is: ",sum(loss_val) / len(loss_val))
                        early_stopping(sum(loss_val) / len(loss_val), model)
                        if early_stopping.early_stop:
                            print(100*"!")
                            print("Early stopping occurs at step: {}, stop training.".format(step))
                            print(100*"!")
                            break
                    model.train()

            if early_stopping.early_stop:
                print(100*"!")
                print("Early stopping, training ends")
                print(100*"!")
                break

            sum_loss.append(sum(train_loss)/len(train_loss))
            print(sum_loss[epoch])

        column_names = [
            'Dataset Type', 'Generation', 'Step', 'Seed', 'Dropout', 'Dropout Rest',
            'Against Precision', 'Against Recall', 'Against F1',
            'Favor Precision', 'Favor Recall', 'Favor F1',
            'Neutral Precision', 'Neutral Recall', 'Neutral F1',
            'Overall Precision', 'Overall Recall', 'Overall F1'
        ]
        best_val_results_df = pd.DataFrame([best_val_result], columns=column_names)

        # Save the DataFrame to CSV (append mode, include header only if the file is new)
        output_file = 'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\epochs_results\\updated_results\\best_results_validation_df.csv'
        best_val_results_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))    
        print('./best_results_validation_df.csv save, done!')
        ###

        # best_val_loss_result[0][0]='best validation' 
        # results_df = pd.DataFrame(best_val_loss_result)    
        # print("results_df are:",results_df.head())
        # results_df.to_csv('C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\epochs_results\\100_Gen_1_Seed_0\\best_loss_results_validation_df.csv',index=False, mode='a', header=True)    
        # print('./best_loss_results_validation_df.csv save, done!')
        #########################################################

        best_test_results_df = pd.DataFrame([best_test_result], columns=column_names)

        # Save the DataFrame to CSV (append mode, include header only if the file is new)
        output_file = 'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\epochs_results\\updated_results\\best_results_test_df.csv'
        best_test_results_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))    
        print('./best_results_test_df.csv save, done!')

        # best_test_result[0][0]='best test'
        # results_df = pd.DataFrame(best_test_result)    
        # print("results_df are:",results_df.head())
        # results_df.to_csv('C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\epochs_results\\100_Gen_1_Seed_0\\results_test_df.csv',index=False, mode='a', header=True)    
        # print('./results_test_df.csv save, done!')
        # ###
        # results_df = pd.DataFrame(best_test_result)    
        # print("results_df are:",results_df.head())
        # results_df.to_csv('C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\epochs_results\\100_Gen_1_Seed_0\\best_results_test_df.csv',index=False, mode='a', header=True)    
        # print('./best_results_test_df.csv save, done!')
        ###
        # best_test_loss_result[0][0]='best test'
        # results_df = pd.DataFrame(best_test_loss_result)    
        # print("results_df are:",results_df.head())
        # results_df.to_csv('C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\epochs_results\\100_Gen_1_Seed_0\\best_loss_results_test_df.csv',index=False, mode='a', header=True)
        # print('./best_loss_results_test_df.csv save, done!')
        #########################################################
        # model that performs best on the dev set is evaluated on the test set

        print(" SMB "* 10)
        print("Best Val Loss :", best_val_loss)
        print("Best Test Loss :", best_test_loss)
        print(" SMB "* 10)

        best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
        best_against.append(test_f1_against[best_epoch])
        best_favor.append(test_f1_favor[best_epoch])
        best_result.append(test_f1_average[best_epoch])

        print("******************************************")
        print("dev results with seed {} on all epochs".format(seed))
        print(val_f1_average)
        best_val_against.append(val_f1_against[best_epoch])
        best_val_favor.append(val_f1_favor[best_epoch])
        best_val.append(val_f1_average[best_epoch])
        print("******************************************")
        print("test results with seed {} on all epochs".format(seed))
        print(test_f1_average)
        print("******************************************")
        print(max(best_result))
        print(best_result)
        
        # update the unlabeled kg file
        concat_text = pd.DataFrame()
        raw_text = pd.read_csv(args['kg_data'],usecols=[0], encoding='ISO-8859-1')
        raw_target = pd.read_csv(args['kg_data'],usecols=[1], encoding='ISO-8859-1')
        seen = pd.read_csv(args['kg_data'],usecols=[3], encoding='ISO-8859-1')
        concat_text = pd.concat([raw_text, raw_target, seen], axis=1)
        concat_text['Stance 1'] = test_kg[best_epoch].tolist()
        concat_text['Stance 1'].replace([0,1,2], ['AGAINST','FAVOR','NONE'], inplace = True)
        concat_text = concat_text.reindex(columns=['Tweet','Target 1','Stance 1','seen?'])
        # concat_text.to_csv("/home/yli300/EMNLP2022/data/raw_train_all_subset_kg_epoch_onecol.csv", index=False)
        print(100*"#")
        concat_text.to_csv(args['kg_data'], index=False)
        print(args['kg_data'],"save, done!")
        print(100*"#")

    # save to Google sheet
    save_result = []
    save_result.append(best_against)
    save_result.append(best_favor)
    save_result.append(best_result)  # results on test set
    save_result.append(best_val_against)
    save_result.append(best_val_favor)
    save_result.append(best_val)  # results on val set
    print(save_result)

    save_result = list(map(list, zip(*save_result))) 

    column_names = [
    'Best Against', 'Best Favor', 'Best Result', 
    'Best Val Against', 'Best Val Favor', 'Best Validation'
    ]
    results_df = pd.DataFrame(save_result, columns=column_names)

    # Path to save the CSV
    output_file = 'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\epochs_results\\save_result_summary.csv'

    # Save the DataFrame to CSV (overwrite or append logic)
    results_df.to_csv(output_file, index=False, mode='w', header=True)

    print(f"Save results successfully stored at: {output_file}")
    # gc = gspread.service_account(filename='../../service_account_google.json')
    # sh = gc.open("Stance_Aug").get_worksheet(sheet_num) 
    # row_num = len(sh.get_all_values())+1
#         sh.update('A{0}'.format(row_num), target_word_pair[target_index])
    # sh.update('B{0}:O{1}'.format(row_num,row_num+30), save_result)

if __name__ == "__main__":
    run_classifier() 
