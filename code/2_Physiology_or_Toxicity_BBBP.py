# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# helper functions and global settings

FORCE_CPU = True       # use CPU instead of GPU
RECREATE_SETS = True    # use existing sets for training, validation and testing
TRAIN_MODELS = False    # conduct training
SAVE_MODELS = False     # save model at the end of epoch
RERUN = 3               # number of traning runs
MAIN_FOLDER = '../saved_models'
line_length = 60

def pretty_print_divider(n=1, lb_n=0, char="#"):
    if isinstance(n, bool):
        n = 1 if n else 0
    if lb_n > 0:
        print("\n" * lb_n, end="")
    elif n > 1:
        print()
    for _ in range(n):
        print(char * line_length)

def pretty_print(message, pb=False, pa=False, lb_n=0, char="#"):
    pretty_print_divider(pb, lb_n=lb_n, char=char)
    available_space = line_length - 7
    formatted_message = f"{char * 3} {message}"
    padding = line_length - len(formatted_message) - 4
    if padding < 0:
        formatted_message = formatted_message[:line_length-7] + "..."
        padding = 0
    print(formatted_message + " " * padding + f" {char * 3}")
    pretty_print_divider(pa, char=char)
# +
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

torch.manual_seed(8) # for reproduction

# set device dynamically (either GPU or CPU)
if torch.cuda.is_available() and not FORCE_CPU:
    device_count = torch.cuda.device_count()
    if device_count > 1:
        pretty_print(f"multiple GPUs detected ({device_count})", pb=True)
        best_device_id = -1
        max_free_mem = 0
        for i in range(device_count):
            free, _ = torch.cuda.mem_get_info(i)
            pretty_print(f"GPU {i} free memory: {free}")
            if free > max_free_mem:
                max_free_mem = free
                best_device_id = i
        device = torch.device(f'cuda:{best_device_id}')
        out_device = f"GPU: cuda:{best_device_id}"
    else:
        device = torch.device('cuda')
        out_device = "GPU: cuda:0"
else:
    device = torch.device('cpu')
    out_device = "CPU"
torch.backends.cudnn.benchmark = True
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
torch.nn.Module.dump_patches = True

pretty_print(f"Using device: {out_device}", pb=True, pa=True)


# +
import time
import numpy as np
import gc
import sys
sys.setrecursionlimit(50000)

import pickle
import json
import csv
import shutil
import copy
import pandas as pd

# import AttentiveFP own modules
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight, featurize_smiles_from_dict
# -


from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score


# +
from rdkit import Chem
from rdkit.Chem import QED
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import mplfinance as mpf
from IPython.display import SVG, display
import seaborn as sns; sns.set(color_codes=True)

os.makedirs(f'{MAIN_FOLDER}', exist_ok=True)

# +
task_name = 'BBBP'
tasks = ['BBBP']
raw_filename = "../data/B3DB.csv"
feature_filename = raw_filename.replace('.csv','.pickle')
filename = raw_filename.replace('.csv','')
prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.smiles.values
pretty_print(f"number of all smiles: {len(smilesList)}")
atom_num_dist = []
remained_smiles = []
canonical_smiles_list = []

for smiles in smilesList:
    try:        
        mol = Chem.MolFromSmiles(smiles)
        atom_num_dist.append(len(mol.GetAtoms()))
        remained_smiles.append(smiles)
        canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    except:
        pretty_print(f"not successfully processed smiles: {smiles}")
        pass
        
pretty_print(f"number of successfully processed smiles: {len(remained_smiles)}")
smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
smiles_tasks_df['cano_smiles'] =canonical_smiles_list
assert canonical_smiles_list[8]==Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][8]), isomericSmiles=True)


# +
random_seed = 188
random_seed = int(time.time())
start_time = str(time.ctime()).replace(':','-').replace(' ','_')
start = time.time()

batch_size = 100
epochs = 800
p_dropout = 0.1
fingerprint_dim = 150

radius = 3
T = 2
weight_decay = 2.9 # also known as l2_regularization_lambda
learning_rate = 3.5
per_task_output_units_num = 2 # for classification model with 2 classes
output_units_num = len(tasks) * per_task_output_units_num


# +
smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())<101]
uncovered = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())>100]

smiles_tasks_df = smiles_tasks_df[~smiles_tasks_df["cano_smiles"].isin(uncovered)] # removing compounds with more than 100 atoms

if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb" ))
else:
    feature_dicts = save_smiles_dicts(smilesList,filename)

remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)
uncovered_df


# +
pretty_print(f"Creating weights", lb_n=2, pb=True, pa=True)
weights = []
for i,task in enumerate(tasks):    
    negative_df = remained_df[remained_df[task] == 0][["smiles",task]]
    positive_df = remained_df[remained_df[task] == 1][["smiles",task]]
    weights.append([(positive_df.shape[0]+negative_df.shape[0])/negative_df.shape[0],\
                    (positive_df.shape[0]+negative_df.shape[0])/positive_df.shape[0]])
    
def create_sets(n=0):
    pretty_print(f"Creating sets for run {n}", lb_n=2, pb=True)
    folder = f'{MAIN_FOLDER}/run_{n}'
    os.makedirs(f'{folder}', exist_ok=True)
    os.makedirs(f'{folder}/sets', exist_ok=True)
        
    if not RECREATE_SETS and os.path.isfile(f'{folder}/sets/valid_df.csv'):
        valid_df = pd.read_csv(f"{folder}/sets/valid_df.csv")
        train_df = pd.read_csv(f"{folder}/sets/train_df.csv")
        test_df = pd.read_csv(f"{folder}/sets/test_df.csv")
    else:    
        test_df = remained_df.sample(frac=1/10, random_state=random_seed) # test set
        training_data = remained_df.drop(test_df.index) # training data

        # training data is further divided into validation set and train set
        valid_df = training_data.sample(frac=1/9, random_state=random_seed) # validation set
        train_df = training_data.drop(valid_df.index) # train set
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        valid_df.to_csv(f'{folder}/sets/valid_df.csv')
        train_df.to_csv(f'{folder}/sets/train_df.csv')
        test_df.to_csv(f'{folder}/sets/test_df.csv')
    return valid_df, train_df, test_df


# +
def create_model():
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smilesList[0]],feature_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]

    loss_function = [nn.CrossEntropyLoss(torch.tensor(weight), reduction='mean') for weight in weights]
    model = Fingerprint(radius, T, num_atom_features, num_bond_features, fingerprint_dim, output_units_num, p_dropout)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    pretty_print(f"Number of parameters: {params}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            pretty_print(f"{name} {param.data.shape}")
    
    return model, optimizer, loss_function



# +
def train(model, dataset, optimizer, loss_function):
    model.train()
    np.random.seed(epoch)
    valList = np.arange(0,dataset.shape[0])
    np.random.shuffle(valList)
    batch_list = []
    
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)
        
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch,:]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.tensor(x_atom),
                                                 torch.tensor(x_bonds),
                                                 torch.tensor(x_atom_index, dtype=torch.long),
                                                 torch.tensor(x_bond_index, dtype=torch.long),
                                                 torch.tensor(x_mask))

        model.zero_grad()
        
        # compute your loss function (torch wants the target wrapped in a variable)
        loss = 0.0
        for i,task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                    per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]

            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.tensor(validInds, dtype=torch.long).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)

            loss += loss_function[i](
                y_pred_adjust,
                torch.tensor(y_val_adjust, dtype=torch.long))
            
        # do backward pass and update gradient
        loss.backward()
        optimizer.step()
        
def eval(model, dataset):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)  
        
    for counter, test_batch in enumerate(batch_list):
        batch_df = dataset.loc[test_batch,:]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.tensor(x_atom),
                                                 torch.tensor(x_bonds),
                                                 torch.tensor(x_atom_index, dtype=torch.long),
                                                 torch.tensor(x_bond_index, dtype=torch.long),
                                                 torch.tensor(x_mask))
        
        atom_pred = atoms_prediction.data[:,:,1].unsqueeze(2).cpu().numpy()
        
        for i,task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) * per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]

            if len(validInds) == 0:
                continue
                
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.tensor(validInds, dtype=torch.long).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
            loss = loss_function[i](y_pred_adjust, torch.tensor(y_val_adjust, dtype=torch.long))
            y_pred_adjust = F.softmax(y_pred_adjust,dim=-1).data.cpu().numpy()[:,1]
            losses_list.append(loss.cpu().detach().numpy())
            
            try:
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
            except:
                y_val_list[i] = []
                y_pred_list[i] = []
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
           
    test_roc = [roc_auc_score(y_val_list[i], y_pred_list[i]) for i in range(len(tasks))]
    test_prc = [auc(precision_recall_curve(y_val_list[i], y_pred_list[i])[1],precision_recall_curve(y_val_list[i], y_pred_list[i])[0]) for i in range(len(tasks))]
    test_precision = [precision_score(y_val_list[i], (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    test_recall = [recall_score(y_val_list[i], (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    test_loss = np.array(losses_list).mean()

    return test_roc, test_prc, test_precision, test_recall, test_loss


# -

if TRAIN_MODELS:
    shutil.rmtree(f'{MAIN_FOLDER}')
    os.makedirs(f'{MAIN_FOLDER}', exist_ok=True)
    for n in range(RERUN):
        valid_df, train_df, test_df = create_sets(n)
        folder = f'{MAIN_FOLDER}/run_{n}'
        folder_models = f'{folder}/models'
        folder_stats = f'{folder}/stats'
        os.makedirs(folder_models, exist_ok=True)
        os.makedirs(folder_stats, exist_ok=True)

        model, optimizer, loss_function = create_model()
        
        best_param ={}
        best_param["roc_epoch"] = 0
        best_param["loss_epoch"] = 0
        best_param["valid_roc"] = 0
        best_param["valid_loss"] = 9e8

        epoch_meta = {}

        for epoch in range(epochs):    
            train_roc, train_prc, train_precision, train_recall, train_loss = eval(model, train_df)
            valid_roc, valid_prc, valid_precision, valid_recall, valid_loss = eval(model, valid_df)
            train_roc_mean = np.array(train_roc).mean()
            valid_roc_mean = np.array(valid_roc).mean()
            epoch_meta[epoch] = {
                'train_roc': train_roc,
                'train_prc': train_prc,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_loss': train_loss,
                'valid_roc': valid_roc,
                'valid_prc': valid_prc,
                'valid_precision': valid_precision,
                'valid_recall': valid_recall,
                'valid_loss': valid_loss
            }

            if valid_roc_mean > best_param["valid_roc"]:
                best_param["roc_epoch"] = epoch
                best_param["valid_roc"] = valid_roc_mean
                
                if valid_roc_mean > 0.87 and SAVE_MODELS:
                    name = f'{folder_models}/model_'+prefix_filename+'_'+start_time+'_'+str(epoch)+'.pt'
                    torch.save(model, name)  
                
                # save best model seperately and save epoch metadata
                pd.DataFrame.from_dict(epoch_meta).transpose().to_csv(f'{folder_stats}/epoch_metadata.csv')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_roc': train_roc,
                    'train_prc': train_prc,
                    'train_precision': train_precision,
                    'train_recall': train_recall,
                    'train_loss': train_loss,
                    'valid_roc': valid_roc,
                    'valid_prc': valid_prc,
                    'valid_precision': valid_precision,
                    'valid_recall': valid_recall,
                    'valid_loss': valid_loss
                }, f'{folder}/best_model.pth')
                torch.save(model, f'{folder}/best_model.pt')  

            if valid_loss < best_param["valid_loss"]:
                best_param["loss_epoch"] = epoch
                best_param["valid_loss"] = valid_loss

            pretty_print(f"RUN: {"0" if n < 10 else ""}{n} EPOCH: {epoch}", pb=3 if epoch == 0 else False)
            pretty_print(f"train_roc: {train_roc}")
            pretty_print(f"valid_roc: {valid_roc}", pa=True)

            # early stopping
            if (epoch - best_param["roc_epoch"] > 18) and (epoch - best_param["loss_epoch"] > 28):
                break

            train(model, train_df, optimizer, loss_function)

        pd.DataFrame.from_dict(epoch_meta).transpose().to_csv(f'{folder_stats}/epoch_metadata.csv')

        # evaluate model
        best_model_eval = torch.load(f'{folder}/best_model.pt', weights_only=False)
        best_model = torch.load(f'{folder}/best_model.pth', weights_only=False)
        model.load_state_dict(best_model['model_state_dict'])
        epoch = best_model['epoch']

        test_roc, test_prc, test_precision, test_recall, test_losses = eval(best_model_eval, test_df)

        meta_best = {
            'epoch': best_model['epoch'],
            'train_roc': best_model['train_roc'],
            'train_prc': best_model['train_prc'],
            'train_precision': best_model['train_precision'],
            'train_recall': best_model['train_recall'],
            'train_loss': best_model['train_loss'],
            'valid_roc': best_model['valid_roc'],
            'valid_prc': best_model['valid_prc'],
            'valid_precision': best_model['valid_precision'],
            'valid_recall': best_model['valid_recall'],
            'valid_loss': best_model['valid_loss'],
            'test_roc': test_roc,
            'test_prc': test_prc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_losses': test_losses
        }

        pretty_print(f"best epoch: {epoch}", pb=True)
        pretty_print(f"test_roc: {test_roc}")
        pretty_print(f"test_roc_mean: {np.array(test_roc).mean()}", pa=True)

        pd.DataFrame.from_dict(meta_best).to_csv(f'{folder_stats}/best_model_metadata.csv')

# +
smiles_to_test = ['O=O',                                                                    # oxygen
                  'C(=O)=O',                                                                # carbon dioxide
                  'CCO',                                                                    # ethanol        
                  'CN1CCC[C@H]1C2=CN=CC=C2',                                                # nicotine
                  'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',                                           # caffeine
                  'CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2',                                      # diphenhydramine
                  'C1C(=O)NC2=C(C=C(C=C2)Br)C(=N1)C3=CC=CC=C3Cl',                           # phenazepam
                  'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O',                             # glucose
                  'C1=CC(=C(C=C1C[C@@H](C(=O)O)N)O)O',                                      # levodopa
                  'CN(C)C(=O)C(CCN1CCC(CC1)(C2=CC=C(C=C2)Cl)O)(C3=CC=CC=C3)C4=CC=CC=C4',    # loperamide
                  'CCOC(=O)N1CCC(=C2C3=C(CCC4=C2N=CC=C4)C=C(C=C3)Cl)CC1',                   # loratadine
                  'C[C@H](C1=CC(=C(C=C1)C2=CC=CC=C2)F)C(=O)O',                              # tarenflurbil
                  'CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C',                        # penicillin
                  'C1=CC(=C(C=C1CCN)O)O']                                                   # dopamine

folder_eval = f'{MAIN_FOLDER}/eval'
os.makedirs(f'{folder_eval}', exist_ok=True)
records = []
for smile_to_test in smiles_to_test:
    for n in range(RERUN):
        folder = f'{MAIN_FOLDER}/run_{n}'
        model_filepath = f'{folder}/best_model.pt'

        if os.path.isfile(model_filepath):
            pretty_print(f"Loading model from {model_filepath}", pb=True)
            model = torch.load(model_filepath, map_location=device, weights_only=False)
            model.eval()

            pretty_print(f"Running inference for SMILES: {smile_to_test}")

            try:
                # Featurize the SMILES string using the new function
                x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = featurize_smiles_from_dict(smile_to_test, feature_dicts)

                # Convert numpy arrays to tensors
                x_atom_tensor = torch.tensor(x_atom)
                x_bonds_tensor = torch.tensor(x_bonds)
                x_atom_index_tensor = torch.tensor(x_atom_index, dtype=torch.long)
                x_bond_index_tensor = torch.tensor(x_bond_index, dtype=torch.long)
                x_mask_tensor = torch.tensor(x_mask)
                
                # Perform prediction
                with torch.no_grad():
                    _, mol_prediction = model(x_atom_tensor, x_bonds_tensor, x_atom_index_tensor, x_bond_index_tensor, x_mask_tensor)

                # Process the output
                probabilities = F.softmax(mol_prediction, dim=1)
                prob_class_0 = probabilities[0, 0].item()
                prob_class_1 = probabilities[0, 1].item()
                predicted_class = torch.argmax(probabilities, dim=1).item()

                pretty_print(f"Raw model output (logits): {mol_prediction.cpu().numpy().flatten()}", pa=True)

                pretty_print(f"Prediction for SMILES: {smile_to_test}", pb=True)
                pretty_print(f"Probability of NOT crossing BBB (class 0): {prob_class_0:.4f}")
                pretty_print(f"Probability of crossing BBB (class 1): {prob_class_1:.4f}")
                pretty_print(f"Predicted class: {predicted_class} ({'Crosses BBB' if predicted_class == 1 else 'Does not cross BBB'})", pa=True)

                records.append({
                    'smile': smile_to_test,
                    'run': n,
                    'mol_prediction': mol_prediction.cpu().numpy().flatten(),    # raw logits
                    'probabilities': probabilities.cpu().numpy(),                # full softmax vector
                    'prob_class_0': prob_class_0,
                    'prob_class_1': prob_class_1,
                    'predicted_class': predicted_class
                })

            except ValueError as e:
                print(f"Error featurizing SMILES: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        else:
            print(f"Model file not found at: {model_filepath}")

    # assemble DataFrame with composite index
    if records:  # only create DataFrame if there are records
        df = pd.DataFrame.from_records(records).set_index(['smile', 'run'])
        df.index.names = ['smile', 'run']
        df.to_csv(f'{folder_eval}/results.csv')
        pretty_print(f"Saved {len(records)} prediction records to {folder_eval}/results.csv", pb=True, pa=True)
    else:
        pretty_print("No prediction records were created. Check if model files exist and inference completed successfully.", pb=True, pa=True)
        # create an empty DataFrame to prevent the next section from failing
        df = pd.DataFrame(columns=['smile', 'run', 'mol_prediction', 'probabilities', 'prob_class_0', 'prob_class_1', 'predicted_class'])
        df.to_csv(f'{folder_eval}/results.csv', index=False)

# +
# do typical stats on the saved results
folder_eval = f'{MAIN_FOLDER}/eval'

if os.path.exists(f"{folder_eval}/results.csv") and os.path.getsize(f"{folder_eval}/results.csv") > 0:
    df = pd.read_csv(f"{folder_eval}/results.csv", index_col=[0, 1])
    df.index.names = ['smile', 'run']
    grp = df.groupby(level='smile')

    # compute basic stats
    stats = grp.agg(
        p0_mean=('prob_class_0', 'mean'),
        p0_std=('prob_class_0', 'std'),
        p0_var=('prob_class_0', 'var'),
        p0_min=('prob_class_0', 'min'),
        p0_max=('prob_class_0', 'max'),
        p0_median=('prob_class_0', 'median'),
        p0_q1=('prob_class_0', lambda x: x.quantile(0.25)),
        p0_q3=('prob_class_0', lambda x: x.quantile(0.75)),
        p1_mean=('prob_class_1', 'mean'),
        p1_std=('prob_class_1', 'std'),
        p1_var=('prob_class_1', 'var'),
        p1_min=('prob_class_1', 'min'),
        p1_max=('prob_class_1', 'max'),
        p1_median=('prob_class_1', 'median'),
        p1_q1=('prob_class_1', lambda x: x.quantile(0.25)),
        p1_q3=('prob_class_1', lambda x: x.quantile(0.75)),
        pred_mode=('predicted_class', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
        pred_nunique=('predicted_class', 'nunique')
    )

    # save the stats DataFrame to a new CSV
    stats.to_csv(f"{folder_eval}/results_stats.csv")

    pretty_print(f"Loaded {len(df)} rows from 'results.csv'", pb=True)
    pretty_print(f"Saved aggregated stats with {len(stats)} smiles to 'results_stats.csv'", pa=True)

    # error bar plot
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(stats))
    ax.errorbar(x - 0.1, stats['p0_mean'], yerr=stats['p0_std'], fmt='o', label='Class 0')
    ax.errorbar(x + 0.1, stats['p1_mean'], yerr=stats['p1_std'], fmt='o', label='Class 1')
    ax.set_xticks(x)
    ax.set_xticklabels(stats.index, rotation=45, ha='right')
    ax.set_ylabel('Probability')
    ax.set_title('Mean ± STD of Class Probabilities per SMILE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{folder_eval}/prob_errorbars.png", dpi=150)
    plt.close(fig)
    pretty_print(f"Saved error bar plot to {folder_eval}/prob_errorbars.png", pb=True, pa=True)

    # candlestick-style plot
    # helper function to create OHLC frame for a given class
    def get_ohlc(df, prob_col):
        # The data is already indexed by 'smile' and 'run', so group by 'smile' (level=0)
        # and then for each group, calculate ohlc.
        # iloc[0] and iloc[-1] depend on the order of runs. Let's assume they are stored sequentially.
        ohlc = df.groupby(level='smile')[prob_col].agg(
            open='first',
            high='max',
            low='min',
            close='last'
        )
        return ohlc

else:
    pretty_print("No results file found or file is empty. Skipping statistics generation.", pb=True, pa=True)
# -


