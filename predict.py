import argparse
import math
import os
import pdb
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.model.metrics as module_metric
from src.utils.config import process_config
from src.utils.ensure_dir import ensure_dir
from src.utils.factory import create_instance
from src.utils.logger import Logger
import sys

def main(config, resume):
    # sys.path.append("项目根目录")
    test_logger = Logger()

    # Choose subsets
    subsets = ['test']

    # build model architecture
    model = create_instance(config.network)(config)
    print(model)

    # Load checkpointed model
    print(f'Loading best model weights: {resume} ...')

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Prepare directory
    _, chkpoint = os.path.split(resume)
    exp_dir = config.trainer.log_dir
    prediction_dir = os.path.join(exp_dir, 'predictions-' + chkpoint.split('.')[0])
    ensure_dir(prediction_dir)

    # Loop over subsets
    subjects = {subset: None for subset in subsets}
    df_total = []
    for subset in subsets:

        # Setup data_loader instances
        dataset = create_instance(config.data_loader)(config, subset=subset)
        df_subset = dataset.df
        data_loader = DataLoader(dataset,
                                 batch_size=dataset.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)

        # Get raw predictions

        bar = tqdm(data_loader, total=len(data_loader))
        bar.set_description(f'[{subset.upper()}]')
        predictions = []
        targets = []
        current_subject = None
        with torch.no_grad():
            for i, out in enumerate(bar):
                data = out['data']
                target = out['target'].cpu().numpy()
                file_id = out['fid']
                # output = model(data.to(device)).cpu()
                output = model(data.to(device))['fused_scores'].cpu()
                probs = output.softmax(dim=1).numpy()

                for j, fid in enumerate(file_id):
                    # print(f"target[j, :] shape = {target[j, :].shape}")
                    # print(f"output[j, :, :] shape = {output[j, :, :].shape}")
                    if current_subject is None:
                        current_subject = fid

                    if current_subject != fid:
                         # Save predictions as pickles with true and predicted labels for each subject as a separate file.
                        with open(os.path.join(prediction_dir, current_subject + '.pkl'), 'wb') as handle:
                            pickle.dump({'targets': targets, 'predictions': predictions}, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        # Reset variables
                        current_subject = fid
                        targets = []
                        predictions = []

                    targets.append(target[j, :])
                    predictions.append(probs[j, :, :])

            # This just saves the last prediction.
        if current_subject is not None:
            with open(os.path.join(prediction_dir, current_subject + '.pkl'), 'wb') as handle:
                pickle.dump({'targets': targets, 'predictions': predictions}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Append subset dataframe to total
        df_total.append(df_subset)

        # Free up memory
        del dataset, data_loader

    # Free up memory
    del model

    # Concatenate subset dataframes and save total dataframe
    df_total = pd.concat(df_total)
    df_total.to_csv(os.path.join(prediction_dir, 'overview.csv'))

    # return
    print('Running predictions based off smaller windows')

    # Create a dataframe for each eval window
    df_pred = []
    evaluation_steps = [1, 3, 5, 10, 15, 30]
    epoch_seconds = 30
    confmat_subject = {fid: {eval_step: None for eval_step in evaluation_steps} for fid in df_total['FileID'].values}
    confmat_total = {eval_step: np.zeros((5, 5)) for eval_step in evaluation_steps}

    for eval_step in evaluation_steps:
        df = pd.DataFrame()
        df['FileID'] = df_total['FileID'].values
        df['Subset'] = df_total['Partition'].values
        df['Cohort'] = df_total['Cohort'].values
        df['Experiment'] = config.exp.name
        df['Window'] = f'{eval_step} s'
        df['Epoch stride'] = eval_step
        df['Effective stride'] = f'{eval_step * epoch_seconds} s'

        for idx, row in tqdm(df.iterrows(), total=len(df)):

            # Get the true and predicted stages
            fid = row.FileID
            with open(os.path.join(prediction_dir, fid + '.pkl'), 'rb') as handle:
                labels = pickle.load(handle)
            t = np.concatenate(labels['targets'], axis=0)
            p = np.concatenate(labels['predictions'], axis=1)

            y_pred = np.argmax(p, axis=0)
            y_true = t

            # 对真实标签和预测标签按相同步长采样
            p_window = y_pred[::eval_step]
            t_window = y_true[::eval_step]

            if len(t_window) != len(p_window):
                # 截断到较短的长度
                min_len = min(len(t_window), len(p_window))
                t_window = t_window[:min_len]
                p_window = p_window[:min_len]
                print(f"警告：FileID {fid} 步长{eval_step}长度不匹配，已截断到 {min_len}")

            # Extract the metrics

            acc = metrics.accuracy_score(t_window, p_window)
            bal_acc = metrics.balanced_accuracy_score(t_window, p_window)
            kappa = metrics.cohen_kappa_score(t_window, p_window)
            f1 = metrics.f1_score(t_window, p_window, average='macro',zero_division=0)
            prec = metrics.precision_score(t_window, p_window, average='macro',zero_division=0)
            recall = metrics.recall_score(t_window, p_window, average='macro',zero_division=0)
            mcc = metrics.matthews_corrcoef(t_window, p_window)

            # Assign metrics to dataframe
            df.loc[idx, 'Overall accuracy'] = acc
            df.loc[idx, 'Balanced accuracy'] = bal_acc
            df.loc[idx, 'Kappa'] = kappa
            df.loc[idx, 'F1'] = f1
            df.loc[idx, 'Precision'] = prec
            df.loc[idx, 'Recall'] = recall
            df.loc[idx, 'MCC'] = mcc

            # Get stage-specific metrics
            precision, recall, f1, support = metrics.precision_recall_fscore_support(
                t_window, p_window, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)

            # Assign to dataframe
            for stage_idx, stage in zip([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM']):
                df.loc[idx, f'F1 - {stage}'] = f1[stage_idx]
                df.loc[idx, f'Precision - {stage}'] = precision[stage_idx]
                df.loc[idx, f'Recall - {stage}'] = recall[stage_idx]
                df.loc[idx, f'Support - {stage}'] = support[stage_idx]

            # Get confusion matrix
            C = metrics.confusion_matrix(
                t_window, p_window, labels=[0, 1, 2, 3, 4])
            confmat_subject[fid][eval_step] = C
            confmat_total[eval_step] += C

        # Update list
        df_pred.append(df)

    # Finalize dataframe
    df_pred = pd.concat(df_pred)

    # Save dataframe
    # exp_dir, chkpoint = os.path.split(resume)
    df_pred.to_csv(os.path.join(prediction_dir, 'predictions.csv'))

    # Save confusion matrices to pickle
    C = {'total': confmat_total, 'subject-specific': confmat_subject}
    with open(os.path.join(prediction_dir, 'confusionmatrix.pkl'), 'wb') as handle:
        pickle.dump(C, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep-sleep-pytorch')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='Path to configuration file (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    default_config = r'E:\python\deep-sleep\experiments\shuju2_xsleep_exp\0515_180106\config.yaml'
    default_resume = r'E:\python\deep-sleep\experiments\shuju2_xsleep_exp\0515_180106\checkpoint-epoch86.pth'

    if args.config is None and args.resume is None:
        args.config = default_config
        args.resume = default_resume
    if args.device is None:
        args.device = '0'

    if args.config:
        # load config file
        config = process_config(args.config)
        # setting path to save trained models and log files
        path = os.path.join(config.trainer.save_dir, config.exp.name)

    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)['config']

    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.yaml', for example.")

    if args.resume is None:
        raise AssertionError(
            "Checkpoint file need to be specified. Add '-r checkpoint.pth', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
