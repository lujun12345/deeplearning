from __future__ import absolute_import, division, print_function

import csv
import json
import logging
import os
from argparse import ArgumentParser
from glob import glob
from random import seed, shuffle
from datetime import datetime
from datetime import timedelta

import h5py
import mne
import numpy as np
import pandas as pd
import pyedflib
import scipy
from scipy import signal

from src.utils.parseXmlEdfp import parse_hypnogram
from src.utils.segmentation import segmentPSG

# Load configuration file
parser = ArgumentParser()
parser.add_argument(
    '-c', '--config-file',
    dest='config',
    type=str,
    # default='data_mros.json',
    default="data_shuju2.json",
    help='Configuration JSON file.'
)

args = parser.parse_args()
with open(os.path.join('E:/python/deep-sleep/src/configs', args.config), 'r') as f:
    config = json.load(f)

# Define the cohorts
COHORTS = config['COHORTS']
COHORT_OVERVIEW_FILE = config['COHORT_OVERVIEW_FILE']
OUTPUT_DIRECTORY = config['OUTPUT_DIRECTORY']
SUBSETS = ['train', 'eval', 'test']
FILTERS = config['FILTERS']
SEGMENTATION = config['SEGMENTATION']
PARTITIONS = config['PARTITIONS']


# Define a logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger(__name__)

# Create folder(s) if not available
if not os.path.exists(os.path.join(OUTPUT_DIRECTORY, 'csv')):
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, 'csv'))
if not os.path.exists(os.path.join(OUTPUT_DIRECTORY, 'h5')):
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, 'h5'))


# Create the filters
def createPSGfilters(config_filters):
    channels = ['eeg', 'eog', 'emg']
    sos = {key: [] for key in channels}
    fs = config_filters['fs_resampling']
    order = config_filters['order']
    fc = config_filters['fc']
    btype = config_filters['btype']

    for channel in channels:
        N = order[channel]
        Wn = [2 * f / fs for f in fc[channel]]
        sos[channel] = scipy.signal.butter(
            order[channel], Wn, btype[channel], output='sos')

    return sos


sos = createPSGfilters(FILTERS)


def write_H5(psg, hypnogram, N, series, name_cohort, subset=None):

    filename = os.path.join(OUTPUT_DIRECTORY, 'h5', series.FileID.lower() + '.h5')
    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('data', data=psg)
        dset = f.create_dataset('hypnogram', data=hypnogram)


def process_file(series_file):
    fileID = series_file['SubjectID']
    file_edf = series_file['File']
    file_hypnogram = series_file['Hypnogram']
    cohort = series_file['Cohort']
    subset = series_file['Partition']

    # We skip the file if the hypnogram and fileID do not match up
    skip_file = False
    if cohort in ["cassette","telemetry"]:
        if fileID != os.path.split(file_hypnogram)[1][:6]:
            skip_file = True

    elif cohort in ["vincents"]:
        if fileID != os.path.split(file_hypnogram)[1][:8]:
            skip_file = True

    elif cohort in ['apnea',"artifacts","kcomplexes","PLMs","REMs"]:
        if fileID != os.path.split(file_hypnogram)[1][10:-4]:
            skip_file = True

    elif cohort in ["patients","Subjects"]:
        if fileID != os.path.split(file_hypnogram)[1][14:-4]:
            skip_file = True

    elif cohort in ["haaglanden"]:
        if fileID != os.path.split(file_hypnogram)[1][:5]:
            skip_file = True

    if skip_file:
        LOG.info(
            '{: <5} | {: <5} | {: <5} | No matching hypnogram file'.format(
                cohort, subset, fileID, file_hypnogram))
        return None, None

    # Load the JSON file containing channel labels
    signal_labels_json_path = os.path.join(
        'E:\python\deep-sleep\src\configs\signal_labels', '{}.json'.format(cohort))

    with open(signal_labels_json_path, 'r') as f:
        cohort_labels = json.load(f)

    # Load hypnogram and change
    try:
        LOG.info(
            '{: <5} | {: <5} | {: <5} | Loading matching hypnogram'.format(
                cohort, subset, fileID))

        if cohort in ["apnea","patients","PLMs","Subjects","REMs","vincents"]:
            try:
                with open(file_hypnogram, 'r') as hyp_file:
                    hypnogram = hyp_file.read()
                    lines = [h.strip() for h in hypnogram.split('\n') if h.strip()]  # 清洗空行和空格
                    hypnogram = []
                    for line in lines:
                        try:
                            # 尝试转换为整数，只保留有效行
                            hypnogram.append(int(line))
                        except ValueError:
                            # 记录无效行，但不中断程序
                            LOG.warning(f"{cohort} | {subset} | {fileID} | 跳过非整数行: {line}")
                    # 检查是否有有效数据
                    if not hypnogram:
                        LOG.error(f"{cohort} | {subset} | {fileID} | 催眠图文件中无有效整数数据")
                        return None, None, None
            except Exception as e:
                LOG.error(f"{cohort} | {subset} | {fileID} | 读取催眠图失败: {str(e)}")


        elif cohort in ["haaglanden"]:
            try:
                hyp_edf = pyedflib.EdfReader(file_hypnogram)
                annotations = hyp_edf.readAnnotations()
                start_times = annotations[0]  # 每个注释的起始时间（秒）
                durations = annotations[1]  # 每个注释的时长（秒）
                descriptions = annotations[2]  # 每个注释的内容（如"Sleep stage W"）

                hyp_edf.close()  # 关闭文件，避免资源泄露

                LOG.info(f'{cohort: <5} | {subset: <5} | {fileID: <5} | Found {len(descriptions)} annotations in hypnogram')
                # 打印前5条注释，验证读取是否正确
                for i in range(min(5, len(descriptions))):
                    LOG.info(
                        f'  Annotation {i + 1}: start={start_times[i]:.0f}s, duration={durations[i]:.0f}s, desc={descriptions[i]}')

            except Exception as e:
                LOG.error(f'{cohort: <5} | {subset: <5} | {fileID: <5} | Failed to read hypnogram annotations: {str(e)}')
                return None, None, None

            epoch_duration = 30  # 每个epoch=30秒（必须与PSG信号分割的epoch时长一致！）
            # 步骤1：计算整个记录的总时长（最后一个注释的结束时间）
            total_duration = start_times[-1] + durations[-1]
            # 步骤2：计算总epoch数（总时长 ÷ 30秒/epoch，向上取整避免遗漏）
            total_epochs = int(np.ceil(total_duration / epoch_duration))
            LOG.info(
                f'{cohort: <5} | {subset: <5} | {fileID: <5} | Total duration: {total_duration:.0f}s → Total epochs: {total_epochs}')
            # 步骤3：初始化标签数组（默认-1，后续填充有效分期）
            hypnogram = np.full(total_epochs, fill_value=-1)
            # 步骤4：遍历每个注释，给对应的epoch填充标签
            for start_sec, dur_sec, desc in zip(start_times, durations, descriptions):
                # 从注释内容中提取睡眠分期（如"Sleep stage W"→"W"，"Sleep stage 1"→"1"）
                desc_lower = desc.lower()
                if "sleep stage" not in desc_lower:
                    continue  # 跳过非睡眠分期的注释（如事件标记）

                # 提取分期关键词（W/1/2/3/4）
                stage_key = desc.split(" ")[-1]  # 分割字符串，取最后一个元素（如"W"、"1"）

                # 计算该注释覆盖的epoch范围
                start_epoch = int(np.floor(start_sec / epoch_duration))  # 起始epoch（向下取整）
                end_sec = start_sec + dur_sec  # 注释结束时间
                end_epoch = int(np.floor((end_sec - 1e-6) / epoch_duration))  # 结束epoch（避免重复覆盖）

                # 确保epoch索引不超出总范围
                start_epoch = max(0, start_epoch)
                end_epoch = min(total_epochs - 1, end_epoch)

                # 给对应epoch填充原始标签（先存字符串，后续标准化）
                if start_epoch <= end_epoch:
                    # 用字典临时映射字符串到原始数值（方便后续处理）
                    stage_to_raw = {
                        "w": 1,  # W（清醒）→ 原始值1
                        "n1": 2,  # N1 → 原始值2
                        "n2": 3,  # N2 → 原始值3
                        "n3": 4,  # N3 → 原始值4
                        "r": 5    # REM → 原始值5
                    }
                    # 提取并转换分期为原始数值
                    if stage_key.lower() in stage_to_raw:
                        raw_label = stage_to_raw[stage_key.lower()]
                        hypnogram[start_epoch:end_epoch + 1] = raw_label  # 批量填充epoch标签
                        LOG.info(f'  Stage {stage_key} → raw label {raw_label}: epochs {start_epoch}~{end_epoch}')

            # --------------------------  标签标准化（统一为0~4标准编码） --------------------------
            # 标准编码（行业通用）：0=清醒(W), 1=N1, 2=N2, 3=N3, 4=REM(R)
            # 基于上面的stage_to_raw映射，原始标签→标准标签的映射：
            label_mapping = {
                1: 0,  # 原始值1（W）→ 标准0（清醒）
                2: 1,  # 原始值2（N1）→ 标准1（N1）
                3: 2,  # 原始值3（N2）→ 标准2（N2）
                4: 3,  # 原始值4（N3）→ 标准3（N3）
                5: 4,  # 原始值5（R）→ REM
            }
            # 执行标准化映射
            hypnogram_standardized = np.full_like(hypnogram, fill_value=-1)
            for raw_label, std_label in label_mapping.items():
                hypnogram_standardized[hypnogram == raw_label] = std_label

            # -------------------------- 过滤异常值（保留有效标签0~4） --------------------------
            valid_mask = (hypnogram_standardized >= 0) & (hypnogram_standardized <= 4)
            hypnogram = hypnogram_standardized[valid_mask]

            # 兜底检查：若没有有效标签，返回失败
            if len(hypnogram) == 0:
                LOG.error(f'{cohort: <5} | {subset: <5} | {fileID: <5} | No valid hypnogram labels after filtering')
                return None, None, None

            # 打印最终结果，验证是否正确
            LOG.info(
                f'{cohort: <5} | {subset: <5} | {fileID: <5} | Final hypnogram: {len(hypnogram)} epochs, unique labels: {np.unique(hypnogram)}')


        elif cohort in ["cassette","telemetry"]:
            try:
                # 用pyedflib打开EDF标签文件（与读取PSG的库一致，无需额外导入）
                hyp_edf = pyedflib.EdfReader(file_hypnogram)
                annotations = hyp_edf.readAnnotations()
                start_times = annotations[0]  # 每个注释的起始时间（秒）
                durations = annotations[1]  # 每个注释的时长（秒）
                descriptions = annotations[2]  # 每个注释的内容（如"Sleep stage W"）

                hyp_edf.close()  # 关闭文件，避免资源泄露

                LOG.info(f'{cohort: <5} | {subset: <5} | {fileID: <5} | Found {len(descriptions)} annotations in hypnogram')
            # 打印前5条注释，验证读取是否正确
                for i in range(min(5, len(descriptions))):
                    LOG.info(
                        f'  Annotation {i + 1}: start={start_times[i]:.0f}s, duration={durations[i]:.0f}s, desc={descriptions[i]}')

            except Exception as e:
                LOG.error(f'{cohort: <5} | {subset: <5} | {fileID: <5} | Failed to read hypnogram annotations: {str(e)}')
                return None, None, None

        # --------------------------  解析注释为30秒epoch的标签（核心逻辑） --------------------------
        # 假设：所有睡眠分期的最小时间单位是30秒（行业标准epoch时长）
            epoch_duration = 30  # 每个epoch=30秒（必须与PSG信号分割的epoch时长一致！）

        # 步骤1：计算整个记录的总时长（最后一个注释的结束时间）
            total_duration = start_times[-1] + durations[-1]
        # 步骤2：计算总epoch数（总时长 ÷ 30秒/epoch，向上取整避免遗漏）
            total_epochs = int(np.ceil(total_duration / epoch_duration))
            LOG.info(
                f'{cohort: <5} | {subset: <5} | {fileID: <5} | Total duration: {total_duration:.0f}s → Total epochs: {total_epochs}')

        # 步骤3：初始化标签数组（默认-1，后续填充有效分期）
            hypnogram = np.full(total_epochs, fill_value=-1)

        # 步骤4：遍历每个注释，给对应的epoch填充标签
            for start_sec, dur_sec, desc in zip(start_times, durations, descriptions):
        # 从注释内容中提取睡眠分期（如"Sleep stage W"→"W"，"Sleep stage 1"→"1"）
                desc_lower = desc.lower()
                if "sleep stage" not in desc_lower:
                    continue  # 跳过非睡眠分期的注释（如事件标记）

        # 提取分期关键词（W/1/2/3/4）
                stage_key = desc.split(" ")[-1]  # 分割字符串，取最后一个元素（如"W"、"1"）

        # 计算该注释覆盖的epoch范围
                start_epoch = int(np.floor(start_sec / epoch_duration))  # 起始epoch（向下取整）
                end_sec = start_sec + dur_sec  # 注释结束时间
                end_epoch = int(np.floor((end_sec - 1e-6) / epoch_duration))  # 结束epoch（避免重复覆盖）

        # 确保epoch索引不超出总范围
                start_epoch = max(0, start_epoch)
                end_epoch = min(total_epochs - 1, end_epoch)

        # 给对应epoch填充原始标签（先存字符串，后续标准化）
                if start_epoch <= end_epoch:
            # 用字典临时映射字符串到原始数值（方便后续处理）
                    stage_to_raw = {
                    "w": 0,  # W（清醒）→ 原始值5
                    "1": 1,  # N1 → 原始值3
                    "2": 2,  # N2 → 原始值2
                    "3": 3,  # N3 → 原始值1
                    "4": 3,  # N4 → 原始值1
                    "r": 4   # REM → 原始值4
            }
            # 提取并转换分期为原始数值
                    if stage_key.lower() in stage_to_raw:
                        raw_label = stage_to_raw[stage_key.lower()]
                        hypnogram[start_epoch:end_epoch + 1] = raw_label  # 批量填充epoch标签
                        LOG.info(f'  Stage {stage_key} → raw label {raw_label}: epochs {start_epoch}~{end_epoch}')

        # --------------------------  标签标准化（统一为0~4标准编码） --------------------------
        # 标准编码（行业通用）：0=清醒(W), 1=N1, 2=N2, 3=N3, 4=REM(R)
        # 基于上面的stage_to_raw映射，原始标签→标准标签的映射：
            label_mapping = {
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                0: 0,
                }
        # 执行标准化映射
            hypnogram_standardized = np.full_like(hypnogram, fill_value=-1)
            for raw_label, std_label in label_mapping.items():
                hypnogram_standardized[hypnogram == raw_label] = std_label

        # -------------------------- 过滤异常值（保留有效标签0~4） --------------------------
            valid_mask = (hypnogram_standardized >= 0) & (hypnogram_standardized <= 4)
            hypnogram = hypnogram_standardized[valid_mask]

        # 兜底检查：若没有有效标签，返回失败
            if len(hypnogram) == 0:
                LOG.error(f'{cohort: <5} | {subset: <5} | {fileID: <5} | No valid hypnogram labels after filtering')
                return None, None, None

        # 打印最终结果，验证是否正确
            LOG.info(
                f'{cohort: <5} | {subset: <5} | {fileID: <5} | Final hypnogram: {len(hypnogram)} epochs, unique labels: {np.unique(hypnogram)}')

        else:
            hypnogram = []

    except Exception as e:  # 捕获具体异常而非所有

        LOG.error(f"{cohort} | {subset} | {fileID} | 处理催眠图时出错: {str(e)}", exc_info=True)  # 打印堆栈信息

        return None, None, None

    hypnogram = np.asarray(hypnogram)

    # Figure out which channels to load
    use_pyed = False
    use_mne = False
    if cohort in ["apnea","patients","Subjects"]:
        edf = mne.io.read_raw_edf(file_edf)
        n_signals = len(edf.ch_names)  # 信号数量 = 通道数
        sampling_frequencies = [edf.info['sfreq']] * n_signals
        signal_labels = edf.ch_names  # 通道标签列表
        use_mne = True

    else:
        edf = pyedflib.EdfReader(file_edf)
        n_signals = edf.signals_in_file
        sampling_frequencies = edf.getSampleFrequencies()
        signal_labels = edf.getSignalLabels()
        use_pyed=True

    signal_label_idx = {category: []
                        for category in cohort_labels['categories']}
    signal_data = {category: [] for category in cohort_labels['categories']}
    rereference_data = False

    for idx, label in enumerate(signal_labels):
        for category in cohort_labels['categories']:
            if label in cohort_labels[category]:
                signal_label_idx[category].append(idx)
                if category in ['A1', 'A2', 'LChin', 'RChin']:
                    rereference_data = True
            else:
                continue

    # Load all the relevant data
    # for chn, idx in signal_label_idx.items():
    #     if isinstance(idx, list):
    #         continue
    #     else:
    #         signal_data[chn] = np.zeros((1, edf.getNSamples()[idx]))
    #         signal_data[chn][0, :] = edf.readSignal(idx)

    for category, indices in signal_label_idx.items():
        # 跳过无匹配通道的类别
        if not indices:
            LOG.warning(f"类别 {category} 未找到匹配的通道，跳过")
            continue

        # 读取该类别下的所有通道数据
        channel_data = []
        for idx in indices:
            try:
                # 读取单个通道信号
                if use_pyed:
                    n_samples = edf.getNSamples()[idx]
                    signal = edf.readSignal(idx)

                elif use_mne:
                    signal = edf.get_data(picks=idx)[0]  # [0] 去除通道维度，保留时间维度
                    n_samples = len(signal)  # 样本数 = 信号长度

                # 保持与原代码一致的形状 (1, n_samples)
                channel_data.append(signal.reshape(1, -1))

                LOG.debug(f"成功读取类别 {category} 的通道 {signal_labels[idx]}（索引 {idx}），样本数：{n_samples}")
            except Exception as e:
                LOG.error(f"读取类别 {category} 的通道索引 {idx} 失败：{str(e)}")
                continue  # 跳过读取失败的通道

        # 存储该类别所有有效通道（合并为 [n_channels, n_samples]）
        if channel_data:
            signal_data[category] = np.concatenate(channel_data, axis=0)
        else:
            LOG.warning(f"类别 {category} 无有效通道数据")

    # Resample signals
    fs = config['FILTERS']['fs_resampling']
    LOG.info('{: <5} | {: <5} | {: <5} | Resampling data'.format(
        cohort, subset, fileID))

    for category in signal_data.keys():
        # 跳过空数据（列表类型）
        if isinstance(signal_data[category], list) or len(signal_data[category]) == 0:
            LOG.debug(f"类别 {category} 无有效数据，跳过重采样")
            continue
        # 获取该类别下所有通道的原始采样频率（与通道索引对应）
        # signal_label_idx[category] 是列表，存储该类别的所有通道索引
        original_fs_list = [sampling_frequencies[idx] for idx in signal_label_idx[category]]

        # 对每个通道单独重采样（shape: [n_channels, n_samples]）
        resampled_channels = []
        for ch_idx in range(signal_data[category].shape[0]):
            original_fs = original_fs_list[ch_idx]
            # 重采样（保持轴一致，axis=1 是时间维度）
            resampled = scipy.signal.resample_poly(
                signal_data[category][ch_idx:ch_idx + 1, :],  # 保留通道维度（1, n_samples）
                fs,
                original_fs,
                axis=1
            )
            resampled_channels.append(resampled)

        # 合并重采样后的通道（shape: [n_channels, resampled_samples]）
        signal_data[category] = np.concatenate(resampled_channels, axis=0)

    psg = {'eeg': np.concatenate((signal_data["EEG_FpzCz"], signal_data["EEG_PzOz"])).astype(dtype=np.float32),
           # 'eog': np.concatenate((signal_data["EOGL"], signal_data["EOGR"])).astype(dtype=np.float32),
           'eog': signal_data["EOG"].astype(dtype=np.float32),
           'emg': signal_data['EMG'].astype(dtype=np.float32)}
    print(f"EEG 通道数: {psg['eeg'].shape[0]}")  # 应该是 2
    print(f"EOG 通道数: {psg['eog'].shape[0]}")  # 应该是 2 → 这里大概率输出 1！
    print(f"EMG 通道数: {psg['emg'].shape[0]}")  # 应该是 1

    # Perform filtering
    for chn in psg.keys():

        for k in range(psg[chn].shape[0]):
            psg[chn][k, :] = scipy.signal.sosfiltfilt(sos[chn], psg[chn][k, :])
    # Do recording standardization
    for chn in psg.keys():
        processed_channels = []
        num_chn=psg[chn].shape[0]
        for k in range(num_chn):
            signal = psg[chn][k:k+1, :].squeeze(axis=0)
            m = np.mean(signal)
            s = np.std(signal)
            s = s if s > 1e-6 else 1.0

            signal = (signal - m) / s

            processed_channels.append(signal)

        psg[chn] = np.array(processed_channels)

    # Segment the PSG data
    if cohort in ["apnea","patients","Subjects","PLMs"]:
        psg_seg = segmentPSG(SEGMENTATION, fs, psg)
    else:
        psg_seg = segmentPSG(SEGMENTATION, fs, psg)

    # Also, if the signals and hypnogram are of different length, we assume that the start time is fixed for both,
    # so we trim the end
    trim_length = np.min([len(hypnogram), psg_seg['eeg'].shape[1]])
    max_length = np.max([len(hypnogram), psg_seg['eeg'].shape[1]])
    LOG.info('{: <5} | {: <5} | {: <5} | Trim/max length: {}/{} | len(hypnogram):{} | psg: {}'.format(
        cohort, subset, fileID, trim_length, max_length, len(hypnogram),psg_seg['eeg'].shape[1]))
    hypnogram = hypnogram[:trim_length]
    psg_seg = {chn: sig[:, :trim_length, :] for chn, sig in psg_seg.items()}

    # We should remove hypnogram episodes which do not conform to standards, ie. (W, N1, N2, N3, R) -> (0, 1, 2, 3, 4)
    keep_idx = []
    if cohort in ["cassette", "telemetry", "haaglanden"]:
        keep_idx = (hypnogram >= 0) & (hypnogram <= 4)

    elif cohort in ['apnea', "PLMs", "REMs"]:
        label_mapping = {
            5: 0,  # wake→0
            4: 4,  # REM→4
            3: 1,  # S1→1
            2: 2,  # S2→2
            1: 3,  # S3→3
            0: 3  # S4→3
        }
        hypnogram_mapped = np.copy(hypnogram)
        for old_label, new_label in label_mapping.items():
            hypnogram_mapped[hypnogram == old_label] = new_label
        hypnogram = hypnogram_mapped
        keep_idx = (hypnogram >= 0) & (hypnogram <= 4)

    elif cohort in ["Subjects","patients"]:
        label_mapping = {
            5: 0,  # wake→0
            4: 4,  # REM→4
            3: 1,  # N1→1
            2: 2,  # N2→2
            1: 3,  # N3→3
        }
        hypnogram_mapped = np.copy(hypnogram)
        for old_label, new_label in label_mapping.items():
            hypnogram_mapped[hypnogram == old_label] = new_label
        hypnogram = hypnogram_mapped
        keep_idx = (hypnogram >= 0) & (hypnogram <= 4)

    elif cohort in ["vincents"]:
        label_mapping = {
            0: 0,  # wake 不变
            1: 4,  # REM → 4
            2: 1,  # S1 → 1
            3: 2,  # S2 → 2
            4: 3,  # S3 → 3
            5: 3  # S4 → 3
        }
        hypno_mapped = hypnogram.copy()
        for old, new in label_mapping.items():
            hypno_mapped[hypnogram == old] = new
        hypnogram = hypno_mapped
        keep_idx = (hypnogram >= 0) & (hypnogram <= 4)

    else:
        keep_idx = (hypnogram >= 0) & (hypnogram <= 4)

    if not isinstance(keep_idx, list):
        psg_seg = {chn: signal[:, keep_idx, :] for chn, signal in psg_seg.items()}
        hypnogram = hypnogram[keep_idx]

    category_counts = None

    if hypnogram is not None and len(hypnogram) > 0:
        # 计算0-4每个类别的数量（minlength=5确保所有类别都被统计）
        counts = np.bincount(hypnogram, minlength=5)
        category_counts = {
            "清醒(W)": counts[0],
            "N1": counts[1],
            "N2": counts[2],
            "N3": counts[3],
            "REM(R)": counts[4],
            "总epoch数": np.sum(counts)
        }
        # 打印日志
        LOG.info(f'{cohort: <5} | {subset: <5} | {fileID: <5} | 标签统计: {category_counts}')

    return psg_seg, hypnogram, category_counts


def process_cohort(paths_cohort, name_cohort):

    # Get a sorted list of all the EDFs
    if name_cohort in ["cassette","telemetry"]:
        list_edf = sorted(glob(os.path.join(paths_cohort['edf'],
                                            "*[Pp][Ss][Gg].[Ee][Dd][Ff]")))

    elif name_cohort in ["vincents"]:
        list_edf = sorted(glob(os.path.join(paths_cohort['edf'],
                                            "*.[Rr][Ee][Cc]")))

    elif name_cohort in ['apnea',"artifacts","kcomplexes","patients","PLMs","REMs","Subjects"]:
        list_edf = sorted(glob(os.path.join(paths_cohort['edf'],'*.[Ee][Dd][Ff]')))

    elif name_cohort in ["haaglanden"]:
        list_edf = sorted(glob(os.path.join(paths_cohort['edf'],
                                            "*.[Ee][Dd][Ff]")))
        list_edf = [f for f in list_edf if "_sleepscoring" not in f]

    else:
        list_edf = sorted(glob(paths_cohort['edf'] + '/**/*.[EeRr][DdEe][FfCc]', recursive=True))

    if not list_edf:
        LOG.info('{: <5} | Cohort is empty, skipping'.format(name_cohort))
        return None

    # This returns a file ID (ie. xxx.edf becomes xxx)
    if name_cohort in ["cassette","telemetry"]:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-10]) for edf in list_edf]))

    elif name_cohort in ["vincents"]:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-4]) for edf in list_edf]))

    elif name_cohort in ['apnea',"artifacts","kcomplexes","patients","PLMs","REMs","Subjects"]:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-4]) for edf in list_edf]))

    elif name_cohort in ["haaglanden"]:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-4]) for edf in list_edf]))

    else:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-4]) for edf in list_edf]))


    # Get a list of the hypnograms
    if name_cohort in ["cassette","telemetry"]:
        list_hypnogram = sorted(
            glob(os.path.join(paths_cohort['stage'],
                              "*[Hh][Yy][Pp][Nn][Oo][Gg][Rr][Aa][Mm].[Ee][Dd][Ff]")))

    elif name_cohort in ["vincents"]:
        list_hypnogram = sorted(
            glob(os.path.join(paths_cohort['stage'],
                              "*_stage.[Tt][Xx][Tt]")))

    elif name_cohort in ["apnea","artifacts","kcomplexes","PLMs","REMs"]:
        # 定义筛选规则（便于后续修改）
        hypnogram_pattern = "Hypnogram_excerpt*.[Tt][Xx][Tt]"  # *匹配任意数字/字符（支持1/10/100等）
        # 拼接路径+递归搜索
        list_hypnogram = sorted(
            glob(os.path.join(paths_cohort['stage'], hypnogram_pattern), recursive=True))

    elif name_cohort in ["patients"]:
        hypnogram_pattern = "HypnogramAASM_patient*.[Tt][Xx][Tt]"
        list_hypnogram = sorted(
            glob(os.path.join(paths_cohort['stage'], hypnogram_pattern), recursive=True))

    elif name_cohort in ["Subjects"]:
        hypnogram_pattern = "HypnogramAASM_subject*.[Tt][Xx][Tt]"
        list_hypnogram = sorted(
            glob(os.path.join(paths_cohort['stage'], hypnogram_pattern), recursive=True))

    elif name_cohort in ["haaglanden"]:
        list_hypnogram = sorted(
            glob(os.path.join(paths_cohort['stage'],
                              "*[Ss][Ll][Ee][Ee][Pp][Ss][Cc][Oo][Rr][Ii][Nn][Gg].[Ee][Dd][Ff]")))
    else:
        return None


    # Make sure that we only keep those recordings who have a corresponding hypnogram
    if name_cohort in ["cassette","telemetry"]:
        hyp_IDs = [os.path.split(hypID)[1][:6] for hypID in list_hypnogram]

    elif name_cohort in ["vincents"]:
        hyp_IDs = [os.path.split(hypID)[1][:8] for hypID in list_hypnogram]

    elif name_cohort in ["apnea","artifacts","kcomplexes","PLMs","REMs"]:
        hyp_IDs = [os.path.split(hypID)[1][10:-4] for hypID in list_hypnogram]

    elif name_cohort in ["patients", "Subjects"]:
        hyp_IDs = [os.path.split(hypID)[1][14:-4] for hypID in list_hypnogram]

    elif name_cohort in ["haaglanden"]:
        hyp_IDs = [os.path.split(hypID)[1][:5] for hypID in list_hypnogram]


    # Depending on the cohort, subjectID is found in different ways
    if name_cohort in ["cassette", "telemetry"]:
        list_subjectID = list_fileID

    elif name_cohort in ['apnea']:
        list_subjectID = [fileID[6:] for fileID in list_fileID]

    elif name_cohort in ["artifacts"]:
        list_subjectID = [fileID[10:] for fileID in list_fileID]

    elif name_cohort in ["PLMs"]:
        list_subjectID = [fileID[5:] for fileID in list_fileID]

    elif name_cohort in ["REMs"]:
        list_subjectID = [fileID[5:] for fileID in list_fileID]
    else:
        list_subjectID = list_fileID

    list_ID_union = list(set(list_subjectID) & set(hyp_IDs))

    # 收集要删除的标签索引（hyp_IDs中不在交集的ID对应的索引）
    del_hyp_indices = []
    for idx, id in enumerate(hyp_IDs):
        if id not in list_ID_union:
            del_hyp_indices.append(idx)

    # 倒序删除（避免索引偏移），同步删除hyp_IDs和list_hypnogram
    for idx in reversed(del_hyp_indices):
        removed_id = hyp_IDs[idx]
        LOG.info('{: <5} | Removing label for ID: {}'.format(name_cohort, removed_id))
        hyp_IDs.pop(idx)  # 删除无效ID
        list_hypnogram.pop(idx)  # 删除对应标签路径

    # 收集要删除的PSG索引（list_fileID中不在交集的ID对应的索引）
    del_edf_indices = []
    for idx, id in enumerate(list_subjectID):
        if id not in list_ID_union:
            del_edf_indices.append(idx)

    # 倒序删除，同步删除list_fileID和list_edf
    for idx in reversed(del_edf_indices):
        removed_id = list_fileID[idx]
        LOG.info('{: <5} | Removing PSG for ID: {}'.format(name_cohort, removed_id))
        list_fileID.pop(idx)  # 删除无效ID
        list_edf.pop(idx)  # 删除对应PSG路径

    # Update fileID
    if name_cohort in ["cassette","telemetry"]:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-10]) for edf in list_edf]))

    elif name_cohort in ["vincents"]:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-4]) for edf in list_edf]))

    elif name_cohort in ['apnea',"artifacts","kcomplexes","patients","PLMs","REMs","Subjects"]:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-4]) for edf in list_edf]))

    elif name_cohort in ["haaglanden"]:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-4]) for edf in list_edf]))

    else:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-4]) for edf in list_edf]))

    # Create empty dataframe for cohort
    df_cohort = pd.DataFrame(
        columns=['File', 'Hypnogram', 'FileID', 'SubjectID',
                 'Cohort', 'Partition', 'Skip', 'HypnogramLength']).fillna(0)
    df_cohort['File'] = list_edf
    df_cohort['Hypnogram'] = list_hypnogram
    df_cohort['FileID'] = list_fileID
    df_cohort['SubjectID'] = list_subjectID
    df_cohort['Cohort'] = name_cohort

    # Define train/eval/test split
    unique_subjects = sorted(list(set(df_cohort['SubjectID'])))
    n_subjects = len(unique_subjects)
    LOG.info('Current cohort: {: <5} | Total: {} subjects, {} EDFs'.format(
        name_cohort, n_subjects, len(list_edf)))
    seed(name_cohort[0])
    shuffle(unique_subjects)
    trainID, evalID, testID = np.split(unique_subjects,
                                       [int(PARTITIONS['TRAIN'] * n_subjects),
                                        int((PARTITIONS['TRAIN'] + PARTITIONS['EVAL']) * n_subjects)])
    LOG.info('{: <5} | Assigning subjects to subsets: {}/{}/{} train/eval/test'.format(
        name_cohort, len(trainID), len(evalID), len(testID)))
    for id in df_cohort['SubjectID']:
        if id in trainID:
            df_cohort.loc[df_cohort['SubjectID'] == id, 'Partition'] = 'train'
        elif id in evalID:
            df_cohort.loc[df_cohort['SubjectID'] == id, 'Partition'] = 'eval'
        elif id in testID:
            df_cohort.loc[df_cohort['SubjectID'] == id, 'Partition'] = 'test'
        else:
            print('No subset assignment for {}.'.format(id))

    # Process files
    stats_list = []
    for idx, row in df_cohort.iterrows():
        psg, hypnogram, category_counts = process_file(row)

        if psg is None:
            LOG.info('{: <5} | Skipping file: {}'.format(
                name_cohort, row['FileID']))
            df_cohort.loc[idx, 'Skip'] = 1
        else:
            psg = np.concatenate([psg[mod]
                                 for mod in ['eeg', 'eog', 'emg']], axis=0)
            N = np.min(
                [len(hypnogram), psg.shape[1]])
            LOG.info('{: <5} | {} | Writing {} epochs'.format(
                name_cohort, row['FileID'], N))

            # Write H5 file for subject
            write_H5(psg, hypnogram, N, row, name_cohort=name_cohort)
            df_cohort.loc[idx, 'HypnogramLength'] = N

        # 新增：初始化统计结果列表

        if category_counts is not None:
            stats_row = {
                "受试者ID": row['FileID'],
                "队列": name_cohort,
                "子集": row['Partition'],
                **category_counts  # 合并类别统计结果
            }
            stats_list.append(stats_row)

    if stats_list:
            # 定义CSV路径和列名
        # csv_path = os.path.join(OUTPUT_DIRECTORY, 'csv', f'{name_cohort}_category_stats.csv')
        csv_path = os.path.join(OUTPUT_DIRECTORY, 'csv', 'category_stats.csv')
        file_csv_exists = os.path.isfile(csv_path)
            # 列名顺序（确保一致性）
        columns = [
                "受试者ID", "队列", "子集",
                "清醒(W)", "N1", "N2", "N3", "REM(R)", "总epoch数"
            ]
            # 写入CSV

        stats_df = pd.DataFrame(stats_list, columns=columns)

        stats_df.to_csv(csv_path, mode='a', header=not file_csv_exists, index=False, encoding='utf-8-sig')
        LOG.info(f'{name_cohort} | 类别统计已保存到: {csv_path}')

    return df_cohort


def main():
    LOG.info('Processing cohorts: {}'.format([*COHORTS]))
    df = []

    # Loop over the different cohorts
    for name_cohort, cohort in COHORTS.items():

        LOG.info('Processing cohort: {}'.format(name_cohort))

        if not cohort['edf'] or not os.path.exists(cohort['edf']):
            LOG.info('Skipping cohort: {}'.format(name_cohort))
            continue

        # process_cohort(current_cohort_overview, current_cohort)
        df_cohort = process_cohort(cohort, name_cohort)

        if isinstance(df_cohort, pd.DataFrame):

            filename = os.path.join(
                OUTPUT_DIRECTORY, 'csv', name_cohort + '.csv')
            df_cohort.to_csv(filename)

            # filename = os.path.join(
            #     OUTPUT_DIRECTORY, 'csv', 'total.csv')
            # file_exists = os.path.isfile(filename)
            # df_cohort.to_csv(filename, mode='a',header=not file_exists,index=False)


    LOG.info('Processing cohorts finalized.')


if __name__ == '__main__':
    main()
