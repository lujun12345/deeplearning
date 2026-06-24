import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from h5py import File
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)
    from joblib import Memory, delayed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.config import process_config
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)
    from src.utils.parallel_bar import ParallelExecutor

DEFAULT_HYP_LENGTH = 30


class MultiCohortDataset(Dataset):

    def __init__(self, config, subject=None, subset='test'):
        self.config = config
        self.subject = subject
        self.subset = subset

        self.batch_size = config.data_loader.batch_size[subset]
        self.data = config.data_loader.data[subset]
        self.data_dir = config.data_loader.data_dir

        self.train_fraction = config.data_loader.train_fraction if subset == 'train' else None
        self.fs = 128  # TODO: put into yaml
        self.modalities = config.data_loader.modalities

        self.num_channels = config.data_loader.num_channels
        self.num_classes = config.data_loader.num_classes
        self.segment_length = config.data_loader.segment_length
        # assert self.segment_length % DEFAULT_HYP_LENGTH == 0, 'Segment length should be a multiple of 30 s!'

        self.df = None
        self.indexes = None
        self.length_recordings = None
        self.num_subjects = None

        # Collect dataframes for cohorts
        cohorts = [data[0] for data in self.data]
        subsets = [data[1] for data in self.data]
        df = pd.concat(
            [pd.read_csv(os.path.join(self.data_dir, 'csv', cohort + '.csv'), index_col=0) for cohort
             in list(set(cohorts))], ignore_index=True)

        # Prune for subsets in cohorts
        self.df = []
        for c, s in zip(cohorts, subsets):
            self.df.append(df.loc[(df.Cohort == c) & (df.Partition == s)])
        self.df = pd.concat(self.df, ignore_index=True)

        # Prune for subsets in files

        list_files = os.listdir(os.path.join(self.data_dir, 'h5'))
        self.df = self.df.loc[df.FileID.str.lower().isin([f[:-3] for f in list_files]), :] \
                         .sort_values(by=['Cohort', 'FileID']) \
                         .reset_index(drop=True)

        # Prune for skipped subjects
        self.df = self.df.loc[pd.isna(self.df.Skip)].reset_index(drop=True)

        # Maybe only take a fraction of training data. This routine sorts the available cohorts by size and adjusts the number of
        # PSGs taken from each successively.
        if self.train_fraction:
            print('Using {} of the data'.format(self.train_fraction))
            if self.train_fraction > 1.0:
                grab_from_each = int(self.train_fraction // len(cohorts))
                n_cohort = {c: None for c in cohorts}
                sum_in_cohorts = [(c, sum(self.df.Cohort == c)) for c in cohorts]
                total = 0
                remaining_cohorts = len(cohorts)
                for c, cohort_sum in sorted(sum_in_cohorts, key=lambda cohort_sum: cohort_sum[1]):
                    if cohort_sum >= grab_from_each:
                        n_cohort[c] = grab_from_each
                        total += grab_from_each
                        remaining_cohorts -= 1
                    else:
                        n_cohort[c] = cohort_sum
                        total += cohort_sum
                        remaining_cohorts -= 1
                        if remaining_cohorts > 0:
                            grab_from_each = int((self.train_fraction - total) // remaining_cohorts)
                        else:
                            print('[ Warning ] No more cohorts to draw data from, lower the requested amount of data!')
                print(f'[ Info ] Requested {self.train_fraction} / received {total}')
                # n_cohort = {c: np.minimum(int(self.train_fraction // len(cohorts)), sum(self.df.Cohort == c)) for c in cohorts}
            df_frac = []
            for c in cohorts:
                if self.train_fraction <= 1.0:
                    df_frac.append(self.df.loc[self.df.Cohort == c].sample(frac=self.train_fraction))
                else:
                    df_frac.append(self.df.loc[self.df.Cohort == c].sample(n=n_cohort[c]))
            self.df = pd.concat(df_frac, ignore_index=True).sort_values(by=['Cohort', 'FileID']) \
                                                           .reset_index(drop=True)

        # Maybe select single subject
        if self.subject:
            self.df = self.df[self.df.FileID == self.subject].sort_values(by=['Cohort', 'FileID']) \
                                                             .reset_index(drop=True)

        self.num_subjects = len(self.df)
        print('Number of subjects: {}'.format(self.num_subjects))


        def get_h5(file):
            with File(os.path.join(self.data_dir, 'h5', file.lower()), 'r') as db:
                with np.printoptions(precision=2, threshold=5, edgeitems=1):
                    hypnogram = db['hypnogram'][:].astype(np.uint8)
                    psg = db['data'][:].astype(np.float32)

            return file.split('.')[0], psg, hypnogram

        # Preloading data as mmaps with joblib
        self.cache_dir = 'data/processed/.cache'
        # memory = Memory(self.cache_dir, mmap_mode='r', verbose=0)
        memory = Memory(location=None, verbose=0)
        get_data = memory.cache(get_h5)

        self.data = {r: None for r in self.df.FileID}
        self.hypnogram = {r: None for r in self.df.FileID}

        with np.printoptions(precision=2, threshold=5, edgeitems=1):
            data = ParallelExecutor(n_jobs=-1, prefer="threads")(total=len(self.df))(
                delayed(get_data)(k + '.h5') for k in self.data.keys()
            )
        for record, psg, hypnogram in tqdm(data, desc='Processing... '):
            self.data[record] = psg
            self.hypnogram[record] = hypnogram


        # We change the hypnogram length depending on the segment size
        self.mult_factor = self.segment_length // DEFAULT_HYP_LENGTH    #10
        # self.length_recordings = (
        #             self.df.HypnogramLength.values // self.mult_factor).astype(np.int)
        # self.df['Length'] = self.length_recordings
        # self.df['FileLength'] = self.df.HypnogramLength * DEFAULT_HYP_LENGTH * self.fs
        #
        self.length_recordings = []
        for fid in self.df['FileID']:
            # 获取 h5 中实际的 hypnogram 长度
            hyp_len = self.hypnogram[fid].shape[0]
            data_len = self.data[fid].shape[1]
            # 计算该文件的有效分段数（避免越界）
            valid_len = min(hyp_len, data_len)
            valid_segments = valid_len // self.mult_factor
            self.length_recordings.append(valid_segments)
        self.length_recordings = np.array(self.length_recordings, dtype=np.int)

        # 更新 df 中的长度字段
        self.df['Length'] = self.length_recordings
        self.df['HypnogramLength'] = [self.hypnogram[fid].shape[0] for fid in self.df['FileID']]  # 替换为实际长度
        self.df['FileLength'] = self.df['HypnogramLength'] * DEFAULT_HYP_LENGTH * self.fs


        self.indexes = []
        for i, fid in zip(np.arange(self.num_subjects), self.df['FileID']):
            n_segments = self.length_recordings[i]
            data_len = self.data[fid].shape[1]  # data 的时间步长度
            hyp_len = self.hypnogram[fid].shape[0]
            for j in range(n_segments):
                # 构造合法的索引区间：[j*mult_factor, (j+1)*mult_factor]
                start = j * self.mult_factor
                end = start + self.mult_factor
                if end > data_len or end > hyp_len:
                    continue  # 跳过越界的分段

                self.indexes.append((fid, range(start, end)))

    def __len__(self):
        # return sum(self.length_recordings)
        return len(self.indexes)

    def __getitem__(self, idx):
        file_id, position = self.indexes[idx]
        cohort = self.df.loc[self.df.FileID == file_id, 'Cohort'].values[0]
        # 将 range 对象转换为切片，避免索引错误
        position_slice = slice(position.start, position.stop)
        # 边界检查
        if position.stop > self.data[file_id].shape[1]:
            raise IndexError(f"Position {position} out of bounds for file {file_id}")

        raw_data = self.data[file_id]  # 维度：[通道数, 时间步, 其他]
        raw_hyp = self.hypnogram[file_id]

        # 1. 校验通道数是否匹配
        if raw_data.shape[0] != self.num_channels:
            raise ValueError(f"文件 {file_id} 通道数异常：实际 {raw_data.shape[0]}，预期 {self.num_channels}")

        # 2. 校验切片长度是否合法
        slice_len = position.stop - position.start
        if slice_len != self.mult_factor:
            raise ValueError(f"文件 {file_id} 切片长度异常：实际 {slice_len}，预期 {self.mult_factor}")

        data = raw_data[:, position_slice, :]  # [通道数, 切片长度, 其他]
        hypnogram = raw_hyp[position_slice]
        # data = self.data[file_id][:, position_slice, :]
        # hypnogram = self.hypnogram[file_id][position_slice]

        out = {'fid': file_id,
               'position': position,
               'data': torch.from_numpy(data.reshape((self.num_channels, -1))[np.newaxis, :, :]),
               'target': torch.LongTensor(hypnogram)}
        # 'target': torch.LongTensor(np.repeat(hypnogram, DEFAULT_HYP_LENGTH)
        return out

    def get_subjects(self):
        return [fid for fid in self.df.FileID.values]

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    sns.set_context('paper')
    num_workers = 0
    print('Num workers: {}'.format(num_workers))
    config = process_config('./src/configs/exp03-frac100.yaml')
    s = datetime.now()
    train_data = MultiCohortDataset(config, subset='train')
    data = next(iter(train_data))
    e = datetime.now()
    print('{}'.format(e - s))
    eval_data = MultiCohortDataset(config, subset='eval')
    test_data = MultiCohortDataset(config, subset='test')

    train_loader = DataLoader(train_data, batch_size=config.data_loader.batch_size.train,
                              shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    # eval_loader = DataLoader(eval_data, batch_size=config.data_loader.batch_size.eval,
    #                          shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
    # test_loader = DataLoader(test_data, batch_size=config.data_loader.batch_size.test,
    #                          shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
    # # Create a dict with classes as keys and indices for values
    # self.sampling_dict = None
    # if self.mult_factor == 1:
    #     self.sampling_dict = {k: [
    #         idx for idx in self.indexes if self.labels[self.df.FileID[idx[0]]][idx[1]] == k] for k in range(self.num_classes)}
    num_epochs = 5
    start_time = datetime.now()
    for n in range(num_epochs):
        print('\nEpoch {} of {}'.format(n+1, num_epochs))
        # for idx, batch in tqdm(enumerate(train_data), total=len(train_data)):
        #     pass
        # for idx, batch in tqdm(enumerate(eval_data), total=len(eval_data)):
        #     pass
        # for idx, batch in tqdm(enumerate(test_data), total=len(test_data)):
        #     pass
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            pass
        print(idx, batch[0].size(), batch[1].size())
        for idx, batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            pass
    end_time = datetime.now()
    print('\nElapsed time: {} | Time per epoch: {}'.format(end_time - start_time, (end_time - start_time)/num_epochs))
    # for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
    #     pass


    # plt.plot(batch[0][np.random.randint(
    #     0, train_data.batch_size-1), 0, :, :].numpy().T + 2*np.arange(4))
    # plt.show()
