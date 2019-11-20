import importlib

import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

import itertools
from operator import itemgetter
from openpyxl import load_workbook
import pickle

import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, DataLoader

def train_val_test_split(path, output, train_ratio, val_ratio, test_ratio):
    filelist = os.listdir(path)
    np.random.shuffle(filelist)

    sum_ratio = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / sum_ratio
    val_ratio = val_ratio / sum_ratio
    test_ratio = test_ratio / sum_ratio

    train_list = filelist[0:int(len(filelist) * train_ratio)]
    val_list = filelist[int(len(filelist) * train_ratio):-int(len(filelist) * test_ratio)]
    test_list = filelist[-int(len(filelist) * test_ratio):]

    print('train:', len(train_list), 'val:', len(val_list), 'test:', len(test_list))
    np.savetxt(os.path.join(output,'train_list.txt'), train_list, fmt='%s', delimiter='\n')
    np.savetxt(os.path.join(output,'val_list.txt'), val_list, fmt='%s', delimiter='\n')
    np.savetxt(os.path.join(output,'test_list.txt'), test_list, fmt='%s', delimiter='\n')
    print('filelist saved to:', output)

def read_xlsx(path):
    workbook = load_workbook(path)
    sheet = workbook[workbook.sheetnames[0]]
    data = sheet.values
    cols = next(data)[0:]
    data = list(data)
    data = (itertools.islice(r, 0, None) for r in data)
    df = pd.DataFrame(data, columns = cols)
    return df
    
class PNCEnrichedSet(InMemoryDataset):
    def __init__(self, sub_list, output, root, path_data, path_label, target_name = None, feature_mask = None, **kwargs):
        self.path = [path_data, path_label]
        self.output = output
        if sub_list is None:
            sub_list = os.listdir(path_data)
        self.sub_list = sub_list
        self.target_name = target_name
        self.feature_mask = feature_mask
        super(PNCEnrichedSet, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['pnc_enriched_raw.pkl']

    @property
    def processed_file_names(self):
        return [self.output]

    def download(self):
        assert len(self.path) == 2
        path_data = self.path[0]
        path_label = self.path[1]
        
        labels = read_xlsx(path_label)
        labels['Sex'], uniques = pd.factorize(labels['Sex'])
        labels = itemgetter('ScanAgeYears','Sex')(labels.set_index('Subject').to_dict())
        
        subjlist = os.listdir(path_data)
        #filelist = [fname for fname in os.listdir(os.path.join(path_data, subjlist[0])) if 'matrix' in fname]
        filelist = ['fdt_network_matrix_lengths',
                    'Enriched_mean_matrix_0', 'Enriched_variance_matrix_0',
                    'Enriched_mean_matrix_1', 'Enriched_variance_matrix_1',
                    'Enriched_mean_matrix_2', 'Enriched_variance_matrix_2',
                    'Enriched_mean_matrix_3', 'Enriched_variance_matrix_3']

        with open(os.path.join(self.raw_dir, 'pnc_enriched_raw_info.txt'), 'w') as f:
            print('Label info:', file = f)
            print('Sex labels (0/1):', uniques.values, file = f)
            print('\n', file = f)
            print('Enriched features:', file = f)
            print(filelist, sep='\n', file = f)
            print('\n', file = f)
            print('Saved subjects:', file = f)
            print(subjlist, sep='\n', file = f)
            print('\n', file = f)

        dataset = {}
        for subj in subjlist:
            print('downloading', subj, '...')
            filepath = os.path.join(path_data, subj, 'fdt_network_matrix')
            matrix = torch.tensor(np.loadtxt(filepath), dtype=torch.float)
            edge_index, value = dense_to_sparse(matrix)
            y = {'ScanAgeYears': labels[0][subj], 'Sex': labels[1][subj]}
            x = torch.ones([matrix.shape[0], 1], dtype=torch.float)
            enriched = []
            for file in filelist:
                filepath = os.path.join(path_data, subj, file)
                matrix = torch.tensor(np.loadtxt(filepath), dtype=torch.float)
                enriched.append(matrix[edge_index[0], edge_index[1]])     
            data = Data(x = x, edge_index = edge_index, edge_attr = value, y = y)
            data.enriched = enriched           
            dataset[subj] = data
    
        with open(os.path.join(self.raw_dir, 'pnc_enriched_raw.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
            print('PNC dataset saved to path:', self.raw_dir)
        
    def process(self):
        with open(os.path.join(self.raw_dir, 'pnc_enriched_raw.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        dataset_list = []
        
        sub_list = np.loadtxt(self.sub_list, dtype = str, delimiter = '\n')

        for subj in sub_list:
            data = dataset[subj]
            if self.target_name is not None:
                data.y = data.y[self.target_name]
            if self.feature_mask is not None:
                data.enriched = [data.enriched[i] for i in self.feature_mask]
            data.enriched = torch.stack(data.enriched, dim=-1)
            dataset_list.append(data)
            
        self.data, self.slices = self.collate(dataset_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
        print('Processed dataset saved as', self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def get_data_loaders(config):
    assert 'loaders' in config, 'Could not find loaders configuration'
    loaders_config = config['loaders']
    class_name = loaders_config.pop('name')
    train_list = loaders_config.pop('train_list')
    val_list = loaders_config.pop('val_list')
    output_train = loaders_config.pop('output_train')
    output_val = loaders_config.pop('output_val')
    batch_size = loaders_config.pop('batch_size')

    m = importlib.import_module('utils.data_handler')
    clazz = getattr(m, class_name)

    return {
        'train': DataLoader(clazz(train_list, output_train, **loaders_config), batch_size=batch_size, shuffle=True),
        'val': DataLoader(clazz(val_list, output_val, **loaders_config), batch_size=batch_size, shuffle=True)
        }