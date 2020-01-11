import importlib

import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from operator import itemgetter
import pickle
from shutil import copyfile

import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset

from nilearn.connectome import ConnectivityMeasure

from utils.helper import read_xlsx

def train_val_test_split(path, output, train_ratio, val_ratio, test_ratio):
    if os.path.splitext(path)[1] == '.txt':
        filelist = np.loadtxt(path, dtype=str)
    else:
        filelist = os.listdir(path)
    np.random.shuffle(filelist)

    sum_ratio = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / sum_ratio
    val_ratio = val_ratio / sum_ratio
    test_ratio = test_ratio / sum_ratio

    train_list = filelist[0:int(len(filelist) * train_ratio)]
    val_list = filelist[int(len(filelist) * train_ratio):int(len(filelist) * (train_ratio + val_ratio))]
    test_list = filelist[int(len(filelist) * (train_ratio + val_ratio)):]

    print('train:', len(train_list), 'val:', len(val_list), 'test:', len(test_list))
    np.savetxt(os.path.join(output,'train_list.txt'), train_list, fmt='%s', delimiter='\n')
    np.savetxt(os.path.join(output,'val_list.txt'), val_list, fmt='%s', delimiter='\n')
    np.savetxt(os.path.join(output,'test_list.txt'), test_list, fmt='%s', delimiter='\n')
    print('filelist saved to:', output)

#train_val_test_split('/home/sherry/Dropbox/PhD/Data/PNC_Enriched/Schaefer/Origin', '/home/sherry/Dropbox/PhD/Data/PNC_Enriched/pnc_schaefer_exp01', 0.7, 0.2, 0.1)

def cross_validation_split(path, output, num_fold, sel_fold = 0, shuffle = True):
    if os.path.splitext(path)[1] == '.txt':
        filelist = np.loadtxt(path, dtype=str)
    else:
        filelist = os.listdir(path)
    if shuffle:
        np.random.shuffle(filelist)
        np.savetxt(os.path.join(output,'all_list_shuffled.txt'), filelist, fmt='%s', delimiter='\n')

    ind0 = int(len(filelist) / num_fold) * sel_fold
    ind1 = int(len(filelist) / num_fold) * (sel_fold + 1)

    train_list = np.concatenate((filelist[ind1:], filelist[0:ind0]))
    val_list = filelist[ind0:ind1]
    test_list = val_list

    print('train:', len(train_list), 'val:', len(val_list), 'test:', len(test_list))
    np.savetxt(os.path.join(output,'train_list.txt'), train_list, fmt='%s', delimiter='\n')
    np.savetxt(os.path.join(output,'val_list.txt'), val_list, fmt='%s', delimiter='\n')
    np.savetxt(os.path.join(output,'test_list.txt'), test_list, fmt='%s', delimiter='\n')
    print('filelist saved to:', output)

#cross_validation_split('/home/sherry/Dropbox/PhD/Results/pnc_strucfunc_exp01/train_all.txt', '/home/sherry/Dropbox/PhD/Results/pnc_strucfunc_exp01', 10, 0, False)

def arrange_data(path, output):
    subpaths = [f.path for f in os.scandir(path) if f.is_dir()]
    for subpath in subpaths:
        subjlist = os.listdir(subpath)
        for subj in subjlist:
            if subj == 'exclude' or os.path.isfile(os.path.join(subpath, subj)):
                continue
            out_subpath = os.path.join(output, subj)
            if not os.path.exists(out_subpath):
                os.makedirs(out_subpath)
            filelist = os.listdir(os.path.join(subpath, subj))
            id = subj.split('_')[1]
            for filename in filelist:
                out_filename = os.path.basename(subpath)+ '_' +filename.split(id+'_')[-1]
                copyfile(os.path.join(subpath, subj, filename), os.path.join(out_subpath, out_filename))
    print('done!')
    return

#arrange_data('/home/sherry/Dropbox/PhD/Data/PNC_Enriched/ForSherry', '/home/sherry/Dropbox/PhD/Data/PNC_Enriched/PNC_Connectomes')

def rename_data(path):
    subjlist = [fname for fname in os.listdir(path) if os.path.isdir(os.path.join(path, fname)) and fname != 'exclude']
    for subj in subjlist:
        filelist = [fname for fname in os.listdir(os.path.join(path, subj)) if 'Restbold' in fname]
        for filename in filelist:
            newname = filename.replace('_Schaefer2018_','_')
            newname = newname.replace('Parcels_LPS_dil2_', '_')
            os.rename(os.path.join(path, subj,filename), os.path.join(path, subj, newname))
    print('done!')
#rename_data('/home/sherry/Dropbox/PhD/Data/PNC_Enriched/PNC_Connectomes')

class ABIDESet(InMemoryDataset):
    def __init__(self, sub_list, output, root, path_data, path_label, target_name = None, feature_mask = None, **kwargs):
        self.path = [path_data, path_label]
        self.output = output
        if sub_list is None:
            sub_list = os.listdir(path_data)
        self.sub_list = sub_list
        self.target_name = target_name
        self.feature_mask = feature_mask
        super(ABIDESet, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['abide_raw.pkl', 'abide_raw_info.txt']

    @property
    def processed_file_names(self):
        return [self.output]

    def download(self):
        assert len(self.path) == 2
        path_data = self.path[0]
        path_label = self.path[1]
        
        labels = read_xlsx(path_label)
        labels = labels.astype({'subject': 'str'})
        labels['SITE_ID'], uniques = pd.factorize(labels['SITE_ID'])
        labels['DX_GROUP'] = 2 - labels['DX_GROUP']
        labels['SEX'] = labels['SEX'] - 1

        labels = itemgetter('SITE_ID','DX_GROUP','DSM_IV_TR','AGE_AT_SCAN','SEX')(labels.set_index('subject').to_dict())
        
        subjlist = os.listdir(path_data)
        filelist = [fname for fname in os.listdir(os.path.join(path_data, subjlist[0])) if 'matrix' in fname]

        with open(os.path.join(self.raw_dir, 'abide_raw_info.txt'), 'w') as f:
            print('Label info:', file = f)
            print('All labels:', 'SITE_ID','DX_GROUP','DSM_IV_TR','AGE_AT_SCAN','SEX')
            print('Site labels (0-n):', uniques.values, file = f)
            print('DX_GROUP (0/1):', 'control, autism', file = f)
            print('DSM_IV_TR (0-n):', 'control, autism, aspergers, PDD-NOS, aspergers or PDD-NOS', file = f)
            print('SEX (0/1):', 'M, F', file = f)
            print('\n', file = f)
            print('Features:', file = f)
            print(filelist, sep='\n', file = f)
            print('\n', file = f)
            print('Saved subjects:', file = f)
            print(subjlist, sep='\n', file = f)
            print('\n', file = f)

        dataset = {}
        for subj in subjlist:
            print('downloading', subj, '...')
            features = []
            for file in filelist:
                filepath = os.path.join(path_data, subj, file)
                # origianl value (-1 ~ 1), adjust value of the matrix to 0 ~ 2
                matrix = torch.tensor(np.loadtxt(filepath), dtype=torch.float32) + 1
                features.append(matrix)
            
            y = {'SITE_ID': labels[0][subj], 'DX_GROUP': labels[1][subj], 'DSM_IV_TR': labels[2][subj],
                    'AGE_AT_SCAN': labels[3][subj], 'SEX': labels[4][subj]}
            x = torch.ones([matrix.shape[0], 1], dtype=torch.float32)
            data = Data(x = x, y = y)
            data.features = features           
            dataset[subj] = data
    
        with open(os.path.join(self.raw_dir, 'abide_raw.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
            print('ABIDE dataset saved to path:', self.raw_dir)
        
    def process(self):
        with open(os.path.join(self.raw_dir, 'abide_raw.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        dataset_list = []
        
        sub_list = np.loadtxt(self.sub_list, dtype = str, delimiter = '\n')

        for subj in sub_list:
            data = dataset[subj]
            if self.target_name is not None:
                data.y = data.y[self.target_name]
            if self.feature_mask is not None:
                data.features = [data.features[i] for i in self.feature_mask]

            edge_index, _ = dense_to_sparse(torch.ones(data.features[0].shape, dtype=torch.float32))
            edge_attr = []
            for feature in data.features:
                edge_attr.append(feature[edge_index[0], edge_index[1]])
            data.edge_index = edge_index
            data.edge_attr = torch.stack(edge_attr, dim = -1)
            data.features = torch.stack(data.features, dim = -1)
            data.features = torch.unsqueeze(data.features, dim = 0)
            dataset_list.append(data)
            
        self.data, self.slices = self.collate(dataset_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
        print('Processed dataset saved as', self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


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
        return ['pnc_features_raw.pkl']

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
        
        subjlist = [fname for fname in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, fname)) and fname != 'exclude']
        filelist = os.listdir(os.path.join(path_data, subjlist[0]))
        filelist.sort()
        ts_index = [i for i in range(len(filelist)) if 'timeseries' in filelist[i]]

        min_ts_length = None
        for subj in subjlist:
            print('checking', subj, '...')
            filename = filelist[ts_index[0]]
            filepath = os.path.join(path_data, subj, filename)
            if not os.path.exists(filepath):
                continue
            matrix = np.loadtxt(filepath)
            if min_ts_length is None or matrix.shape[0] < min_ts_length:
                min_ts_length = matrix.shape[0]
    
        with open(os.path.join(self.raw_dir, 'pnc_enriched_raw_info.txt'), 'w') as f:
            print('Label info:', file = f)
            print('All labels:', 'ScanAgeYears','Sex')
            print('Sex labels (0/1):', uniques.values, file = f)
            print('Timeseries length (min):', min_ts_length, file = f)
            print('\n', file = f)
            print('Features:', file = f)
            print(filelist, sep='\n', file = f)
            print('\n', file = f)
            print('Saved subjects:', file = f)
            print(subjlist, sep='\n', file = f)
            print('\n', file = f)

        with open(os.path.join(self.raw_dir, 'pnc_features_raw.pkl'), 'wb') as f:
            pickle.dump([labels, path_data, filelist, subjlist, min_ts_length], f)
            print('PNC dataset saved to path:', self.raw_dir)
        
    def process(self):
        with open(os.path.join(self.raw_dir, 'pnc_features_raw.pkl'), 'rb') as f:
            labels, path_data, filelist, _, min_ts_length = pickle.load(f)

        if self.feature_mask is not None:
            if np.isscalar(self.feature_mask):
                self.feature_mask = [i for i in range(len(filelist)) if self.feature_mask == int(filelist[i].split('_')[1])]
            filelist = [filelist[i] for i in self.feature_mask]
        ts_index = [i for i in range(len(filelist)) if 'timeseries' in filelist[i]]
        sc_index = [i for i in range(len(filelist)) if 'connmat' in filelist[i]]

        dataset_list = []
        sub_list = np.loadtxt(self.sub_list, dtype = str, delimiter = '\n')
        epsilon = 1e-5
        for subj in sub_list:
            print('processing', subj, '...')
            features = []
            for filename in filelist:
                filepath = os.path.join(path_data, subj, filename)
                if not os.path.exists(filepath):
                    raise ValueError('invalid path '+filepath)
                matrix = np.loadtxt(filepath)
                features.append(matrix)

            data = Data(x = None, y = None)
            data.y = {'ScanAgeYears': labels[0][subj], 'Sex': labels[1][subj]}
            data.subj = int(subj.split('_')[0])
            if self.target_name is not None:
                data.y = data.y[self.target_name]
            ts = []
            for i in ts_index:
                ts.append(features[i][:min_ts_length, :])
            data.fconn = torch.tensor(ConnectivityMeasure(kind='correlation').fit_transform(ts), dtype=torch.float32)
            sc = []
            for i in sc_index:
                sc_matrix = features[i] + epsilon
                sc.append(sc_matrix / np.sum(sc_matrix, axis = 0))
            data.sconn = torch.tensor(sc, dtype=torch.float32)
            data.x = data.fconn[0]
            data.edge_index, _ = dense_to_sparse(torch.ones(data.sconn[0].shape, dtype=torch.float32))
            data.edge_attr = data.sconn[0].clone().detach()[data.edge_index[0], data.edge_index[1]]
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
    loader_class_name = loaders_config.pop('loader_name')
    train_list = loaders_config.pop('train_list')
    val_list = loaders_config.pop('val_list')
    test_list = loaders_config.pop('test_list')
    output_train = loaders_config.pop('output_train')
    output_val = loaders_config.pop('output_val')
    output_test = loaders_config.pop('output_test')
    batch_size = loaders_config.pop('batch_size')

    m = importlib.import_module('utils.data_handler')
    clazz = getattr(m, class_name)
    m = importlib.import_module('torch_geometric.data')
    loader_clazz =getattr(m, loader_class_name)

    return {
        'train': loader_clazz(clazz(train_list, output_train, **loaders_config), batch_size=batch_size, shuffle=True),
        'val': loader_clazz(clazz(val_list, output_val, **loaders_config), batch_size=batch_size, shuffle=True),
        'test': loader_clazz(clazz(test_list, output_test, **loaders_config), batch_size=batch_size, shuffle=True)
        }