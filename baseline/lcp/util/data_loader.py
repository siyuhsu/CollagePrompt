import os
import os.path as osp
import random
import json
import pickle
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


class ImagenetGridDataset(InMemoryDataset):
    def __init__(self, meta_file="dataset/train/metainfo/imagenet_train_3x3_collage_pred_info.pkl", 
                 img_dir="dataset/train/subset/feat", 
                 processed_path = "baseline/lcp/processed/imagenet_train_3x3_collage_pyg.pt",
                 transform=None,
                 ):
        random.seed(0)
        self.meta_info = pickle.load(open(meta_file, 'rb'))
        
        self.grid_list = list(self.meta_info.keys())
        
        self.num_classes = self.num_classes()
        
        self.img_dir = img_dir
        self.transform = transform

        self.processed_path = processed_path

        if not osp.exists(self.processed_path):
            self.process_data()
        
        self.data, self.slices = torch.load(self.processed_path)
        

    def num_classes(self) -> int:
        num_classes = 1
        return num_classes
    
    def len(self):
        return len(self.grid_list)
    
    def ord2edge(self, ord):
        edges_list = []
        for i in range(len(ord)):
            if (i % 3) != 2:
                st, ed = ord[i], ord[i + 1]
                edges_list.append([st, ed])
                edges_list.append([ed, st])

            if i < 6:
                st, ed = ord[i], ord[i + 3]
                edges_list.append([st, ed])
                edges_list.append([ed, st])
        
        return edges_list
        
    def get_sources(self, idx):
        img_names = self.meta_info[self.grid_list[idx]]['ori']
        return img_names

    def _get_(self, idx):

        img_names = self.meta_info[self.grid_list[idx]]['ori']
        features = [torch.tensor(np.load(os.path.join(self.img_dir, img.replace(".JPEG", ".npy")))) for img in img_names]
        order = self.meta_info[self.grid_list[idx]]['ord']
        edge_list = self.ord2edge(order)
        pred = self.meta_info[self.grid_list[idx]]['pred']
        
        x = torch.squeeze(torch.stack(features, dim=0)).to(dtype=torch.float32)
        edge_index = torch.tensor(edge_list).t().contiguous()
        y = torch.tensor(pred, dtype=torch.float32).mean().unsqueeze(0)
        num_nodes = int(len(img_names))
        data = Data(x, edge_index, y=y, num_nodes=num_nodes)
        data.idx = torch.tensor([idx], dtype=torch.long)

        return data
    
    
    def process_data(self):
        data_list = []
        for i in tqdm(range(self.len())):
            data = self._get_(i)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_path)
 
 
class ImagenetGridTestSet(InMemoryDataset):
    def __init__(self, meta_file="dataset/val/imagenet/metainfo/imagenet_val_3x3_collage_info.json", 
                 collage_dir="dataset/val/imagenet/collage/3x3",
                 img_dir="dataset/val/imagenet/set/feat", 
                 processed_path = "baseline/lcp/processed/imagenet_val_3x3_pyg.pt",
                 transform=None,
                 ):
        random.seed(0)
        self.meta_info = json.load(open(meta_file))
        grid_folders = [ os.path.join(collage_dir, i) for i in os.listdir(collage_dir) if os.path.isdir(os.path.join(collage_dir, i))]
        self.grid_list = [ os.listdir(i)[0] for i in grid_folders]
        self.num_classes = self.num_classes()
        self.img_dir = img_dir
        self.transform = transform

        self.processed_path = processed_path

        if not osp.exists(self.processed_path):
            self.process_data()
        
        self.data, self.slices = torch.load(self.processed_path)

    def num_classes(self) -> int:
        num_classes = 1
        return num_classes
    
    def len(self):
        return len(self.grid_list)
    

    def ord2edge(self, ord):
        edges_list = []
        for i in range(len(ord)):
            if (i % 3) != 2:
                st, ed = ord[i], ord[i + 1]
                edges_list.append([st, ed])
                edges_list.append([ed, st])

            if i < 6:
                st, ed = ord[i], ord[i + 3]
                edges_list.append([st, ed])
                edges_list.append([ed, st])
        
        return edges_list
        
    def get_sources(self, idx):
        img_names = [v['image'] for v in self.meta_info[self.grid_list[idx]].values()]
        return img_names

    def _get_(self, idx):

        img_names = [v['image'] for v in self.meta_info[self.grid_list[idx]].values()]
        features = [torch.tensor(np.load(os.path.join(self.img_dir, img.replace(".JPEG", ".npy")))) for img in img_names]
        order = np.random.permutation(9)
        edge_list = self.ord2edge(order)
        pred = 0
        
        x = torch.squeeze(torch.stack(features, dim=0)).to(dtype=torch.float32)
        edge_index = torch.tensor(edge_list).t().contiguous()
        y = torch.tensor(pred, dtype=torch.float32).mean().unsqueeze(0)
        num_nodes = int(len(img_names))
        data = Data(x, edge_index, y=y, num_nodes=num_nodes)
        data.idx = torch.tensor([idx], dtype=torch.long)

        return data
    
    
    def process_data(self):
        data_list = []
        for i in tqdm(range(self.len())):
            data = self._get_(i)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_path)

 
class Caltech3x3GridTestSet(InMemoryDataset):
    def __init__(self, meta_file="dataset/val/caltech101/metainfo/caltech101_test_3x3_collage_info.json", 
                 collage_dir="dataset/val/caltech101/collage/3x3",
                 img_dir="dataset/val/caltech101/set/feat", 
                 processed_path = "baseline/lcp/processed/caltech101_val_3x3_pyg.pt",
                 transform=None,
                 ):
        
        random.seed(0)
        self.meta_info = json.load(open(meta_file))
        grid_folders = [ os.path.join(collage_dir, i) for i in os.listdir(collage_dir) if os.path.isdir(os.path.join(collage_dir, i))]
        
        self.grid_list = [ os.listdir(i)[0] for i in grid_folders]
        # random.shuffle(self.grid_list)
        self.num_classes = self.num_classes()
        # self.num_features = 512
        self.img_dir = img_dir
        self.transform = transform

        self.processed_path = processed_path

        if not osp.exists(self.processed_path):
            self.process_data()
        
        self.data, self.slices = torch.load(self.processed_path)
        # self.splits = self.get_idx_split()

    def num_classes(self) -> int:
        num_classes = 1
        return num_classes
    
    def len(self):
        return len(self.grid_list)
    

    def ord2edge(self, ord):
        edges_list = []
        for i in range(len(ord)):
            if (i % 3) != 2:
                st, ed = ord[i], ord[i + 1]
                edges_list.append([st, ed])
                edges_list.append([ed, st])

            if i < 6:
                st, ed = ord[i], ord[i + 3]
                edges_list.append([st, ed])
                edges_list.append([ed, st])
        
        return edges_list
        
    def get_sources(self, idx):
        img_names = [v['image'] for v in self.meta_info[self.grid_list[idx]].values()]
        return img_names

    def _get_(self, idx):

        img_names = [v['image'] for v in self.meta_info[self.grid_list[idx]].values()]
        features = [torch.tensor(np.load(os.path.join(self.img_dir, img.replace(".jpg", ".npy")))) for img in img_names]
        order = np.random.permutation(9)
        edge_list = self.ord2edge(order)
        pred = 0
        
        x = torch.squeeze(torch.stack(features, dim=0)).to(dtype=torch.float32)
        edge_index = torch.tensor(edge_list).t().contiguous()
        y = torch.tensor(pred, dtype=torch.float32).mean().unsqueeze(0)
        num_nodes = int(len(img_names))
        data = Data(x, edge_index, y=y, num_nodes=num_nodes)
        data.idx = torch.tensor([idx], dtype=torch.long)

        return data
    
    
    def process_data(self):
        data_list = []
        for i in tqdm(range(self.len())):
            data = self._get_(i)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_path)





class Imagenet2x2GridDataset(InMemoryDataset):
    def __init__(self, meta_file="dataset/train/metainfo/imagenet_train_2x2_collage_pred_info.pkl", 
                 img_dir="dataset/train/subset/feat", 
                 processed_path = "baseline/lcp/processed/imagenet_train_2x2_collage_pyg.pt",
                 transform=None,
                 ):
        
        random.seed(0)
        self.meta_info = pickle.load(open(meta_file, 'rb'))
        
        self.grid_list = list(self.meta_info.keys())
        
        self.num_classes = self.num_classes()
        # self.num_features = 512
        self.img_dir = img_dir
        self.transform = transform

        self.processed_path = processed_path

        if not osp.exists(self.processed_path):
            self.process_data()
        
        self.data, self.slices = torch.load(self.processed_path)
        # self.splits = self.get_idx_split()

    def num_classes(self) -> int:
        num_classes = 1
        return num_classes
    
    def len(self):
        return len(self.grid_list)
    

    def ord2edge(self, ord):
        edges_list = []
        for i in range(len(ord)):
            if i % 2 != 1:
                st, ed = ord[i], ord[i + 1]
                edges_list.append([st, ed])
                edges_list.append([ed, st])
        return edges_list
        
    def get_sources(self, idx):
        img_names = self.meta_info[self.grid_list[idx]]['ori']
        return img_names

    def _get_(self, idx):

        img_names = self.meta_info[self.grid_list[idx]]['ori']
        features = [torch.tensor(np.load(os.path.join(self.img_dir, img.replace(".JPEG", ".npy")))) for img in img_names]
        order = self.meta_info[self.grid_list[idx]]['ord']
        edge_list = self.ord2edge(order)
        pred = self.meta_info[self.grid_list[idx]]['pred']
        
        x = torch.squeeze(torch.stack(features, dim=0)).to(dtype=torch.float32)
        edge_index = torch.tensor(edge_list).t().contiguous()
        y = torch.tensor(pred, dtype=torch.float32).mean().unsqueeze(0)
        num_nodes = int(len(img_names))
        data = Data(x, edge_index, y=y, num_nodes=num_nodes)
        data.idx = torch.tensor([idx], dtype=torch.long)

        return data
    
    
    def process_data(self):
        data_list = []
        for i in tqdm(range(self.len())):
            data = self._get_(i)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_path)


 
class Imagenet2x2GridTestSet(InMemoryDataset):
    def __init__(self, meta_file="dataset/val/imagenet/metainfo/imagenet_val_2x2_collage_info.json", 
                 collage_dir="dataset/val/imagenet/collage/2x2",
                 img_dir="dataset/val/imagenet/set/feat", 
                 processed_path = "baseline/lcp/processed/imagenet_val_2x2_pyg.pt",
                 transform=None,
                 ):

        random.seed(0)
        self.meta_info = json.load(open(meta_file))
        grid_folders = [ os.path.join(collage_dir, i) for i in os.listdir(collage_dir) if os.path.isdir(os.path.join(collage_dir, i))]
        
        self.grid_list = [ os.listdir(i)[0] for i in grid_folders]
        # random.shuffle(self.grid_list)
        self.num_classes = self.num_classes()
        # self.num_features = 512
        self.img_dir = img_dir
        self.transform = transform

        self.processed_path = processed_path

        if not osp.exists(self.processed_path):
            self.process_data()
        
        self.data, self.slices = torch.load(self.processed_path)
        # self.splits = self.get_idx_split()

    def num_classes(self) -> int:
        num_classes = 1
        return num_classes
    
    def len(self):
        return len(self.grid_list)
    

    def ord2edge(self, ord):
        edges_list = []
        for i in range(len(ord)):
            if i % 2 != 1:
                st, ed = ord[i], ord[i + 1]
                edges_list.append([st, ed])
                edges_list.append([ed, st])
        return edges_list

        
    def get_sources(self, idx):
        img_names = [v['image'] for v in self.meta_info[self.grid_list[idx]].values()]
        return img_names

    def _get_(self, idx):

        img_names = [v['image'] for v in self.meta_info[self.grid_list[idx]].values()]
        features = [torch.tensor(np.load(os.path.join(self.img_dir, img.replace(".JPEG", ".npy")))) for img in img_names]
        order = np.random.permutation(4)
        edge_list = self.ord2edge(order)
        pred = 0
        
        x = torch.squeeze(torch.stack(features, dim=0)).to(dtype=torch.float32)
        edge_index = torch.tensor(edge_list).t().contiguous()
        y = torch.tensor(pred, dtype=torch.float32).mean().unsqueeze(0)
        num_nodes = int(len(img_names))
        data = Data(x, edge_index, y=y, num_nodes=num_nodes)
        data.idx = torch.tensor([idx], dtype=torch.long)

        return data
    
    
    def process_data(self):
        data_list = []
        for i in tqdm(range(self.len())):
            data = self._get_(i)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_path)

class Caltech2x2GridTestSet(InMemoryDataset):
    def __init__(self, meta_file="dataset/val/caltech101/metainfo/caltech101_test_2x2_collage_info.json", 
                 collage_dir="dataset/val/caltech101/collage/2x2",
                 img_dir="dataset/val/imagenet/set/feat", 
                 processed_path = "baseline/lcp/processed/caltech-101_2x2_pyg.pt",
                 transform=None,
                 ):
        # super(GridDataset, self).__init__(root=None, )

        random.seed(0)
        self.meta_info = json.load(open(meta_file))
        grid_folders = [ os.path.join(collage_dir, i) for i in os.listdir(collage_dir) if os.path.isdir(os.path.join(collage_dir, i))]
        
        self.grid_list = [ os.listdir(i)[0] for i in grid_folders]
        # random.shuffle(self.grid_list)
        self.num_classes = self.num_classes()
        # self.num_features = 512
        self.img_dir = img_dir
        self.transform = transform

        self.processed_path = processed_path

        if not osp.exists(self.processed_path):
            self.process_data()
        
        self.data, self.slices = torch.load(self.processed_path)
        # self.splits = self.get_idx_split()

    def num_classes(self) -> int:
        num_classes = 1
        return num_classes
    
    def len(self):
        return len(self.grid_list)
    

    def ord2edge(self, ord):
        edges_list = []
        for i in range(len(ord)):
            if i % 2 != 1:
                st, ed = ord[i], ord[i + 1]
                edges_list.append([st, ed])
                edges_list.append([ed, st])
        return edges_list

        
    def get_sources(self, idx):
        img_names = [v['image'] for v in self.meta_info[self.grid_list[idx]].values()]
        return img_names

    def _get_(self, idx):

        img_names = [v['image'] for v in self.meta_info[self.grid_list[idx]].values()]
        features = [torch.tensor(np.load(os.path.join(self.img_dir, img.replace(".jpg", ".npy")))) for img in img_names]
        order = np.random.permutation(4)
        edge_list = self.ord2edge(order)
        pred = 0
        
        x = torch.squeeze(torch.stack(features, dim=0)).to(dtype=torch.float32)
        edge_index = torch.tensor(edge_list).t().contiguous()
        y = torch.tensor(pred, dtype=torch.float32).mean().unsqueeze(0)
        num_nodes = int(len(img_names))
        data = Data(x, edge_index, y=y, num_nodes=num_nodes)
        data.idx = torch.tensor([idx], dtype=torch.long)

        return data
    
    
    def process_data(self):
        data_list = []
        for i in tqdm(range(self.len())):
            data = self._get_(i)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_path)


