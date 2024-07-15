import argparse
import glob
import os
import os.path as osp
import time
from tqdm import tqdm
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, InMemoryDataset, Data
from util.utils import process_adj
import numpy as np
import pygad
from util.utils import ord_grid_adj
from util.data_loader import  ImagenetGridDataset

 
class Bench3x3GridTestSet(InMemoryDataset):
    def __init__(self, meta_file="dataset/val/caltech101/metainfo/caltech101_test_3x3_collage_info.json", 
                 collage_dir="dataset/val/ucf101/collage/3x3",
                 img_dir="dataset/val/ucf101/set/feat", 
                 processed_path = "baseline/lcp/processed/caltech-101_3x3_pyg.pt",
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


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_adj', type=float, default=0.1, help='learning rate for adj')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=False, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=False, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=500, help='patience for early stopping')
parser.add_argument('--save_path', type=str, default='weights/weights_train', help='Save Checkpoints')


parser.add_argument('--save_pred_path', type=str, default='results/ucf101_3x3_pred_ga.json', help='Save json')


args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = ImagenetGridDataset()
testset = Bench3x3GridTestSet(
                 meta_file="dataset/val/ucf101/metainfo/ucf101_test_3x3_collage_info.json", 
                 collage_dir="dataset/val/ucf101/collage/3x3",
                 img_dir="dataset/val/ucf101/set/feat", 
                 processed_path = "baseline/lcp/processed/ucf101_3x3_pyg.pt")

args.num_classes = dataset.num_classes
args.num_features = 512

print(args)

# Normalize targets to mean = 0 and std = 1.
mean = dataset.data.y.mean().item()
std = dataset.data.y.std().item()
dataset.data.y = (dataset.data.y - mean) / std





num_training = int(len(dataset.grid_list) * 0.9)
num_val = len(dataset.grid_list) - num_training
# num_test = len(dataset.grid_list) - (num_training + num_val)
training_set, validation_set = random_split(dataset, [num_training, num_val])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)


model = Model(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_all = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            

        loss_train = loss_all
        val_error = compute_test(val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'Validation MAE: {:.6f}'.format(val_error), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(val_error)
        torch.save(model.state_dict(), os.path.join(args.save_path, '{}.pth'.format(epoch)))

        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

    print("Best Epoch: {:04d}".format(best_epoch))

    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def compute_test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(args.device)
        out = model(data)
        
        error += (out * std - data.y * std).abs().sum().item()  # MAE
    return  error / len(loader.dataset)


def create_population(num_solutions):
    population_ord = []

    for idx in range(num_solutions):
        ord_rand = np.random.permutation(9)
        population_ord.append(ord_rand)

    return population_ord




def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

def ga_approximate_adj(model, loader):

    for param in model.parameters():
        param.requires_grad = False

    num_generations = 3 # Number of generations.
    num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
    num_solutions = 20
    initial_population = create_population(num_solutions) # Initial population

    test_pred_dict = {}
    
    
    for i, batch in tqdm(enumerate(loader)):
        start_time = time.time()
        best_score = 0.0
        
        def fitness_func_wrapper(ga_instanse, solution, solution_idx):

            def fitness_func(ga_instanse, solution, sol_idx, *args):
                global model
                solution_adj = ord_grid_adj(solution)
                
                batch.edge_index, _ = process_adj(solution_adj)
                
                out = model(batch.to('cuda'))
                pred = out * std
                loss_mae = F.l1_loss(pred, torch.ones_like(pred))

                abs_error = loss_mae + 0.00000001

                solution_fitness = 1.0 / abs_error

                return solution_fitness.cpu().numpy()
            
            args = batch
            fitness = fitness_func(ga_instanse, solution, solution_idx, *args)
            return fitness
        
        ga_instance = pygad.GA(num_generations=num_generations, 
                            num_parents_mating=num_parents_mating, 
                            initial_population=initial_population,
                            gene_space=range(0, 9),
                            gene_type=int,
                            fitness_func=fitness_func_wrapper,
                            on_generation=on_generation,
                            allow_duplicate_genes=False,
                            stop_criteria=["saturate_10"])

        ga_instance.run()


        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")
        print(f"Time taken = {time.time() - start_time} seconds")
        # print progress
        num_of_batch = len(loader)
        print(f"Progress: {i}/{num_of_batch}")

        idx = int(batch.idx.item())
        batch_source = testset.get_sources(idx)
        ord_indices = solution
        image_list = [batch_source[i] for i in ord_indices]
        

        solution_adj = ord_grid_adj(solution)
        batch.edge_index, batch.edge_attr = process_adj(solution_adj)
        best_score = model(batch.to('cuda'))  * std

        test_pred_dict[idx] = {
            'image_list': image_list,
            
            'pred': best_score.item()
        }
        

    
    with open(args.save_pred_path, 'w') as f:
        json.dump(test_pred_dict, f)

if __name__ == '__main__':
    
    model.load_state_dict(torch.load("baseline/lcp/weights/weights_train_3x3/499.pth", map_location=args.device))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    ga_approximate_adj(model, test_loader)
