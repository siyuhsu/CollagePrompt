import argparse

import os
import time
from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
from models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
import numpy as np
import pygad
from util.utils import process_adj, ord_grid_adj_2x2
from util.data_loader import  Imagenet2x2GridDataset, Imagenet2x2GridTestSet


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
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=500, help='patience for early stopping')
parser.add_argument('--save_path', type=str, default='./weights_train_2x2', help='Save Checkpoints')
parser.add_argument('--save_pred_path', type=str, default='./preds/imagenet_val_2x2_cases.json', help='Save json')


args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = Imagenet2x2GridDataset()
testset = Imagenet2x2GridTestSet()

args.num_classes = dataset.num_classes
args.num_features = 512

print(args)


mean = dataset.data.y.mean().item()
std = dataset.data.y.std().item()
dataset.data.y = (dataset.data.y - mean) / std





num_training = int(len(dataset.grid_list) * 0.9)
num_val = len(dataset.grid_list) - num_training

training_set, validation_set = random_split(dataset, [num_training, num_val])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=1, shuffle=True)
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


def create_population_2x2(num_solutions):
    population_ord = []

    for idx in range(num_solutions):
        ord_rand = np.random.permutation(4)
        population_ord.append(ord_rand)

    return population_ord


def brute_force_adj(model, loader):

    for param in model.parameters():
        param.requires_grad = False



    test_pred_dict = {}
    
    total_time = 0

    for i, batch in tqdm(enumerate(loader)):
        if i == 1000:
            break

        start_time = time.time()
        best_score = 0.0
        num_solutions = 25
        initial_population = create_population_2x2(num_solutions) # Initial population
        
        def fitness_func_wrapper(solutions):

            def fitness_func(solution, *args):
                global model
                solution_adj = ord_grid_adj_2x2(solution)

                batch.edge_index, _ = process_adj(solution_adj)

                out = model(batch.to('cuda'))
                pred = out * std
                loss_mae = F.l1_loss(pred, torch.ones_like(pred))

                abs_error = loss_mae + 0.00000001

                solution_fitness = 1.0 / abs_error

                return solution_fitness.cpu().numpy()
            
            args = batch
            best_fitness = -1
            for idx, solution in enumerate(solutions):
                fitness = fitness_func(solution, *args)
                if fitness > best_fitness:
                    best_fitness = fitness
                    solution_idx = idx
                    best_solution = solution
            return best_solution, best_fitness, solution_idx 
      


        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = fitness_func_wrapper(initial_population)
        time_cost = time.time() - start_time
        total_time += time_cost
        average_time = total_time / (i + 1)

        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")
        print(f"Time taken = {time_cost} seconds")
        print(f"Average time taken = {average_time} seconds")

        num_of_batch = len(loader)
        print(f"Progress: {i}/{num_of_batch}")

        idx = int(batch.idx.item())
        batch_source = testset.get_sources(idx)
        ord_indices = solution
        image_list = [batch_source[i] for i in ord_indices]
        

        solution_adj = ord_grid_adj_2x2(solution)
        batch.edge_index, batch.edge_attr = process_adj(solution_adj)
        best_score = model(batch.to('cuda'))  * std

        test_pred_dict[idx] = {
            'image_list': image_list,
            'pred': best_score.item()
        }

    
    with open(args.save_pred_path, 'w') as f:
        json.dump(test_pred_dict, f)



def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Best solution  = {ga_instance.best_solution()[0]}")
    print(f"Best solution fitness = {ga_instance.best_solution()[1]}")


def ga_approximate_adj(model, loader):

    for param in model.parameters():
        param.requires_grad = False

    num_generations = 5 # Number of generations.
    num_parents_mating = 3 # Number of solutions to be selected as parents in the mating pool.
    num_solutions = 5
    initial_population = create_population_2x2(num_solutions) # Initial population
    print("Initial Population:")
    print(initial_population)

    test_pred_dict = {}
    
    total_time = 0
    total_generation = 0
    
    for i, batch in tqdm(enumerate(loader)):
        if i == 5:
            break
        
        start_time = time.time()
        best_score = 0.0
        
        def fitness_func_wrapper(ga_instanse, solution, solution_idx):

            def fitness_func(ga_instanse, solution, sol_idx, *args):
                global model
                solution_adj = ord_grid_adj_2x2(solution)
                
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
                            crossover_type="uniform",
                            gene_space=range(0, 4),
                            gene_type=int,
                            fitness_func=fitness_func_wrapper,
                            on_generation=on_generation,
                            allow_duplicate_genes=False,
                            save_solutions=True
                            ,save_best_solutions=True,
                            stop_criteria=["saturate_3"])

        ga_instance.run()


        # Returning the details of the best solution.
        all_solutions = ga_instance.best_solutions
        all_solutions_fitness = ga_instance.best_solutions_fitness

        generation_completed = ga_instance.generations_completed   
        total_generation += generation_completed
        average_generation = total_generation / (i + 1)

        time_cost = time.time() - start_time
        total_time += time_cost
        average_time = total_time / (i + 1)
        
        print(f"Time taken = {time_cost} seconds")
        print(f"Average time taken = {average_time} seconds")
        print(f"Average Generation completed = {average_generation}")
        # print progress
        num_of_batch = len(loader)
        print(f"Progress: {i}/{num_of_batch}")

        idx = int(batch.idx.item())
        bi = 0
        batch_dict = {}
        for sol, fit in zip(all_solutions, all_solutions_fitness):
            batch_source = testset.get_sources(idx)
            ord_indices = sol
            image_list = [batch_source[i] for i in ord_indices]
            
            sid = str(idx)+f"_{bi}"
            batch_dict[sid] = {
                
                'image_list': image_list,
                
                'solution': [int(i) for i in sol],
                'fitness': float(fit)
            }
            bi +=1
        test_pred_dict[idx] = batch_dict
        

    
    with open(args.save_pred_path, 'w') as f:
        json.dump(test_pred_dict, f, indent=4)

def ga_approximate_adj_save_best(model, loader):

    for param in model.parameters():
        param.requires_grad = False

    num_generations = 5 # Number of generations.
    num_parents_mating = 3 # Number of solutions to be selected as parents in the mating pool.
    num_solutions = 5
    initial_population = create_population_2x2(num_solutions) # Initial population
    print("Initial Population:")
    print(initial_population)

    test_pred_dict = {}
    
    total_time = 0
    total_generation = 0
    
    for i, batch in tqdm(enumerate(loader)):
        if i == 1000:
            break
        
        start_time = time.time()
        best_score = 0.0
        
        def fitness_func_wrapper(ga_instanse, solution, solution_idx):

            def fitness_func(ga_instanse, solution, sol_idx, *args):
                global model
                solution_adj = ord_grid_adj_2x2(solution)
                
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
                            crossover_type="uniform",
                            crossover_probability=0.5,
                            mutation_probability=0.3,
                            gene_space=range(0, 4),
                            gene_type=int,
                            fitness_func=fitness_func_wrapper,
                            on_generation=on_generation,
                            allow_duplicate_genes=False,
                            save_solutions=True
                            ,save_best_solutions=True,
                            stop_criteria=["saturate_3"])


        ga_instance.run()


        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        all_solutions = ga_instance.best_solutions
        all_solutions_fitness = ga_instance.best_solutions_fitness

        generation_completed = ga_instance.generations_completed   
        total_generation += generation_completed
        average_generation = total_generation / (i + 1)

        time_cost = time.time() - start_time
        total_time += time_cost
        average_time = total_time / (i + 1)
        
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")
        print(f"Time taken = {time_cost} seconds")
        print(f"Average time taken = {average_time} seconds")
        print(f"Average Generation completed = {average_generation}")
        # print progress
        num_of_batch = len(loader)
        print(f"Progress: {i}/{num_of_batch}")

        idx = int(batch.idx.item())
        batch_source = testset.get_sources(idx)
        
        ord_indices = solution
        image_list = [batch_source[i] for i in ord_indices]
        # y = batch.y.item()

        solution_adj = ord_grid_adj_2x2(solution)
        batch.edge_index, batch.edge_attr = process_adj(solution_adj)
        best_score = model(batch.to('cuda'))  * std

        test_pred_dict[idx] = {
            'image_list': image_list,
            # 'y': y,
            'pred': best_score.item()
        }
        # break

    
    with open(args.save_pred_path, 'w') as f:
        json.dump(test_pred_dict, f)

if __name__ == '__main__':
    # Model training
    best_model = train()
    
    
    # model.load_state_dict(torch.load("./weights_train_2x2/410.pth", map_location=args.device))
    
    # ga_approximate_adj(model, test_loader)
    # # brute_force_adj(model, test_loader)
