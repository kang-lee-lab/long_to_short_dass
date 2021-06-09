from itertools import permutations, combinations
import os
import pandas as pd
import random
import time

data_folder = "./data"

def models_combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    result = []
    hm_score = 0
    start_time = time.time()
    
    for indices in permutations(range(n), r):
        if (time.time() - start_time >= 300):
            break
        if sorted(indices) == list(indices):
            models = tuple(pool[i] for i in indices)
            if hm_scores(models) == hm_score:
                result.append(models)
                if len(result) >= 2000:
                    break
                
            elif hm_scores(models) > hm_score:
                result = [models]
                hm_score = hm_scores(models)
    return result
        
def hamming_distance(lst1, lst2):
    s1 = set(lst1)
    s2 = set(lst2)
    
    hm_dist = len(s1) - len(s1.intersection(s2))
    return hm_dist

def hm_scores(models):
    hm = 0
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            hm += hamming_distance(models[i], models[j])
    return hm

def generate_comb(lst: list, models_to_train: int, num_dass: int):
    # lst: full list of questions to choose from
    # models_to_train: total number of models
    # num_dass: number of questions to choose from lst
    
    comb = list(combinations(lst, num_dass))
    
    models = list(models_combinations(comb, models_to_train))
    
    print(len(models))

    return models


question_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
questions = [20, 9, 30, 11, 19, 2, 36, 28, 4, 23, 7, 27, 1, 18, 40]
models_to_train = 10 
random.seed(42)
output = {}

for i in question_numbers:
    models_combs = generate_comb(questions, models_to_train, i)
    out_model = random.randint(0, len(models_combs)-1)
    output['{0} DASS Questions'.format(i)] = models_combs[out_model]

df = pd.DataFrame(output)

df.to_csv(os.path.join(data_folder, "models_for_training.csv"), index=None)
    
    
    
    
    
