#Function to generate models for training in module 7
import random
from itertools import combinations as comb

random.seed(42)

def generator(q_lst: list, num_DASS: int, models_to_train: int):
    # lst: full list of questions to choose from
    # models_to_train: total number of models
    # num_dass: number of questions to choose from lst
    
    #Initialize counter
    question_counter = {}    
    for i in q_lst:
        question_counter[i] = 0
        
    #Randomly select the first set of questions
    all_comb = list(comb(q_lst, num_DASS))
    index = random.randint(0, len(all_comb)-1)
    models_set = [all_comb[index]]
    
    #Increase counter
    for models in models_set:
        for questions in models:
            question_counter[questions] += 1
    max_count = 1
    
    while len(models_set) < models_to_train:
        new_model = []
        while len(new_model) < num_DASS:
            min_count_questions = find_minimum(question_counter, max_count)
            
            index = random.randint(0, len(min_count_questions)-1)
            # Reselect if the new question is already in the model
            while min_count_questions[index] in new_model:
                index = random.randint(0, len(min_count_questions)-1)
            question_to_add = min_count_questions[index]
            
            new_model.append(question_to_add)
            question_counter[question_to_add] += 1
            
            max_count = update_max_count(question_counter)
        
        models_set.append(new_model)
        
    sorted_models = []
    for models in models_set:
        sorted_models.append(sorted(models))
    
    return sorted_models


def find_minimum(question_counter: dict, max_count: int):
    #Find the list of questions with lowest counter
    min_count_questions = []
    for question, counter in question_counter.items():
        if counter < max_count:
            min_count_questions.append(question)
            
    return min_count_questions

def update_max_count(question_counter: dict):
    #Update the maximum counter
    values = question_counter.values()
    max_count = max(values)
    if max(values) == min(values):
        max_count += 1      
        
    return max_count
    