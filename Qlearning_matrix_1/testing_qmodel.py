import pandas as pd
import numpy as np
import pickle
import random

#5 states agent testing script

seed = 5714
np.random.seed(seed)

#Experiment values
levels = 9
pca_dim = 2

random_pick = 20

min_profit = 200
max_lose = -200

#files data
day_matrix = '2018-02-26'
day_file = '2018-02-26'

#Import the q-matrix
filehandler = open('sim_files/q_matrix_CL_'+day_matrix+'_'+str(levels)+'_'+str(pca_dim)+'.obj', 'rb')
matrix_q = pickle.load(filehandler)

#Loading data
data_class = pd.read_csv("C:/Users/karen/PycharmProjects/Qlearning_matrix_1/sim_files/CL_"+day_file+"_class_"+str(levels)+"_"+str(pca_dim)+"_"+day_matrix+".csv")
data_class = data_class.fillna(method='ffill')
print(data_class.shape)
lim_min = 0
lim_max = data_class.shape[0]
print(lim_max)

###############################################################

#States functions

#Function to turn string of base to dec
def get_state(value, base):
    num = int(value, base)
    return num


#Function for turning a dec into another base
def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n != 0:
        digits.append(int(n % b))
        n = int(n/b)
    return digits[::-1]


# Getting the agent state. 0: flat, 1:short, 2:long, 3:fail, 4:HALT
def get_agent_state(act_state, base):
    digits = numberToBase(act_state, base)
    return digits[-1]


#Getting actual global state of system. Environment state + agent state
def get_actual_state(act_agent_state,  str_state, base):
    word_state = str(str_state) + str(act_agent_state)
    state = get_state(word_state, base)
    return state


#Function to define the next agent state. (This state is added to the world state to get the actual state)
#actions - 0:buy, 1:sell, 2:stay here, -1:invalid
def get_next_agent_state(init_state, action, total_profit, base, min_profit, max_lose):
    init_agent_state = get_agent_state(init_state, base)
    next_agent_state = -1
    if total_profit >= min_profit:
        return 4
    if total_profit <= max_lose:
        return 3
    #Transitions for FLAT
    if init_agent_state == 0 and action == 0:
        next_agent_state = 2
    if init_agent_state == 0 and action == 1:
        next_agent_state = 1
    if init_agent_state == 0 and action == 2:
        next_agent_state = 0
    #Transitions for Short
    if init_agent_state == 1 and action == 0:
        next_agent_state = 0
    if init_agent_state == 1 and action == 1:
        next_agent_state = -1
    if init_agent_state == 1 and action == 2:
        next_agent_state = 1
    #Transitions for Long
    if init_agent_state == 2 and action == 0:
        next_agent_state = -1
    if init_agent_state == 2 and action == 1:
        next_agent_state = 0
    if init_agent_state == 2 and action == 2:
        next_agent_state = 2
    return next_agent_state


###############################################################

#Functions for decision making, profit and reward

#Function for taking a decision based in the actual state
def make_decision(act_state, q_matrix):
    matrix_val = np.asarray(q_matrix[act_state, :]).reshape(-1)
    max_val = max(matrix_val)
    max_pos = [i for i, j in enumerate(matrix_val) if j == max_val]
    if len(max_pos) > 1:
        act_agent_state = get_agent_state(act_state, base)
        if act_agent_state == 0:
            return 2
        if act_agent_state == 1:
            return 2
        if act_agent_state == 2:
            return 2
    return max_pos[0]


#Get actual profit in $
def get_actualProfit(ini_state, next_state, ini_price, end_price, base):
    cost = 0
    ini_agent_state = get_agent_state(ini_state, base)
    next_agent_state = get_agent_state(next_state, base)

    if ini_agent_state == 0 and next_agent_state == 2:
        cost = ((end_price - ini_price) * 1000) - 3.03
    if ini_agent_state == 0 and next_agent_state == 1:
        cost = ((ini_price - end_price) * 1000) - 3.03
    if ini_agent_state == 1 and next_agent_state == 1:
        cost = ((ini_price - end_price) * 1000)
    if ini_agent_state == 2 and next_agent_state == 2:
        cost = ((end_price - ini_price) * 1000)
    if ini_agent_state == 1 and (next_agent_state == 0 or next_agent_state == 3 or next_agent_state == 4):
        cost = ((ini_price - end_price) * 1000) - 3.03
    if ini_agent_state == 2 and (next_agent_state == 0 or next_agent_state == 3 or next_agent_state == 4):
        cost = ((end_price - ini_price) * 1000) - 3.03
    return cost


###################################################################

#Functions for running simulation

# Function for running once the training model  (variables: list of names of columns used for model state)
def run_simulation(init_agent_state, base, matrix_q, data_class, init_row, min_profit, max_lose, bag):
    n = init_row
    act_agent_state = init_agent_state
    next_agent_state = -1
    act_state = -1
    next_state = -1
    act_price = -1
    next_price = -1

    action = -1

    total_profit = 0

    while (act_agent_state != 4 and act_agent_state != 3) and n <= (data_class.shape[0]-2):
        #Getting actual state
        str_state = int(data_class.loc[n]['group'])
        act_state = get_actual_state(act_agent_state, str_state, base)

        #Selection action
        action = make_decision(act_state, matrix_q)

        #Getting next state
        next_n = n+1
        next_agent_state = get_next_agent_state(act_agent_state, action, total_profit, base, min_profit, max_lose)
        str_next_state = int(data_class.loc[next_n]['group'])
        #print(str(act_state) + " " + str(act_agent_state) + " " + str(next_agent_state))
        next_state = get_actual_state(next_agent_state, str_next_state, base)

        #Getting profit
        act_price = data_class.loc[n]['Price']
        next_price = data_class.loc[next_n]['Price']
        act_profit = get_actualProfit(act_state, next_state, act_price, next_price, base)

        total_profit = total_profit + act_profit

        #Data in bag
        bag.append([act_agent_state, next_agent_state, act_state, next_state, action, act_price, next_price, act_profit, total_profit, str_state, str_next_state])
        #print(bag)
        #updating index
        act_agent_state = next_agent_state
        n = n + 1
    return total_profit


##################################################################

#Param for simulation function
base = levels
init_agent_state = 0

episode_profits = []

sweeping = np.asarray(random.sample(range(lim_min, lim_max), random_pick))
#print(sweeping)
for i in range(random_pick):
    env_states = []
    total_profit = run_simulation(init_agent_state, base, matrix_q, data_class, sweeping[i],min_profit, max_lose, env_states)
    episode_profits.append(total_profit)
    # turning env_state in a table
    bag_obs = pd.DataFrame(env_states,
                           columns=['act_agent_state', 'next_agent_state', 'act_state', 'next_state', 'action',
                                    'act_price', 'next_price', 'act_profit', 'total_profit', 'str_act_state',
                                    'str_next_state'])
    # Safe experiment results in file
    bag_obs.to_csv('sim_results/CL_'+day_file+'_sim_'+str(i)+'_'+str(levels)+'_'+str(pca_dim)+'_'+day_matrix+'.csv', index=False, sep=',', encoding='utf-8')
    print(str(i)+' : '+str(total_profit) + ' file init: ' + str(sweeping[i]))


df_profits = pd.DataFrame({'profits':np.asarray(episode_profits)})
#print(df_profits)
res = np.asarray(episode_profits)
tot_pos = res[np.where(res >0)]
tot_zero = res[np.where(res == 0)]
print("+Profit: " + str(len(tot_pos)) + '  Zero Profit: ' + str(len(tot_zero)))
print(df_profits.describe())