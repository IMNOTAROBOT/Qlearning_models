import pandas as pd
import numpy as np
import math
import pickle
import random
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import datetime

#Regular model with no greediness and only 5 states of agent

#Model param
episodes = 10
min_profit = 2000
max_loss = -2000

#Q-learning param
alfa = 0.5   #learning rate
gamma = 0.6  #discount factor

#Number of rows used for training
random_pick = 10

#File data
day_file = '2018-02-26'
k_clust = 9
pca_dim = 2

#Loading table
data_file = pd.read_csv("C:/Users/karen/PycharmProjects/Qlearning_matrix_1/sim_files/CL_"+day_file+"_labeled_"+str(k_clust)+"_"+str(pca_dim)+".csv")

#Picking just the group column
data_class = pd.DataFrame(data_file['group'].copy())
data_class = data_class.astype(int)
print(data_class.shape)
lim = data_class.shape[0]
print(lim)

################################################
#Functions for managing state id

#Function to turn string of base to dec
def get_state(value, base):
    num = int(value, base)
    return num


#Function to get a string of the state according to values in columns
def get_string_state(row):
    word = ''
    for val in row:
        word = word + str(val)
    return word


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


################################################

#Functions for managing rewards and profits

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


#Simple reward function
def get_reward(init_state, next_state,step_profit_, base, init_price_, next_price_):
    max_step_profit = (next_price_ - init_price_) * 1000
    act_agent_state_ = get_agent_state(init_state, base)
    next_agent_state_ = get_agent_state(next_state, base)
    step_reward = 0
    if act_agent_state_ == next_agent_state_:
        if max_step_profit > 0 and (act_agent_state_ == 0 or act_agent_state_ == 1):
            val = round(max_step_profit / 3, 0)
            step_reward = step_reward - (abs(val) * 4)
        if max_step_profit < 0 and (act_agent_state_ == 0 or act_agent_state_ == 2):
            val = round(max_step_profit / 3, 0)
            step_reward = step_reward - (abs(val) * 4)
        # In the right state
        if max_step_profit > 0 and act_agent_state_ == 2:
            val = round(max_step_profit / 3, 0)
            step_reward = step_reward + (abs(val) * 4)
        if max_step_profit < 0 and act_agent_state_ == 1:
            val = round(max_step_profit / 3, 0)
            step_reward = step_reward + (abs(val) * 4)

    # If changing state with loss
    if act_agent_state_ != next_agent_state_:
        if step_profit_ > 0:
            val = round(abs(step_profit_) / 3, 0) * 5
            step_reward += val
        else:
            val = round(step_profit_ / 3, 0) * 5
            step_reward -= abs(val)

    # If getting to a ending state 3 or 4
    if next_agent_state_ == 4:
        val = 50
        step_reward += val
    if next_agent_state_ == 3:
        val = 50
        step_reward -= val
    return step_reward


#Function to define the next agent state. (This state is added to the world state to get the actual state)
#actions - 0:buy, 1:sell, 2:stay here, -1:invalid
def get_next_agent_state(init_state, action, total_profit, base, min_profit, max_loss):
    init_agent_state = get_agent_state(init_state, base)
    next_agent_state = -1
    if total_profit >= min_profit:
        return 4
    if total_profit <= max_loss:
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


#Function for select action at random (uniform distribution)
def get_action_uniform(agent_state, cum_profit_, min_profit_, max_loss_):
    if cum_profit_ > min_profit_ or cum_profit_ < max_loss_:
        if agent_state == 0:
            return 2
        if agent_state == 1:
            return 0
        if agent_state == 2:
            return 1
    val = np.random.multinomial(1, [0.3333, 0.3333, 0.3334], size=1)
    if agent_state == 1:
        val = np.random.multinomial(1, [0.5, 0.0, 0.5], size=1)
    if agent_state == 2:
        val = np.random.multinomial(1, [0.0, 0.5, 0.5], size=1)
    val = np.asarray(val).reshape(-1)
    val = np.nonzero(val)
    res = np.asarray(val).reshape(-1)
    return res[0]

############################################################

#Functions for managing states of system and q learning model

#Getting actual global state of system. Environment state + agent state
def get_actual_state(act_agent_state,  str_state, base):
    word_state = str_state + str(act_agent_state)
    state = get_state(word_state, base)
    return state


#Getting the max q value for next state.
def maxQ(next_state, matrix_q):
    matrix_val = np.asarray(matrix_q[next_state,:]).reshape(-1)
    return max(matrix_val)


############################################################

#Functions for running the model

# Function for running once the training model  (variables: list of names of columns used for model state)
def run_model(init_agent_state, base, alfa, gamma, matrix_q, data_class, j,min_profit, max_loss):
    n = j
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
        str_state = data_class.loc[n]['Str_State']
        act_state = get_actual_state(act_agent_state, str_state, base)

        #Selection action
        action = get_action_uniform(act_agent_state, total_profit,min_profit,max_loss)

        #print("state " + str(str_state) + " " + str(act_state) + " " + str(act_agent_state) + " " + str(action))

        #Getting next state
        next_n = n+1
        next_agent_state = get_next_agent_state(act_agent_state, action, total_profit, base, min_profit, max_loss)

        #print(next_agent_state)

        str_next_state = data_class.loc[next_n]['Str_State']
        next_state = get_actual_state(next_agent_state, str_next_state, base)

        #Getting profit
        act_price = data_class.loc[n]['Price']
        next_price = data_class.loc[next_n]['Price']
        act_profit = get_actualProfit(act_state, next_state, act_price, next_price, base)
        act_reward = get_reward(act_state, next_state,act_profit, base, act_price, next_price)
        #print("Reward: " + str(act_reward))
        total_profit = total_profit + act_profit
        #print("cum_profit: " + str(total_profit) + " step_profit: " + str(act_profit))
        #Qlearning func
        matrix_q[int(act_state),int(action)] = ((1-alfa) * matrix_q[int(act_state),int(action)]) + (alfa * (act_reward + (gamma * maxQ(next_state, matrix_q))))

        #updating index
        act_agent_state = next_agent_state
        n = n + 1

    return matrix_q

###########################################################
#Defining a reward matrix
# 3 possible actions per state.
#Agent states: 5

#Number of levels per class
base = k_clust
no_columns = 1
actions = 3
num_states = int(math.pow(base, no_columns + 1))
init_agent_state = 0

#Q Matriz
matrix_q = np.matlib.zeros((num_states, actions))

#Import the q-matrix
#day_ = '2018-02-26'
#filehandler = open('sim_files/q_matrix_CL_'+day_+'_'+str(k_clust)+'_'+str(pca_dim)+'.obj', 'rb')
#matrix_q = pickle.load(filehandler)

###########################################################

#Getting the state in string and in dec number
words_states = data_class.apply(lambda dic: get_string_state(dic), axis=1)
states = words_states.apply(lambda dic: get_state(dic, base))

#Adding the states to class_data DF
data_class["Str_State"] = words_states
data_class["State"] = states


data_class['Price'] = data_file['Price']

#print(data_class.shape)
#print(data_class.head(10))
#data_class.to_csv('sim_files/CL_'+day_file+'_states_'+str(k_clust)+'.csv', index = False, sep=',', encoding='utf-8')

#################################################################

#Functions for running the training model and saving the q-matrix in a file

#matrix_q = run_model(init_agent_state ,base, alfa, gamma, matrix_q, data_class[0:lim], 0,min_profit, max_loss)

for j in range(episodes):
    print("Episode: " + str(j))
    sweeping = np.asarray(random.sample(range(0, lim), random_pick))
    #print(sweeping)
    for i in range(random_pick):
        matrix_q = run_model(init_agent_state, base, alfa, gamma, matrix_q, data_class[0:lim], sweeping[i], min_profit, max_loss)


###################################################
#Set forbidden actions in matrix and bised decision when unknown
total_states_wnas = int(math.pow(base,no_columns))
for i in range(total_states_wnas):
    state_en = numberToBase(i, base)
    state_env = get_string_state(state_en)
    state_tot1 = get_actual_state(1,state_env, base)
    matrix_q[int(state_tot1),1] = -99999.0
    state_tot2 = get_actual_state(2, state_env, base)
    matrix_q[int(state_tot2), 0] = -99999.0



###################################################

file_py = open('sim_files/q_matrix_CL_'+day_file+'_'+str(k_clust)+'_'+str(pca_dim)+'.obj', 'wb')
pickle.dump(matrix_q, file_py)

print(len(matrix_q))

matrix_q_frame = pd.DataFrame(matrix_q)
matrix_q_frame.to_csv('sim_files/q_matrix_CL_'+day_file+'_'+str(k_clust)+'_'+str(pca_dim)+'.csv', index = False, sep=',', encoding='utf-8')