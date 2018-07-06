import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Script for analysing the data used for the env representation. Generates a cluster of env states with K - means algorithm
n_klus = 9
pca_red = 2

#Picking the variable used for the model
columns = ['TotalMarketDepth','BidAskRatio','NP10','NC4','FC10','FP1']

#For one day
init_day = '2018-02-26'
end_day = '2018-02-27'

######################################################################################################

#Reading file with data
data_table_all = pd.read_csv("C:/Users/karen/PycharmProjects/Qlearning_matrix_1/data_files/CL_2018022_20180302.csv")
data_table_all = data_table_all.fillna(method='ffill')
data_table_all['DateTimeStamp'] = pd.to_datetime(data_table_all['DateTimeStamp'], format='%Y-%m-%d %H:%M:%S')

#####################################################################################################

start_date = pd.to_datetime(init_day + ' 00:00:00.000', format='%Y-%m-%d %H:%M:%S')
end_date = pd.to_datetime(end_day + ' 00:00:00.000', format='%Y-%m-%d %H:%M:%S')

#Function for getting tuples of certain period of time
def get_frame_time(frame, startdate, enddate):
    mask = (frame['DateTimeStamp'] > startdate) & (frame['DateTimeStamp'] <= enddate)
    return frame.loc[mask]

#Getting just tuples for a certain period
data_table = get_frame_time(data_table_all, start_date,end_date)

print(data_table.shape)

#####################################################################################################

#Getting logReturn value
def get_logReturn(grp):
    price_t0 = grp[0]
    price_t1 = grp[-1]
    diff = price_t1 / price_t0
    return math.log(diff)


#Generating logReturn
roll = '15S'
data_table['LogReturn_'+roll]= data_table.rolling(roll,on='DateTimeStamp')['Price'].apply(lambda x : get_logReturn(x))
data_table = data_table.reset_index()

#####################################################################################################

#Preparing the data for normalization and scaling

data_x = data_table[columns]
y = data_table['Price']

#Scaling and normalizing
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_x)
df_normalized = pd.DataFrame(np_scaled, columns=columns)

##########################################################################

#Some plots for showing normality of data
#for i in range(len(columns)):
#    if columns[i] != 'LogReturn_10S':
#        pd.plotting.scatter_matrix(df_normalized.ix[:, ['LogReturn_10S',columns[i]]], alpha = 0.3, figsize = (14,8), diagonal = 'kde');
#        plt.show()


##########################################################################

#PCA tranformation to only two dimensions
print("PCA transformation ...")

#convert it to numpy arrays
X=df_normalized.values

pca = PCA(n_components=pca_red)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)


#Grouping variables k-means to generate the states
print("K-means clustering...")

kmeans = KMeans(n_clusters= n_klus, random_state=0, max_iter=100).fit(X_pca)
print(kmeans.cluster_centers_)

#Save to a file
file_py = open('sim_files/kmeans_cluster_'+init_day+'_'+str(n_klus)+'_'+str(pca_red)+'.obj', 'wb')
pickle.dump(kmeans, file_py)

labels = list(kmeans.labels_)
labels = np.asarray(labels).reshape(len(labels),1)
#print(labels)

#Plot clusters

plt.plot()
plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_)
plt.title("State clusters")
plt.show()

##Adding the cluster group for states and price reshaping data

labeled_data = np.append(X_pca,labels, axis=1)

df_labeled_data = pd.DataFrame(data=labeled_data, columns=['dim1','dim2','group'])

df_labeled_data['Price'] = y
df_labeled_data['LogReturn_'+roll] = data_table['LogReturn_'+roll]

for column in columns:
    df_labeled_data[column] = data_table[column]

pd.plotting.scatter_matrix(df_labeled_data.ix[:, ['LogReturn_'+roll,'group']], alpha = 0.3, figsize = (14,8), diagonal = 'kde');
plt.show()

df_labeled_data.to_csv('sim_files/CL_'+init_day+'_labeled_'+str(n_klus)+'_'+str(pca_red)+'.csv', index = False, sep=',', encoding='utf-8')
