import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Script used to classify data of other days with the info of the cluster of certain day using Kmeans
n_klus = 9
pca_dim = 2
columns = ['TotalMarketDepth','BidAskRatio','NP10','NC4','FC10','FP1']

#For one week
init_day = '2018-02-26'
end_day = '2018-02-27'

cluster_day = '2018-02-26'

#####################################################################################################

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
print(data_table.head(10))
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

####################################################################################################

#Picking the variable used for the model
y = data_table['Price']
data_x = data_table[columns]

#Transforming the data
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_x)
df_normalized = pd.DataFrame(np_scaled, columns=columns)

#PCA tranformation to only two dimensions
#convert it to numpy arrays
print("PCA transformation ...")

X=df_normalized.values

pca = PCA(n_components=pca_dim)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

#Grouping variables k-means to generate the states
print("K-means clustering...")

#Load cluster from file
filehandler = open('sim_files/kmeans_cluster_'+cluster_day+'_'+str(n_klus)+'_'+str(pca_dim)+'.obj', 'rb')
kmeans = pickle.load(filehandler)

labels_predict = kmeans.predict(X_pca)
labels_list = list(labels_predict)
labels = np.asarray(labels_list).reshape(len(labels_list),1)
#print(labels)

#Plot clusters
plt.plot()
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_predict)
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

df_labeled_data.to_csv('sim_files/CL_'+init_day+'_class_'+str(n_klus)+'_'+str(pca_dim)+'_'+cluster_day+'.csv', index = False, sep=',', encoding='utf-8')
