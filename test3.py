import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import math
import matplotlib.pyplot as plt
import datetime 

#bale sto data to arxeio train (h opoio allo arxeio nomizoume)
train_data = pd.read_csv("train.csv")
feat_data = pd.read_csv("features.csv", sep=";")

""" print (train_data.head())
print (feat_data.head()) """

print("info")
print(train_data.info)
print(feat_data.info)

#elegekse ton typo twn dedomenwn ths kathe sthlhs
print("printing data types")
print(train_data.dtypes) 
print(feat_data.dtypes)

print("datacolumns")
print(train_data.columns)
print(feat_data.columns)

#tsekare an loipoyn kapoies times
print("Checking for missing values")
print(train_data.isnull().sum())
print("Checking for missing values features")
print(feat_data.isnull().sum())


#to unemployment exei missing values
#bazoume ton meso oro sta pedia missing
feat_data['Unemployment'] = feat_data['Unemployment'].fillna(np.mean(feat_data['Unemployment'] ))

print("Checking for missing values features")
print(feat_data.isnull().sum())

#diegrapse to Date pedio apo to arxeio features 
#Ta date formats se kathe arxeio einai diaforetika opote tha kratisoume 
#Ta Dates apo to arxxeio train poy einai kai perissotera


print(feat_data.head())

feat_data['Date'] = pd.to_datetime(feat_data.Date)

print(feat_data.head())
#print(feat_data.Date.dt.month)
print(feat_data.dtypes)
print("end")







# Merge feature and training data
""" new_df = pd.merge(feat_data, train_data, on = ['Store', 'IsHoliday'], how = 'inner')



print("Checking for missing values new_df")
print(new_df.isnull().sum())
print("printing merged data")
print(new_df.shape)
print(new_df.head()) """











#Set index to the data column
""" data.index = pd.to_datetime(data['Date'])
print (data.head())

#Diegrapse th deyterh sthlh date 
data  = data.drop(['Date'], axis = 1)
print (data.head()) """





#den fainetai na mas leipoun kapoia values
#logika ayto tha allaksei otan kanoyme merge kai to features.csv

#orise tis x,y metablhtes (y ayto poy theloume na problepsoume)



""" plt.plot(x, y, 'x', color='blue', alpha=0.5)
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

 """




""" predict = "Weekly_Sales"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
 
print (X)
print("tell me y!")
print (y)

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

print("teell me y!!")
print (y_test) """