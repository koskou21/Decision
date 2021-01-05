import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


#kanoume eisagwgh twn fakelwn twn dedomenwn mas
train = pd.read_csv('train.csv')
features = pd.read_csv('features.csv')

#emfanizoume ta dedomena mas 
print('Features : ', features.shape)
print('Train    : ', train.shape)

#analuoume ksexwrista ta dedomena mas pairnontas plirofories gia to periexomeno tous
features.info()
train.info()

#elegxos kenwn metavlhtwn sta dedomena mas
features.isnull().sum()

#symplirwsi kenwn metavlhtwn sta dedomena mas me ton antistoixo meso oro ths metavlhths opou uparxoun ta kena
#features['Unemployment'] = features['Unemployment'].fillna(np.mean(features['Unemployment'] ))

#elegxos kai epalitheusi gia mh mhdenika stoixeia
features.isnull().sum() 

#sygxwneush twn duo mhtrwwn kai emfanisi tou eniaiou
print(features.columns)
print(features.shape)

print(train.columns)
print(train.shape)
#print(new_df.shape)

#new_df.head()
#epituxhs sugxwneush dedomenwn

#taksinomisi vash hmeromhnias se periptwsh mh taksinomimenou mitrwou
#df = df.sort_values(by = 'Date')
#df.head(arithmos grammwn)

#synoliko athroisma pwlhsewn se kairo diakopwn kai oxi
is_holiday = train[train.IsHoliday == True]
print('Sales on Holidays     : ', is_holiday['Weekly_Sales'].count())

non_holiday = train[train.IsHoliday == False]
print('Sales on Non-Holidays : ', is_holiday['Weekly_Sales'].count())

