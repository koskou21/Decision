import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import math
import matplotlib.pyplot as plt
import datetime 
import seaborn as sns


# Για να μην γράφουμε κάθε φορά τις παύλες στα prints
dashes = '-'*50  


#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# -------------------------------                 IMPORT CSV FILES                   -------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

# Στο train_data έχουμε το αρχείο train.csv
# Στο feat_data έχουμε το αρχείο features.csv

train_data = pd.read_csv("train.csv")
feat_data = pd.read_csv("features.csv", sep=";")

# Έλεγχος του τύπου δεδομένων κάθε στήλης

print("printing data types")
print(train_data.dtypes) 
print(feat_data.dtypes)


#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------       TREAT MISSING VALUES       --------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------




# Έλεγχος για τιμές που λείπουν 

print("Checking for missing values")
print(train_data.isnull().sum())
print("Checking for missing values features")
print(feat_data.isnull().sum())


# Η στήλη unemployment έχει κάποιες τιμές που λείπουν 
# Αντικαθιστούμε αυτά τα πεδία με τον μέσο όρο 

feat_data['Unemployment'] = feat_data['Unemployment'].fillna(np.mean(feat_data['Unemployment'] ))

# Έλεγχος για τιμές που λείπουν

print("Checking for missing values features")
print(feat_data.isnull().sum())

# Δεν λέιπουν τιμές 





#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#---------------------------            FIX DIFFERENT DATE FORMATS     ---------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------


# Τα αρχέια features και train έχουν διαφορετικά date formats 5/2/10 2010-02-05
# Τα date formats πρέπει να είναι τα ίδια για να γίνει σωστά το inner join παρακάτω
# Παρακάτω μετατρέπουμε τα πεδία Date και στα δύο αρχεία από object se datetime64[ns]


# feat_data

# dayfirst=True γιατί τα θέλουμε και τα δύο στο ευρωπαικό format (η datetime κάνει default στο αμερικάνικο και μας δημιουργεί προβλήματα)
feat_data['Date'] = pd.to_datetime(feat_data.Date, dayfirst=True)

print(dashes,"feat_data",dashes)
print(feat_data.head())
print(feat_data.dtypes)



# train_data 

train_data['Date'] = pd.to_datetime(train_data.Date)

print(dashes,"train_data",dashes)
print(train_data.head())
print(train_data.dtypes) 







#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#----------------------------        MERGE train_data + feat_data to all_data     ----------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------






# Χρειαζόμαστε όλες τις στήλες των train_data + feat_data σε ένα για να χτίσουμε το μοντέλο

print(dashes,"features and train shape",dashes)
print(feat_data.shape)
print(train_data.shape)


# Merge feature and training data in all_data
all_data = pd.merge(feat_data, train_data, on = ['Store', 'Date', 'IsHoliday'], how = 'inner')


#Check for missing values in all_data
print(dashes,"Checking for missing values all_data",dashes)
print(all_data.isnull().sum())
print(dashes,"all_data shape and tail(10)",dashes)
print(all_data.shape)
print(all_data.tail(10))



#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#----------------------------        MERGE test + feat_data to test_data     ---------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------



# Τo test_data (train.csv) περιέχει τις ~10.000 τελευταίες γραμμές (sort by date) του train.csv 
# Αυτό το κάνουμε για να μπορούμε να συγκρίνουμε τις προβλέψεις μας με τα δεδομένα που έχουμε 


test = pd.read_csv("test.csv")

print(dashes,'test info',dashes)
print(test.head())
print(test.dtypes) 
print(test.isnull().sum())


# Κάνε τη στήλη dates και του αρχείου test.csv datetime64[ns] για τον ίδο λόγο που αναφέραμε παραπάνω 

test['Date'] = pd.to_datetime(test.Date)

print(dashes,"test head and types",dashes)
print(test.shape)
print(test.head())
print(test.dtypes) 



# Merge feature and training data in all_data
test_data = pd.merge(feat_data, test, on = ['Store', 'Date', 'IsHoliday'], how = 'inner')



print(dashes,"Checking for missing values test_data",dashes)
print(test_data.isnull().sum())
print(dashes,"test_data shape and head(10)",dashes)
print(test_data.shape)
print(test_data.head(10))




# Se ayto to shmeio mporei na theloumisoume na ta kanoume etsi kai alliws sort according to date 
# tha to lysoume an prokypsei 

#print(all_data.head(20))


""" 
# Sort the data w.r.t 'Date' as we have time series problem
df = df.sort_values(by = 'Date')
df.head(20)
 """




#-------------------------------------------------------------------------------------------------
#--------------------------------                                      ---------------------------
#--------------------------------       Corelation+Outliers             ---------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


#          Όλα τα plots είναι μέσα σε comments για να μην καθυστερεί η εκτέλεση του προγράμματος 






# Total count of sales on holidays and non-holidays
is_holiday = train_data[train_data.IsHoliday == True]

count = 0
count2 = 0
for holiday in train_data['IsHoliday']:
    if (holiday): count = count +1
    else: count2 = count2 + 1

print("Holidays: ",count)
print("NotHoliday: ", count2)



# Θέλουμε να δούμε τη συσχέτιση μεταξύ των μεταβλητών.Παρακάτω το δείχνουμε με τη βοήθεια ενός heatmap
# Θέλουμε επίσης να δούμε αν υπάρχουν μεταβλητές οι οποίες έχουν τεράστια συσχέτιση μεταξύ τους, π.χ. 0.9
# Αν υπάρχουν τέτοιες μεταβλητές τότε αυτό σημαίνει ότι ουσιαστικά είναι "ίδιες" ως προς το πως επηρεάζουν την πρόβλεψη.
# Αυτό σημαίνει ότι δεν θα μπορούμε να ξεχωρίσουμε το αντίκτυπο της μίας από την άλλη στο μοντέλο και σε αυη την περίπτωση θα πρέπει να το ανιτμετοπίσουμε
# Πριν προχωρίσουμε 



corr = all_data.corr()

""" plt.figure(figsize = (15, 10))
sns.heatmap(corr, annot = True, cmap="RdBu")
#plt.show()  """





#Με τη βοήθεια του heatmap βλέπουμε ότι δεν αντιμετοπίζουμε τέτοια προβλήματα
#corr.to_csv('corelation.csv')  #δεν θα το ξαναχρειαστούμε μάλλον αλλά το αφήνω 

# Ένα άλλο πρόβλημα που μπορεί να αντιμετοπίσουμε είναι να έχουμε αρνητικές τιμές στην μεταβλητή y που θέλουμε να προβλέψουμε
# Σε αυτή την περίπτωση αντικαθιστούμε αυτές τις τιμές με 0




""" sns.displot(all_data.Weekly_Sales)
plt.show()
print(all_data.loc[all_data.Weekly_Sales < 0, 'Weekly_Sales']) """

# Φαίνεται ότι όντως έχουμε αρνητικές τιμές
# Τις αντικαθιστούμε με 0



all_data.loc[all_data.Weekly_Sales < 0, 'Weekly_Sales'] = 0


""" sns.displot(all_data.Weekly_Sales)
plt.show()
print(all_data.loc[all_data.Weekly_Sales < 0, 'Weekly_Sales'].count())
 """






#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#---------------------------       LINEAR REGRESSION        --------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------








# Import the required library
from sklearn.model_selection import train_test_split

#X columns
X_columns =  all_data.columns.difference(['Weekly_Sales'])
print(X_columns)

#train-test split
X_train,X_test,y_train,y_test = train_test_split( all_data[X_columns], all_data['Weekly_Sales'], test_size = 0.20, random_state = 0)

print(dashes, "X+y variables", dashes)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train.columns)



























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