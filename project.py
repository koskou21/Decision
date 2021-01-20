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
#----------------------------                                                       --------------------------
# ---------------------------                   IMPORT CSV FILES                    -------------------------
#----------------------------                                                       --------------------------
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
#----------------------------                                                       --------------------------
#----------------------------                   TREAT MISSING VALUES                --------------------------
#----------------------------                                                       --------------------------
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
#----------------------------                                                     ----------------------------
#---------------------------            FIX DIFFERENT DATE FORMATS                ----------------------------
#----------------------------                                                     ----------------------------
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
#----------------------------                                                     ----------------------------
#----------------------------        MERGE train_data + feat_data to all_data     ----------------------------
#----------------------------                                                     ----------------------------
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
#----------------------------                                                  -------------------------------
#----------------------------        MERGE test + feat_data to test_data       -------------------------------
#----------------------------                                                  -------------------------------
#-------------------------------------------------------------------------------------------------------------


# Χρειαζόμαστε ένα αρχείο test.csv για λόγους που εξηγούνται στην αναφορά (ελπίζω)
# Εν συντομία, αργότερα θα ελέγξουμε την επεκταστιμότητα του αλγορίθμου που εκπαιδεύσαμε για ένα (ας πούμε) διαφορετικό dataset 
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




#---------------------------------------------------------------------------------------------------------
#--------------------------------                                               --------------------------
#--------------------------------           Plots + Handling Outliers           --------------------------
#--------------------------------                                               --------------------------
#---------------------------------------------------------------------------------------------------------


#          Όλα τα plots είναι μέσα σε comments για να μην καθυστερεί η εκτέλεση του προγράμματος 





# Δεν είμαι σίγουρος ότι το χρειαζόμαστε, ίσως το σβήσω.

# Total count of sales on holidays and non-holidays
""" is_holiday = train_data[train_data.IsHoliday == True]

count = 0
count2 = 0
for holiday in train_data['IsHoliday']:
    if (holiday): count = count +1
    else: count2 = count2 + 1

print("Holidays: ",count)
print("NotHoliday: ", count2) """




# Ένα πρόβλημα που μπορεί να αντιμετοπίσουμε είναι να έχουμε αρνητικές τιμές στην μεταβλητή y που θέλουμε να προβλέψουμε
# Σε αυτή την περίπτωση αντικαθιστούμε αυτές τις τιμές με 0
# Εναλλακτικά θα μπορούσαμε να διαγράψουμε αυτές τις γραμμες




""" sns.displot(all_data.Weekly_Sales)
plt.show()
print(all_data.loc[all_data.Weekly_Sales < 0, 'Weekly_Sales'])  """

# Φαίνεται ότι όντως έχουμε αρνητικές τιμές
# Τις αντικαθιστούμε με 0



all_data.loc[all_data.Weekly_Sales < 0, 'Weekly_Sales'] = 0


""" sns.displot(all_data.Weekly_Sales)
plt.show()
print(all_data.loc[all_data.Weekly_Sales < 0, 'Weekly_Sales'].count())
 """



#-------------------------------------------------------------------------------------------------
#--------------------------------                                       --------------------------
#--------------------------------             Fix dates again           --------------------------
#--------------------------------                                       --------------------------
#-------------------------------------------------------------------------------------------------

# Δυστυχώς η linreg που χρησιμοποιούμε για το linear regression δεν δέχεται datetime objects ή strings γενικότερα
# Επομένως θα πρέπει να φτιάξουμε καινούργιες στήλες που να περιέχουν την πληροφορία που χρειαζόμαστε. 

# Επιπλέον, φτιάχνουμε μία καινούργια στήλη "Days2Christmas" η οποία μετράει τις μέρες που απομένουν για τα Χριστούγεννα.
# Περιμένουμε ότι οι πωλήσεις θα αυξηθούν όσο πλησιάζουν οι γιορτές και μάλιστα τα IsHoliday είναι μέσα σε αυτές ή προηγούνται αυτών.



# Ετος
all_data['Year'] = pd.to_datetime(all_data['Date'], format = '%Y-%m-%d').dt.year
test_data['Year'] = pd.to_datetime(test_data['Date'], format = '%Y-%m-%d').dt.year

# Μηνας
# Βγάλαμε αυτή τη στήλη καθώς έχει -1 correlation με το Days2Christmas

#all_data['Month'] = pd.to_datetime(all_data['Date'], format = '%Y-%m-%d').dt.month
#test_data['Month'] = pd.to_datetime(test_data['Date'], format = '%Y-%m-%d').dt.month

# Ημερα
all_data['Day'] = pd.to_datetime(all_data['Date'], format = '%Y-%m-%d').dt.day
test_data['Day'] = pd.to_datetime(test_data['Date'], format = '%Y-%m-%d').dt.day

# Στήλη για τις μέρες που απομένουν μέχρι τα επόμενα Χριστούγεννα 
all_data["Days2Christmas"] = (pd.to_datetime(all_data['Year'].astype(str)+ "-12-31", format="%Y-%m-%d") -pd.to_datetime(all_data["Date"], format="%Y-%m-%d")).dt.days.astype(int)

# Στήλη για τις μέρες που απομένουν μέχρι τα επόμενα Χριστούγεννα 
test_data["Days2Christmas"] = (pd.to_datetime(test_data['Year'].astype(str) + "-12-31", format="%Y-%m-%d") - pd.to_datetime(test_data["Date"], format="%Y-%m-%d")).dt.days.astype(int)





# Σε αυτό το σημείο Θέλουμε να δούμε τη συσχέτιση μεταξύ των μεταβλητών.Παρακάτω το δείχνουμε με τη βοήθεια ενός heatmap
# Θέλουμε επίσης να δούμε αν υπάρχουν μεταβλητές οι οποίες έχουν τεράστια συσχέτιση μεταξύ τους, π.χ. 0.9 (μία τέτοια συχέτιση παρατηρήσαμε μεταξύ "Month" και "Days2Christmas")
# Αν υπάρχουν τέτοιες μεταβλητές τότε αυτό σημαίνει ότι ουσιαστικά είναι "ίδιες" ως προς το πως επηρεάζουν την πρόβλεψη.
# Αυτό σημαίνει ότι δεν θα μπορούμε να ξεχωρίσουμε το αντίκτυπο της μίας από την άλλη στο μοντέλο και σε αυη την περίπτωση θα πρέπει να το ανιτμετοπίσουμε
# Πριν προχωρίσουμε 


#-------------------------                          --------------------
#-------------------------       Correlation        --------------------
#-------------------------                          --------------------





corr = all_data.corr()

""" 
plt.figure(figsize = (15, 10))
sns.heatmap(corr, annot = True, cmap="RdBu")
plt.show()  """



# Τυπώνουμε το αποτέλεσμα σε ένα αρχείο
# corr.to_csv('correlation.csv') 
# Δεν θα το ξαναχρειαστούμε μάλλον αλλά το αφήνω 




# Προσθέσαμε στήλες άρα τα ξανατυπώνουμε
print(dashes,dashes)
print(all_data.head())
print(dashes,"test_data final!",dashes)
print(test_data.head())




#--------------------------------------------------------------------------------------------------
#----------------------------                                --------------------------------------
#---------------------------        Feature Selection        --------------------------------------
#---------------------------                                 --------------------------------------
#--------------------------------------------------------------------------------------------------




# Διαλέγουμε ποιές στήλες θέλουμε να χρησιμοποιήσουμε για την εκπαίδευση
# Ορίζουμε τις X,y μεταβλητές 


select_columns = all_data.columns.difference(['Weekly_Sales', 'Date', 'Unemployment', 'Temperature'])

# Για να τσεκάρουμε αν όντως έχουν μείνει οι σωστές στήλες
print(dashes,'check',dashes)
print(all_data[select_columns].head())


from sklearn.model_selection import train_test_split


# Δημιουργούμε τις X,y μεταβλητές 
X_train,X_test,y_train,y_test = train_test_split( all_data[select_columns], all_data['Weekly_Sales'], test_size = 0.20, random_state = 0)

# Έλεγχος
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Δημιουργία X,y μεταβλητών για το dataset που ετοιμάσαμε, δηλαδή το test.csv μέσα στο test_data
# Δεν χρειαζόμαστε X2_train / y2_train αφόυ δεν θα εκπαιδεύσουμε τον αλγόριθμό μας με βάση αυτό το dataset, απλά θέλουμε να ελέγξουμε την επεκταστιμότητά του

X2_test = test_data[select_columns]
y2_test = test_data['Weekly_Sales']





#--------------------------------------------------------------------------------------------------
#----------------------------                                --------------------------------------
#---------------------------        LINEAR REGRESSION        --------------------------------------
#---------------------------                                 --------------------------------------
#--------------------------------------------------------------------------------------------------







print('debug')

print(dashes,dashes)
print(X_train)
print(dashes,dashes)

print(dashes,dashes)

print(X_test)




from sklearn.linear_model import LinearRegression

# Checking errors
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Initialize and fit the data into the model
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_train_pred = linreg.predict( X_train )
print(dashes,'print pred',dashes)

print(y_train_pred)



#Predicting for test
y_test_pred = linreg.predict( X_test )

all_data_test_pred = pd.DataFrame({'actual' : y_test,'Predicted' : y_test_pred})
print(dashes,'all_data pred',dashes)

print(all_data_test_pred.head(10))

print(dashes,dashes)


# Calculating Mean Absoluate Error
print(dashes,dashes)

print('Train MAE : ', mean_absolute_error(y_train, y_train_pred).round(2))
print('Test MAE  : ', mean_absolute_error(y_test, y_test_pred).round(2))

# Calculate Root Mean Squared Error
print(dashes,dashes)

print('Train RMSE : ', np.sqrt(mean_squared_error(y_train, y_train_pred)).round(2))
print('Test  RMSE : ', np.sqrt(mean_squared_error(y_test, y_test_pred)).round(2))

accuracy = np.round(linreg.score( X_test, y_test) * 100, 2)
print(dashes,dashes)

print('Linear Regression Accuracy : ', accuracy )








import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
print(dashes,dashes)



#-----------------------------------------------------------------------------------------------------------
#------------------------------                                       --------------------------------------
#------------------------------              RANDOM FOREST             --------------------------------------
#------------------------------                                        --------------------------------------
#-----------------------------------------------------------------------------------------------------------





from sklearn.ensemble import RandomForestRegressor
paragrid_rf = {'n_estimators': [100,150,200,250,300,350,400]}
from sklearn.model_selection import GridSearchCV


gscv_rf = GridSearchCV(estimator = RandomForestRegressor(), param_grid = paragrid_rf, cv = 5,verbose = True)


# Set number of estimators and depth 
# Τις χρησιμοποιόυμε σε >1 σημεία οπότε τα έβαλα σε μεταβλητές
estimators = 40
depth = 20



prediction_data = ("e:",estimators,"d:",depth)
print(prediction_data)
rfr = RandomForestRegressor(n_estimators = estimators,max_depth=depth)    

# Δεν έχει κολήσει απλά θέλει το χρόνο του    
print("Loading...")

y_pred_train = rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
predict_rfr = rfr.predict(X_test)

print(dashes)
print(pd.DataFrame({'Actual': y_test,'Predicted' : predict_rfr}))
print(dashes)



#print('Train DT MAE : ', mean_absolute_error(y_train, y_pred_train).round(2))
print('Test  DT MAE : ', mean_absolute_error(y_test, predict_rfr).round(2))
#print('Train DT RMSE : ', np.sqrt(mean_squared_error(y_train, y_pred_train)).round(2))
print('Test  DT RMSE : ', np.sqrt(mean_squared_error(y_test, predict_rfr)).round(2)) 

forest = ('Acc:', np.round(rfr.score(X_test, y_test) * 100, 2))
print(forest)




# Σε αυτό το σημείο θέλουμε να ελέγξουμε την επεκταστιμότητα του αλγοριθμου πάνω σε ένα dataset στο οποίο δεν έχει εκπαιδευτεί, δηλαδή το dataset test.csv
predict_test = rfr.predict(X2_test)
forest_test = ('Acc:', np.round(rfr.score(X2_test, y2_test) * 100, 2))
print(forest_test,"-> for train_data")




# Για την εκτύπωση των αποτελεσμάτων στo forest_evaluation.txt
""" 
import sys
with open("forest_evaluation.txt", "a") as text_file:
    print(prediction_data,forest, file=text_file) 
"""