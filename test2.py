#just checking if github works

#Sauce: https://github.com/dataprofessor/code/blob/master/python/linear_regression.ipynb
#Sauce2: https://github.com/vasukapil2015/Walmart-Store-Sales-Forecasting/blob/main/Walmart_Store_Sales_Forecasting.ipynb

#Προσπάθησα να συνδυάσω ένα πίο απλό παράδειγμα εκπαίδευσης linear regression 
#https://www.youtube.com/watch?v=R15LjD8aCzc
#Και της εργασίας που βρήκαμε που μοιάζει ή είναι η ίδια με τη δική μας προκειμένου να την
#καταλάβω καλύτερα


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Βάλε στη μεταβλητή train το αρχείο train.csv
#kanoume eisagwgh twn arxeiwn  twn dedomenwn mas
train = pd.read_csv('train.csv')
features = pd.read_csv('features.csv')


#emfanizoume ta dedomena mas wwww
print('Features : ', features.shape)
print('Train    : ', train.shape)

#Κάνε import και το αρχείο features και κάντα merge ώστε να έχουμε σε ένα αρχείο όλες τις
#πληροφορίες που χρειαζόμαστε

#βάλε όλες τις στήλες εκτός του weekly sales ως X variables σε ένα matrix
#βάλε την στήλη weekly sale ως Y variable σε ένα άλλο matrix αφού αυτή είναι η εξαρτημένη
#μεταβλητή που θέλουμε να προβλέψουμε 

#το sklearn μας χρειάζεται για το linear regression

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#φτιάξε το μοντέλο με τα X,Y matrixes που ορίσαμε παραπάνω
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

#Σε αυτό το σημείο θα πρέπει να είμαστε σε θέση να προβλέψουμε τις επόμενες πωλήσεις
#Αυτό μπορούμε να το κάνουμε ετοιμάζοντας ένα νέο matrix Χ με τις τιμές που έχουμε από
#και να προσπαθίσουμε να προβλέψουμε το Υ (αντί να έιναι δωσμένο)
#και τέλος να συγκρίνουμε το αποτέλεσμα με το πραγματικό Υ

Y_pred = model.predict(X_test) #όπου Χ_test αυτό που ανέφερα στα προηγούμενα σχόλια


#Εκτύπωσε την αποτελεσματικότητα του μοντέλου που φτιάξαμε
#Δεν είμαι σίγουρος τι σημαίνουν αυτές οι μετρικές αλλά θα το βρούμε στην πορεία

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred)



#Οπτικοποίηση της σύγκρισης μεταξύ του πραγματικού Υ και αυτού που προβλέψαμε

import seaborn as sns
Y_test                  #απ ότι καταλαβαίνω, αυτό είναι το πραγματικό Υ
np.array(Y_test)
Y_pred                  #και αυτό που προβλέψαμε

#σε αυτό το σημείο περιμένω να δούμε μια γραφική αναπαράσταση που θα δείχνει 
#πόσο "κοντά" είναι η πρόβλεψη στην πραγματικότητα

sns.scatterplot(Y_test, Y_pred, alpha=0.5) #alpha=0.5 για καλύτερη ορατότητα 


