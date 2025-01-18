import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#main function
def main():
# Setting condition to display all rows and only 5 columns
pd.set_option('display.max_rows', 5, 'display.max_columns', 13)
train = pd.read_csv(r'{your_location}', index_col=0)
predi = pd.read_csv(r'{your_location}')
print()
print(' Welcome to REAL ESTATE PREDICTOR!
')
print()
print('The following options are available: ')
print()
print('1. Make a Prediction')
print('2. Add your Prediction')
print('3. Modify previously made Prediction')
print('4. Delete Row')
print('5. Visualise Predicted Data')
print()
ch = input('Your Choice (1 - 5): ')
print()
if ch.isdigit() != True:
print('Invalid input')
exit(1)
else:

14

ch = int(ch)
if ch not in range(1, 6):
print('Invalid input')
exit(1)
if ch == 1:
pred(train)
if ch == 2:
add()
if ch == 3:
modif(predi)
if ch == 4:
delt(predi)
if ch == 5:
visuali()

#else successfully exit from the program
else:
exit(0)
def add():
print(' ADD A PREDICTION ')
x = pd.read_csv(r'D:\class 12\python_\pred.csv')
b = pd.DataFrame([], columns=x.columns)
a = []
for n in range(len(x.columns)):
print(x.columns[n], ':', sep="")
a.append(int(input()))
a = np.array(a)
b.loc[0] = a

# Append to the csv, do not add an index or header
b.to_csv('pred.csv', mode='a', index=False, header=False)
print('Your row has been added.')

#Function to modify elements of the csv
def modif(df):
#Accepting the row and column index labels

15

row = input('Row: ')
column = input('Column: ')
#Checking if row label is a digit (if row and columns are digits df.iloc[]
must be used instead of df.loc[])
#Changing string to int if digit
if row.isdigit():
row = int(row)
#Checking if column label is a digit
#Changing string to int if digit
if column.isdigit():
column = int(column)
#Accepting the modified value
value = input('Data Value: ')
#Ensuring row and column is int to use df.iloc[]
if isinstance(row, int) and isinstance(column, int):
df.iloc[row, column] = value
#if not int, using df.loc[]
else:
df.loc[row, column] = value
#writing modified df to csv file
df.to_csv('pred.csv', index=False)
print('Your data has been modified')
def visuali():
pred_csv = pd.read_csv('pred.csv')
print('Visualisation of predicted data')
print('Your chosen field shall be plotted against the SalePrice.')
print('Field/Column: ')
field = input()
plt.plot(pred_csv[field], pred_csv['SalePrice'], marker='*')
plt.title(f'{field} Vs Sale Price')
plt.ylabel(field)
plt.xlabel('Sale Price')
plt.show()
def pred(df):
#Creation of model
model = RandomForestRegressor(random_state = 1)
#x stands for all columns in dataframe except SalePrice
x = df.iloc[:, : 11]

16

#y includes the SalePrice column
y = df.SalePrice
#x and y are split into training and testing dataset
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 1)
#fitting of model
model.fit(train_x, train_y)
#evaluating predictions and calculating mean absolute error to provide
range of cost to user
prediction = model.predict(val_x)
#error is used to provide a range to user
error = mean_absolute_error(val_y, prediction)
#instead of splitting the dataframe passed into the function into training
and testing data
#the entire dataframe is used to train
#this shall help gain better accuracy
train_x = x
train_y = y
model.fit(train_x, train_y)
print(' MAKE A PREDICTION ')
print()
num_pred = int(input('Number predictions to be made: '))
#Creation of dataframe
#this dataframe shall be used to store the data provided by the user
#this dataframe shall be passed into predict()
b = pd.DataFrame([], columns = x.columns)

for j in range(num_pred):
print('House', j + 1, ':')
print()
a = []
for n in range(len(x.columns)):
print(x.columns[n], ':', sep="")

a.append(int(input()))
a = np.array(a)
b.loc[j] = a
prediction = model.predict(b)

17

b['SalePrice'] = prediction
# Append to the csv, do not add an index or header
b.to_csv('pred.csv', mode = 'a', index= False, header = False)
print(b)
print()
print('Predicted Sale Price:', prediction[0])
print(f'Range of Possible Error: {round(error)}')

print('Predictions written at pred.csv')

def delt(df):
print(' DELETE A ROW ')
n = int(input('Number of rows to be deleted: '))
rws = []
for x in range(n):
rws.append(int(input('Row label: ')))
df.drop(rws, axis= 0, inplace = True)

df.to_csv('pred.csv', index=False)
print(f'Row(s) {rws} has/have been deleted')
main()