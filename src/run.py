import numpy as np
from proj1_helpers import *
from implementations import *
from helpers import *
DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'

# load train data 
y_train, x_train, ids = load_csv_data(DATA_TRAIN_PATH)

# store the index for different group
a, b, c, d = index_search(x_train)
# set null value to mean
x_train = fix_null(x_train)
# use feature augumentation
x_train = feature_aug(x_train, 8)
# standardize the data
x_train = standlization(x_train)
# add a column to fit bias
x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]

# split train data and transfer to numpy array
x_train, y_train = divide(a,b,c,d,x_train,y_train)
x_train = np.array(x_train)
y_train = np.array(y_train)

# load test data 
y_test, x_test, ids = load_csv_data(DATA_TEST_PATH)

# get the index for different group
m,n,k,l = index_search(x_test) 

# polinomial degree
degree = 8

# corresponding feature parameter num
para_num = 30 * degree + 1

# preprocess the test data, the same as training
x_test = fix_null(x_test)
x_test = feature_aug(x_test, degree)
x_test = standlization(x_test)
x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

# split test data and transfer to numpy array
x_test, y_test = divide(m,n,k,l,x_test,y_test)
x_test = np.array(x_test)
y_test = np.array(y_test)

# get w parameters of 4 groups
ws = np.zeros((4,para_num))
for i in range(4):
    w = ridge_regression_demo_new(x_train[i], y_train[i])
    ws[i] = w

# predict the test data of 4 group
for i in range(4):
    y_test[i] = predict_labels(ws[i],x_test[i])

# merge the predictions of 4 groups
index = [m,n,k,l]
length = len(y_test[0]) + len(y_test[1]) + len(y_test[2]) + len(y_test[3])
kk = np.zeros((length))
for i in range(4):
    for j in range(len(index[i])):
        kk[index[i][j]] = y_test[i][j]

# Export CSV file
create_csv_submission(ids, kk, 'haha.csv')
