#######################
# Logistic Regression #
#######################

# Goad: build a logistic regression model to predict whether a student gets admitted into a university based #
# on scores of two exams. #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############
# Readin Data #
###############
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(data.head())


##################
# Visualize Data #
##################
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=[10, 6])
ax.scatter(positive['Exam 1'], positive['Exam 2'],
           s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50,
           c='r', marker='x', label="Not Admitted")
ax.legend()
ax.set_xlabel('Exam 1 Scores')
ax.set_ylabel('Exam 2 Scores')
plt.show()
# From the plot, we can see that there is a clear boundary between two classifications.
# We are able to train a model to predict the result


####################
# Sigmoid Function #
####################
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Check if the sigmoid function works correctly
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()


##############################
# Cost function and gradient #
##############################
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum((first - second) / (len(X)))


# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, "Ones", 1)  # Syntax: insert(index, list_name, list_values)

# set X (training data) and y (target variable)
cols = data.shape[1]
# subsetting pandas data frames by using iloc[] method
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

print(cost(theta, X, y))  # We have 0.6931471805599457

# Note that this is gradient of cost (Not Gradient Descent)


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad


print(gradient(theta, X, y))


#######################
# Learning parameters #
#######################
# In Python, we ccan use "optimize" from SciPy to find optimal parameters
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print(result)

print(cost(result[0], X, y))  # We have 0.20349770158947458


###########################
# Evaluating Logistic Reg #
###########################
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
# zip(X, Y) to create an iterator that produces tuples of the form (x, y).
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0))
           else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%' .format(accuracy))


###########################
# Find Decision Boundary  #
###########################
coef = -(result[0] / result[0][2])
bx = np.arange(130, step=0.1)
by = coef[0] + coef[1] * bx

fig, ax = plt.subplots(figsize=[10, 6])
ax.scatter(positive['Exam 1'], positive['Exam 2'],
           s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50,
           c='r', marker='x', label="Not Admitted")
ax.plot(bx, by, 'g')
ax.set_title('Boundary Decision')
ax.legend()
ax.set_xlabel('Exam 1 Scores')
ax.set_ylabel('Exam 2 Scores')
plt.show()


###################################
# Regularized logistic regression #
###################################

###############
# Readin Data #
###############
path2 = 'ex2data2.txt'
data2 = pd.read_csv(path2, header=None, names=['Test 1', 'Test 2', 'Accepted'])
print(data2.head())

##################
# Visualize Data #
##################
positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=[10, 6])
ax.scatter(positive['Test 1'], positive['Test 2'],
           s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'],
           s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
# From the plot, we can see that shows that our dataset cannot be separated into positive and
# negative examples by a straight-line through the plot (Non-Linear Decision Boundary).


###################
# Feature Mapping #
###################
# Thus, One way to fit the data better is to create more features from each data point.
# A logistic regression classifier trained on this higher-dimension feature vector will have a
# more complex decision boundary and will appear nonlinear when drawn in our 2-dimensional plot.

degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

data2.head()


####################
# Regularized Cost #
####################
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * \
        np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + \
                ((learningRate / len(X)) * theta[:, i])

    return grad


####################
# Intialize Values #
####################
# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]
X2 = data2.iloc[:, 1: cols]
y2 = data2.iloc[:, 0: 1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)

# Also, assign a value to learningRate(fix later)
learningRate = 1
print(costReg(theta2, X2, y2, learningRate))  # 0.6931471805599454
print(gradientReg(theta2, X2, y2, learningRate))


#######################
# Learning parameters #
#######################
# In Python, we ccan use "optimize" from SciPy to find optimal parameters
result2 = opt.fmin_tnc(func=costReg, x0=theta2,
                       fprime=gradientReg, args=(X2, y2, learningRate))
print(result2)

theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0))
           else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))

# Another way to find accuracy
from sklearn import linear_model
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X2, y2.ravel())
# Can increase accuracy by manipulating argument values
print(model.score(X2, y2))
