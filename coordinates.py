import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})



def prnt(X, s):
    print("MATRIX", s)
    print(X)
    print(X.shape, '\n')
    print("\n")


#take X and y as input and output
file = open("input.txt", "r")

input = file.readlines()
m = len(input)

file.close()

file = open("input.txt", "r")
input = file.readlines()
print(len(input))
file.close()
m = len(input)
X = np.zeros([m, 2])
y = np.zeros([m, 1])

# print(X)

for i in range(m):
    hmm = input[i].split()
    X[i, 0] = hmm[0]
    X[i, 1] = hmm[1]
    y[i] = hmm[2]

# print(X)
# print(y)
#################################

showshape = 1

"""SOME INITIALIZATIONS"""
input_layer = 2
hidden_layer = 5
output_layer = 4
Lambda = 1

theta1 = np.random.rand(hidden_layer, input_layer + 1)
theta2 = np.random.rand(output_layer, hidden_layer + 1)

X = np.transpose(X)

print('\n')
new = np.ones([1, m])
X = np.vstack([new, X])
# prnt(X,"x")


"""OUTPUT Y FORMATION"""
# print(y)
yy = np.zeros([output_layer, m])

for i in range(m):
    yy[int(y[i]) - 1, i] = 1

y = yy
# prnt(y,"y")


"""This function combines the forward propagation, backpropagation and
cost computation inside a single method and returns the cost and the
gradients of theta vectors which will be used in GRADIENT DESCENT"""


def function(theta1, theta2, i):
    """FORWARD PROPAGATION"""
    z = np.dot(theta1, X);
    a = expit(z)
    a = np.vstack([new, a])
    hx = np.dot(theta2, a)
    hx = expit(hx)
    # if i==0 or i==itr-1:
    # prnt(hx,"hux")

    """COST COMPUTATION"""
    q1 = np.multiply(y, np.log(hx))
    q2 = np.multiply((1 - y), np.log(1 - hx))
    p1 = -1 * (q1 + q2) / m;
    p1 = sum(sum(p1))

    temp = np.multiply(theta1[:, 1:input_layer + 1], theta1[:, 1:input_layer + 1])
    p2 = sum(sum(temp))
    temp = np.multiply(theta2[:, 1:hidden_layer + 1], theta2[:, 1:hidden_layer + 1])
    p2 += sum(sum(temp))
    p2 = p2 * Lambda / (2 * m)

    J = p1 + p2

    """BACHPROPAGATION"""
    delta3 = hx - y
    delta2 = np.dot((np.transpose(theta2)), delta3)
    temp = np.multiply(a, 1 - a)
    delta2 = np.multiply(delta2, temp)

    grad2 = np.dot(delta3, (np.transpose(a)))
    grad2 = grad2 / m
    grad2[:, 1:hidden_layer + 1] += (Lambda / m) * theta2[:, 1:hidden_layer + 1]

    delta2 = delta2[1:hidden_layer + 1, :]
    grad1 = np.dot(delta2, (np.transpose(X)))
    grad1 = grad1 / m
    grad1[:, 1:input_layer + 1] += (Lambda / m) * theta1[:, 1:input_layer + 1]

    return J, grad1, grad2


def gradientDescent(itr, theta1, theta2, alpha):
    it = np.zeros([itr, 1])
    costhist = np.zeros([itr, 1])
    for i in range(itr):
        c, g1, g2 = function(theta1, theta2, i)  # compute gradient #compute cost
        theta1 = theta1 - alpha * g1  # update theta
        theta2 = theta2 - alpha * g2  # update theta
        it[i] = i
        costhist[i] = c
        if i == 0 or i == itr - 1:
            print("iteration=", i, "  cost=", c)
        if i == itr - 1:
            ft1 = theta1
            ft2 = theta2

    plt.figure(figsize=(9, 6))
    plt.plot(it, costhist)
    plt.grid()
    plt.show()
    return ft1, ft2


def check(theta1, theta2, X):
    z = np.dot(theta1, X);
    a = expit(z)
    a = np.vstack([new, a])
    hx = np.dot(theta2, a)
    hx = expit(hx)
    #prnt(hx, "hux of check")

    l = X.shape[1]

    ans = np.zeros([l, 1])

    for j in range(l):
        mx = -100
        for i in range(output_layer):
            if hx[i, j] > mx:
                ans[j] = int(i + 1)
                mx = hx[i, j]

    return ans


"""SOME OTHER INITIALIZATIONS AND CALLNG GD"""
alpha = 0.75
itr = 420
ft1, ft2 = gradientDescent(itr, theta1, theta2, alpha)
# prnt(y,"y")

# =============================================================================
# n=int(input())
# x=np.zeros([2,n])
# for i in range(n):
#     x1=int(input())
#     x2=int(input())
#     x[0,i]=x1
#     x[1,i]=x2
# =============================================================================

x = np.array([[7, 8]
                 , [6, 1]
                 , [1, -6]
                 , [-1, -6]
                 , [-88, -6]
                 , [-1, -66]
                 , [-8, 1]])
n = x.shape[0]
x = np.transpose(x)

new = np.ones([1, n])
x = np.vstack([new, x])
# prnt(x,"x")

ans = check(ft1, ft2, x)
print(ans)
