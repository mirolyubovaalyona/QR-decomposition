import numpy as np
from math import sqrt


def gs(X):
    O = np.zeros(X.shape)
    for i in range(X.shape[1]):
        # orthogonalization
        vector = X[:, i]
        space = O[:, :i]
        projection = vector @ space
        vector = vector - np.sum(projection * space, axis=1)
        # normalization
        norm = np.sqrt(vector @ vector)
        vector /= abs(norm) < 1e-8 and 1 or norm
        
        O[:, i] = vector
    return O


def qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q.round(6), A.round(6)
 
def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H


a=[[20, 2, 3, 7],
[1, 12, -2, -5],
[5, -3, 13, 0],
[0, 0, -3, 15]]

a=np.array(a)

b=[5, 4, -3, 7]

n=len(b)

qq, rr= np.linalg.qr(a)

#q, r = qr(a)
q=gs(a)
r=q.transpose().dot(a)

print(qq)
print(rr)

print("То что я считаю")
print(q)
print(r)

y=q.transpose().dot(b)

x=[0]*n
for i in range(n-1, -1, -1):
    s=0
    for k in range(n-1, i, -1):
        s = s+ r[i][k]*x[k]
    x[i] = (y[i] - s)/rr[i][i]

 
x=np.linalg.solve(r, y)
print("x:", x)

print(np.linalg.solve(a, b))
