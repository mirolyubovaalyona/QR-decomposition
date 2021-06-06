import numpy as np
from math import sqrt



def qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A
 
def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H


a=[[  5,  11, -15],
 [ 12,  34, -51],
 [-24, -43,  92]]

a=np.array(a)

b=[12, 13, 14]

n=len(b)

qq, rr= np.linalg.qr(a)
q, r = qr(a)
q= q.round(6)
r= r.round(6)

print(qq)
print(rr)

print(q)
print(r)


y=[0]*n
x=[0]*n

y[0]=b[0]/t[0][0]
for i in range(1, n):
    s = 0;
    for k in range(0, i):
        s  += t[i][k]*y[k]
    y[i] = (b[i] - s)/t[i][i]


x[n-1]=y[n-1]/tt[n-1][n-1]
for i in range(n-2, -1, -1):
    ss=0
    for k in range(n-1, i, -1):
        ss = ss+ tt[i][k]*x[k]
    x[i] = (y[i] - ss)/tt[i][i]

print("x:", x)

print(np.linalg.solve(a, b))
