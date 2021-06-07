import numpy as np
from math import sqrt



def gs(X, n):
    a=X.transpose()
    u=e=[np.array(n)]*n
    for i in range(n):
        u[i]=np.array(a[i])
        for j in range(i-1, -1, -1):
            aa=np.array(a[i])
            ee=np.array(e[j])
            u[i]=u[i]-np.array((np.array(aa).dot(np.array(ee)))).dot(np.array(ee))
        nor=np.linalg.norm(u[i],  ord=2)
        e[i]=u[i]/nor
     
    return np.array(e).transpose()




a=[[1, 2, 4],
[3, 3, 2],
[4, 1, 3]]

a=np.array(a)

b=[5, 4, -3]

n=len(b)


q=gs(a, n)
r=q.transpose().dot(a)



y=q.transpose().dot(b)

x=[0]*n
for i in range(n-1, -1, -1):
    s=0
    for k in range(n-1, i, -1):
        s = s+ r[i][k]*x[k]
    x[i] = (y[i] - s)/r[i][i]

 
x=np.linalg.solve(r, y)
print("x:", x)

print(np.linalg.solve(a, b))
