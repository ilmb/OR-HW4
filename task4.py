
# coding: utf-8

# In[1]:


get_ipython().system(u'pip install cvxopt')


# In[172]:


from cvxopt import matrix, solvers, sparse
import numpy as np

A = matrix([ [1.0, -3.0, -6.0, 0.0, -4.0, -1.0], [-5.0, -1.0, 2.0, -4.0, 1.0, 0.0]])
b = matrix([ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
c = matrix([ 1.0, 1.0 ])
sol= solvers.lp(c, A, b, solver = 'glpk')
print(sol)
x = sol['x']
print(x)
# [1.0, 0.4]
f_min = sol['primal objective']
print(f_min)
# 1.3999999981965823
v = 1/f_min 
print(v)
# Game price:0.7142857160301338
p_0 = v*x
print(p_0)
# Стратегия для первого игрока: [7.14e-01, 2.86e-01]
##### Для второго игрока
G = A.trans()
q = matrix(np.eye(6))
h = matrix(np.zeros((6,1)))
h = sparse([[c, h]])
h = matrix(h, (8, 1), 'd')
G = sparse([G, q])
sol= solvers.lp(b, (-1)*G, h, solver = 'glpk')
y = sol['x']
print(y)
# [ 2.00e-01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.20e+00]
f_min2 = sol['primal objective']
print(f_min2)
# -1.4
game_price = -1/f_min2
print(game_price)
#Game price: 0.7142857142857143
q2_0 = v * y
print(q2_0)
# Стратегия для первого игрока: [ 1.43e-01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 8.57e-01]


