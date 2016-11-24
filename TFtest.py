#! /usr/bin/env python

# memoize
f_hash = {}
def f(x, t, b):
    if (x,t,b) not in f_hash:
        if x==1 and t==0:
             f_hash[(x,t,b)] = 1.
        elif x==0 and t==1:
             f_hash[(x,t,b)] = 1 - b
        elif x==2 and t==1:
             f_hash[(x,t,b)] = b
        elif x==0 and t>1:
             #f_hash[(x,t,b)] = 
             f_hash[(x,t,b)] = (1-b) + b*f(0, t-1, b)**2
        elif x>0 and t>0:
             #f_hash[(x,t,b)] = tf.mul(b, sum(tf.mul(f(x-x_, t-1, b), f(x_, t-1, b)) for x_ in range(x+1)))
             f_hash[(x,t,b)] = b*sum(f(x-x_, t-1, b)*f(x_, t-1, b) for x_ in range(x+1))
        else:
             f_hash[(x,t,b)] = 0.
    return f_hash[(x,t,b)]

import random
def simulate(b, t):
    x = 1
    t_ = 0
    while t_ < t and x > 0:
        x = sum(2 if random.random() < b else 0 for _ in range(x))
        t_ += 1
    return x

#import tensorflow as tf
import ad, scipy, scipy.optimize



memo = {}
def power(x, n):
    assert n >= 0 and type(n) == int
    if (x, n) not in memo:
        if n == 0:
            memo[(x, n)] = 1
        else:
            memo[(x, n)] = x*power(x, n-1)
            #memo[(x, n)] = x**n
    return memo[(x, n)]

x = ad.adnumber(1.2, 'x')
y = x**2 #power(x, 2)
print y
print y.gradient([x])



b_true = .7
t = 10
x_ = [simulate(b_true, t) for _ in range(1000)]

#b = tf.Variable(1.)
#y = -sum(f(x,t,tf.sigmoid(b)) for x in x_)

#b = tf.Variable(.5)
#y = -sum(f(x,t,b) for x in x_)

from ad.admath import *
cost = lambda params: -sum(log(f(x,t,*params)) for x in x_)
cost_jac, cost_hes = ad.gh(cost)

print 'gradient error:', scipy.optimize.check_grad(cost, cost_jac, [.5])

print scipy.optimize.minimize(cost, [.5], bounds=((0.001,.999),), jac=cost_jac, tol=1e-9)#, hess=cost_hes)


#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(y)

#sess = tf.Session()
#sess.run(tf.initialize_all_variables())

#print sess.run(b), -sess.run(y)
#for _ in range(1000):
#    sess.run(train_step)
#    print sess.run(b), -sess.run(y)
