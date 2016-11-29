#! /usr/bin/env python

import tensorflow as tf
#from scipy import array
#from scipy.special import logit, expit
#from scipy.optimize import minimize
from math import log
from collections import Counter
import random, time

def build_tensor(x, t, b, logit=False):
    """generate tensorflow graph, given max x and max t"""
    if logit:
        b = tf.sigmoid(b)
    # t=0 slice
    T = [[1. if x_==1 else 0. for x_ in range(x+1)]]
    # add subsequent time slices
    for t_ in range(1, t+1):
        T.append([  (1-b if x_==0 else 0)
                  + 2*b*sum(T[-1][x_-x__]*T[-1][x__] for x__ in range(x_//2))
                  + (b*T[-1][x_//2]**2 if x_ % 2 == 0 else 0)
                  for x_ in range(x+1)
                 ])
    return tf.pack(T)

memo = {}
def L(x, t, b):
    """recursive likelihood (with gradient), for benchmarking tensorflow"""
    if (x, t, b) not in memo:
        if t==0 and x==1:
            memo[(x, t, b)] = (1., [0.])
        elif t>0 and x>=0:
            f, dfdb = ((1-b, -1) if x==0 else (0, 0))
            for x_ in range(x//2):
                neighbor1_f, neighbor1_gradf = L(x-x_, t-1, b)
                neighbor2_f, neighbor2_gradf = L(x_, t-1, b)
                f += 2*b*neighbor1_f*neighbor2_f
                dfdb += 2*neighbor1_f*neighbor2_f + 2*b*neighbor1_gradf[0]*neighbor2_f + 2*b*neighbor1_f*neighbor2_gradf[0]
            if x % 2 == 0:
                neighbor_f, neighbor_gradf = L(x/2, t-1, b)
                f += b*neighbor_f**2
                dfdb += neighbor_f**2 + b*2*neighbor_f*neighbor_gradf[0]
            memo[(x, t, b)] = (f, [dfdb])
        else:
            memo[(x, t, b)] = (0., [0.])
    return memo[(x, t, b)]

def l(params, x_counter, t, sign=1):
    """log likelihood for list of trees"""
    b = expit(params[0])
    result = 0.
    dresultdb = 0.
    for x in x_counter:
        f, gradf = L(x, t, b)
        print(x,t, f, logit_b)
        result += sign*x_counter[x]*log(f)
        dresultdb += sign*x_counter[x]*gradf[0]/f
    return result, array([dresultdb])

def simulate(b, t):
    """simulate tree"""
    x = 1
    t_ = 0
    while t_ < t and x > 0:
        x = sum(2 if random.random() < b else 0 for _ in range(x))
        t_ += 1
    return x

def main():
    b_true = .7
    t = 5
    ntrees = 100
    x_counter = Counter([simulate(b_true, t) for _ in range(ntrees)])
    print('\nsimulating {0} trees run for {1} time steps with branching probability {2:.2f}'.format(ntrees, t, b_true))
    print('\nbranching probability inference results')

    timer = time.time()
    result = minimize(l, (logit(.5),), args=(x_counter, t, -1), jac=True, tol=.001)
    print('\n         scipy: {0:.2f} ({1:.2f} sec)'.format(expit(result.x[0]), time.time() - timer))
    assert result.success

    # now let's see how long tensorflow takes to make the likelihood this small
    timer = time.time()
    logit_b = tf.Variable(0.)
    x_max = max(x_counter.keys())
    T = build_tensor(x_max, t, logit_b, logit=True)
    # now we get an index array into T[t,:], based on the data
    w = tf.constant([x_counter[x] for x in range(x_max+1)], dtype=tf.float32)
    #y = -tf.reduce_sum(tf.multiply(w, tf.log(T[t,:])))
    y = -tf.einsum('i,i->', w, tf.log(T[t,:]))
    #y = -sum(x_counter[x]*tf.log(T[t,x]) for x in x_counter)
    time_build = time.time() - timer
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(y)
    #while sess.run(y) > result.fun:
    for _ in range(100):
        sess.run(train_step)
    time_run = time.time() - timer - time_build
    print('    tensorflow: {0:.2f} ({1:.2f} sec to build, {2:.2f} sec to run)\n'.format(expit(sess.run(logit_b)), time_build, time_run))

if __name__ == "__main__":
    main()
