stepSizes = [.001, .01, .1, 1(1.2)]
import numpy as np
grad  = lambda x, y: np.array([2*x, .18*y])
print('Without Momentum')
for j in stepSizes:
    x_0 = np.array([1, 1])
    iters = 1000
    while iters > 0:
        iters -= 1
        x_0 = x_0 - j*grad(x_0[0], x_0[1])

    print(str(j)+': '+str(x_0))
print('With Momentum (Beta = .5)')
for j in stepSizes:
    x_0 = np.array([1, 1])
    B = .1
    z_k = np.array([0, 0])
    iters = 1000
    while iters > 0:
        iters -= 1
        z_k = grad(x_0[0], x_0[1]) + B*z_k
        x_0 = x_0 - j*z_k

    print(str(j)+': '+str(x_0))

