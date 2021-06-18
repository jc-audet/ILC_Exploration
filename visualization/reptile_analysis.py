import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig

import copy

from scipy.integrate import odeint, solve_ivp
from scipy.stats import norm



def ILC_paper_f(x, y):

    ## Define parameters
    w=0.3

    ## Define landscape
    # Z1 = -200*np.cos(w*x)*np.cos(w*y)*np.exp(-w*(x+4)**2-w*(y+4)**2) + (x+y-2)**2
    Z1 = -200*np.cos(w*x)*np.cos(w*y)*np.exp(-w*(x+4)**2-w*(y+4)**2) + (x-2)**2
    Z2 = -200*np.cos(w*x)*np.cos(w*y)*np.exp(-w*(x+4)**2-w*(y+4)**2) + (y-2)**2

    return Z1, Z2

def ILC_paper_grad(x, y):

    ## Define parameters
    w=0.3

    ## Define grads
    grad_Z1_x = 200*( w * np.cos(w*y) * np.exp(-w*(x+4)**2 - w*(y+4)**2) * ( np.sin(w*x) + 2*(x+4)*np.cos(w*x) ) ) + 2*(x-2)
    grad_Z1_y = 200*( w * np.cos(w*x) * np.exp(-w*(x+4)**2 - w*(y+4)**2) * ( np.sin(w*y) + 2*(y+4)*np.cos(w*y) ) )
    grad_Z2_x = 200*( w * np.cos(w*y) * np.exp(-w*(x+4)**2 - w*(y+4)**2) * ( np.sin(w*x) + 2*(x+4)*np.cos(w*x) ) )
    grad_Z2_y = 200*( w * np.cos(w*x) * np.exp(-w*(x+4)**2 - w*(y+4)**2) * ( np.sin(w*y) + 2*(y+4)*np.cos(w*y) ) ) + 2*(y-2)

    return grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y

def ILC_paper_grad_noise(x, y):

    ## Define parameters
    w=0.3
    sig = 3

    ## Define grads
    grad_Z1_x = (200*( w * np.cos(w*y) * np.exp(-w*(x+4)**2 - w*(y+4)**2) * ( np.sin(w*x) + 2*(x+4)*np.cos(w*x) ) ) + 2*(x-2)) + np.random.normal(0, sig, np.shape(x))
    grad_Z1_y = (200*( w * np.cos(w*x) * np.exp(-w*(x+4)**2 - w*(y+4)**2) * ( np.sin(w*y) + 2*(y+4)*np.cos(w*y) ) )) + np.random.normal(0, sig, np.shape(x))
    grad_Z2_x = (200*( w * np.cos(w*y) * np.exp(-w*(x+4)**2 - w*(y+4)**2) * ( np.sin(w*x) + 2*(x+4)*np.cos(w*x) ) )) + np.random.normal(0, sig, np.shape(x))
    grad_Z2_y = (200*( w * np.cos(w*x) * np.exp(-w*(x+4)**2 - w*(y+4)**2) * ( np.sin(w*y) + 2*(y+4)*np.cos(w*y) ) ) + 2*(y-2)) + np.random.normal(0, sig, np.shape(x))

    return grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y

def home_hard_f(x, y):
    
    ## Define parameters
    w = 0.3

    ## Define landscape
    Z1 = -100*np.cos(w*x)*np.cos(w*y)*np.exp(-w*(x)**2-w*(y-4)**2) + (x+2.5)**2 + (y+2.5)**2
    Z2 = -100*np.cos(w*x)*np.cos(w*y)*np.exp(-w*(x)**2-w*(y-4)**2) + (x-2.5)**2 + (y+2.5)**2

    return Z1, Z2

def home_hard_grad(x, y):

    ## Define parameters
    w = 0.3

    ## Define grads
    grad_Z1_x = 100*( w * np.cos(w*y) * np.exp(-w*(x)**2 - w*(y-4)**2) * ( np.sin(w*x) + 2*(x)*np.cos(w*x) ) ) + 2*(x+2.5)
    grad_Z1_y = 100*( w * np.cos(w*x) * np.exp(-w*(x)**2 - w*(y-4)**2) * ( np.sin(w*y) + 2*(y-4)*np.cos(w*y) ) ) + 2*(y+2.5)
    grad_Z2_x = 100*( w * np.cos(w*y) * np.exp(-w*(x)**2 - w*(y-4)**2) * ( np.sin(w*x) + 2*(x)*np.cos(w*x) ) ) + 2*(x-2.5)
    grad_Z2_y = 100*( w * np.cos(w*x) * np.exp(-w*(x)**2 - w*(y-4)**2) * ( np.sin(w*y) + 2*(y-4)*np.cos(w*y) ) ) + 2*(y+2.5)

    return grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y

def home_hard_grad_noise(x, y):

    ## Define parameters
    w = 0.3
    sig = 1

    ## Define grads
    grad_Z1_x = 100*( w * np.cos(w*y) * np.exp(-w*(x)**2 - w*(y-4)**2) * ( np.sin(w*x) + 2*(x)*np.cos(w*x) ) ) + 2*(x+2.5) + np.random.normal(0, sig, np.shape(x))
    grad_Z1_y = 100*( w * np.cos(w*x) * np.exp(-w*(x)**2 - w*(y-4)**2) * ( np.sin(w*y) + 2*(y-4)*np.cos(w*y) ) ) + 2*(y+2.5) + np.random.normal(0, sig, np.shape(x))
    grad_Z2_x = 100*( w * np.cos(w*y) * np.exp(-w*(x)**2 - w*(y-4)**2) * ( np.sin(w*x) + 2*(x)*np.cos(w*x) ) ) + 2*(x-2.5) + np.random.normal(0, sig, np.shape(x))
    grad_Z2_y = 100*( w * np.cos(w*x) * np.exp(-w*(x)**2 - w*(y-4)**2) * ( np.sin(w*y) + 2*(y-4)*np.cos(w*y) ) ) + 2*(y+2.5) + np.random.normal(0, sig, np.shape(x))

    return grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y

def home_hard_hess(x, y):

    ## Define parameters
    w = 0.3

    ## Define hesss
    hess_Z1_xx = - 200 * w**2 * x * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*y) * (np.sin(w*x) + 2*x*np.cos(w*x)) 
    + 100 * w * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*y) * (-2*w*x*np.sin(w*x) + w*np.cos(w*x) + 2*np.cos(w*x)) + 2
    hess_Z1_xy = - 200 * w**2 * (y-4) * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*y) * (np.sin(w*x) + 2*x*np.cos(w*x))
    - 100 * w**2 * np.exp(-w*x**2 - w*(y-4)**2) * np.sin(w*y) * (np.sin(w*x) + 2*x*np.cos(w*x))
    hess_Z1_yx = - 200 * w**2 * x * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*x) * (np.sin(w*y) + 2*(y-4)*np.cos(w*y)) 
    - 100 * w**2 * np.exp(-w*x**2 - w*(y-4)**2) * np.sin(w*x) * (np.sin(w*y) + 2*(y-4)*np.cos(w*y))
    hess_Z1_yy = - 200 * w**2 * (y-4) * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*x) * (np.sin(w*y) + 2*(y-4)*np.cos(w*y)) 
    + 100 * w * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*x) * (-2*w*(y-4)*np.sin(w*y) + w*np.cos(w*y) + 2*np.cos(w*y)) + 2
    hess_Z2_xx = - 200 * w**2 * x * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*y) * (np.sin(w*x) + 2*x*np.cos(w*x)) 
    + 100 * w * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*y) * (-2*w*x*np.sin(w*x) + w*np.cos(w*x) + 2*np.cos(w*x)) + 2

    hess_Z2_xy = - 200 * w**2 * (y-4) * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*y) * (np.sin(w*x) + 2*x*np.cos(w*x)) 
    - 100 * w**2 * np.exp(-w*x**2 - w*(y-4)**2) * np.sin(w*y) * (np.sin(w*x) + 2*x*np.cos(w*x))

    hess_Z2_yx = - 200 * w**2 * x * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*x) * (np.sin(w*y) + 2*(y-4)*np.cos(w*y)) 
    - 100 * w**2 * np.exp(-w*x**2 - w*(y-4)**2) * np.sin(w*x) * (np.sin(w*y) + 2*(y-4)*np.cos(w*y))
    hess_Z2_yy = - 200 * w**2 * (y-4) * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*x) * (np.sin(w*y) + 2*(y-4)*np.cos(w*y)) 
    + 100 * w * np.exp(-w*x**2 - w*(y-4)**2) * np.cos(w*x) * (-2*w*(y-4)*np.sin(w*y) + w*np.cos(w*y) + 2*np.cos(w*y)) + 2

    return hess_Z1_xx, hess_Z1_xy, hess_Z1_yx, hess_Z1_yy, hess_Z2_xx, hess_Z2_xy, hess_Z2_yx, hess_Z2_yy

def home_ez_f(x, y):
    
    ## Define parameter
    w = 0.3

    ## Define landscape
    Z1 = -100*np.cos(w*x)*np.cos(w*y)*np.exp(-w*(x-2.5)**2-w*(y+2.5)**2) + (x)**2 + (y-4)**2
    Z2 = -100*np.cos(w*x)*np.cos(w*y)*np.exp(-w*(x+2.5)**2-w*(y+2.5)**2) + (x)**2 + (y-4)**2

    return Z1, Z2

def home_ez_grad(x, y):

    ## Defined grads
    grad_Z1_x = 100*( w * np.cos(w*y) * np.exp(-w*(x-2.5)**2 - w*(y+2.5)**2) * ( np.sin(w*x) + 2*(x-2.5)*np.cos(w*x) ) ) + 2*(x)
    grad_Z1_y = 100*( w * np.cos(w*x) * np.exp(-w*(x-2.5)**2 - w*(y+2.5)**2) * ( np.sin(w*y) + 2*(y+2.5)*np.cos(w*y) ) ) + 2*(y-4)
    grad_Z2_x = 100*( w * np.cos(w*y) * np.exp(-w*(x+2.5)**2 - w*(y+2.5)**2) * ( np.sin(w*x) + 2*(x+2.5)*np.cos(w*x) ) ) + 2*(x)
    grad_Z2_y = 100*( w * np.cos(w*x) * np.exp(-w*(x+2.5)**2 - w*(y+2.5)**2) * ( np.sin(w*y) + 2*(y+2.5)*np.cos(w*y) ) ) + 2*(y-4)

    return grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y

def avg_grad(x, y, grad):

    grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y = grad(x, y)
    avg_grad_x = -grad_Z1_x - grad_Z2_x
    avg_grad_y = -grad_Z1_y - grad_Z2_y

    return [avg_grad_x, avg_grad_y]

def reptile(x, y, grad, lr):

    k = 20

    x_1, y_1 = copy.deepcopy(x), copy.deepcopy(y)
    x_2, y_2 = copy.deepcopy(x), copy.deepcopy(y)
    for i in range(k):

        grad_Z1_x, grad_Z1_y, _, _ = grad(x_1, y_1)
        x_1, y_1 = x_1 - lr*grad_Z1_x, y_1 - lr*grad_Z1_y
        _, _, grad_Z2_x, grad_Z2_y = grad(x_2, y_2)
        x_2, y_2 = x_2 - lr*grad_Z2_x, y_2 - lr*grad_Z2_y

    return [((x_1 - x) + (x_2 - x)) / float(k), ((y_1 - y) + (y_2 - y)) / float(k)]

def dynamical_sys(x0, y0, f, grad, method, lr, m):

    # Initialization
    S = np.empty((2,0))
    S = np.append(S, np.array([x0, y0]), axis=1)

    x, y = x0, y0
    v_x, v_y = 0, 0
    for i in range(100):
        grad_x, grad_y = method(x, y, grad, lr)
        v_x = m*v_x - 0.25*grad_x
        v_y = m*v_y - 0.25*grad_y
        x = x - v_x
        y = y - v_y
        S = np.append(S, np.array([x, y]), axis=1)

    plt.plot(S[0,:], S[1,:], 'r')
    plt.scatter(x0, y0, c='k')
    plt.scatter(x, y, c='r')

def dynamical_sys_init(x0, y0, f, grad, method, lr, m):

    x, y = x0, y0
    v_x, v_y = 0, 0
    for i in range(150):
        grad_x, grad_y = method(x, y, grad)
        v_x = m*v_x - lr*grad_x
        v_y = m*v_y - lr*grad_y
        x = x - v_x
        y = y - v_y

    # print("(x,y) = (" + str(x) + ", " + str(y) + ")")
    if np.sqrt(np.power(x+3.125, 2) + np.power(y+3.125, 2)) < 0.5:
        return [1, 1]
    elif np.sqrt(np.power(x-2, 2) + np.power(y-2, 2)) < 1:
        return [1, 0]
    else:
        return [0, 0]


if __name__ == '__main__':

    #########################################################
    ## Choose vector vizualization
    #########################################################
    strmplt = True

    #########################################################
    ## Choose method 
    #########################################################
    # method = avg_grad
    # method_name = "avg"
    method = reptile
    method_name = "reptile"
    # method = geom_mean
    # method_name = "geom"
    # method = geom_AND


    #########################################################
    ## Learning parameter
    #########################################################
    lr = 0.01
    m = 0
    SGD = False

    #########################################################
    ## Choose function
    #########################################################
    f = ILC_paper_f
    grad = ILC_paper_grad
    # f = home_ez_f
    # grad = home_ez_grad
    # f = home_hard_f
    # grad = home_hard_grad
    # hess = home_hard_hess

    #########################################################
    ## Define space
    #########################################################
    x = np.arange(-6, 6, 0.2)
    y = np.arange(-6, 6, 0.2)

    x, y = np.meshgrid(x, y)

    #########################################################
    ## Function
    #########################################################
    Z1, Z2 = f(x, y)
    grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y = grad(x, y)

    #########################################################
    ## Gradient combination
    #########################################################
    avg_grad_x = - grad_Z1_x - grad_Z2_x
    avg_grad_y = - grad_Z1_y - grad_Z2_y

    grad_x, grad_y = method(x, y, grad, 0.01)

    #########################################################
    ## Hessian ridges vizualization
    #########################################################
    # hess_Z1_xx, hess_Z1_xy, hess_Z1_yx, hess_Z1_yy, hess_Z2_xx, hess_Z2_xy, hess_Z2_yx, hess_Z2_yy = hess(x,y)
    # hess_Z1 = np.stack([np.stack([hess_Z1_xx, hess_Z1_xy], axis=2), np.stack([hess_Z1_yx, hess_Z1_yy], axis=2)], axis=3)
    # hess_Z1_val, hess_Z1_vec = eig(hess_Z1)
    
    # fig = plt.figure()
    # plt.contourf(x, y, hess_Z1_val[:,:,0], levels = 1000, cmap='jet')
    # plt.streamplot(x, y, hess_Z1_vec[:,:,0,0], hess_Z1_vec[:,:,1,0], density=1.5, color='k', linewidth=1, arrowsize=0.5)
    # fig = plt.figure()
    # plt.contourf(x, y, hess_Z1_val[:,:,1], levels = 1000, cmap='jet')
    # plt.streamplot(x, y, hess_Z1_vec[:,:,0,1], hess_Z1_vec[:,:,1,1], density=1.5, color='k', linewidth=1, arrowsize=0.5)
    # fig = plt.figure()
    # plt.contourf(x, y, Z1 + Z2, levels = 1000, cmap='jet')

    #########################################################
    ## Expected gradient field with AND-Mask
    #########################################################
    # diff_grad_x = np.minimum(np.abs(grad_Z1_x), np.abs(grad_Z2_x))
    # diff_grad_y = np.minimum(np.abs(grad_Z1_y), np.abs(grad_Z2_y))

    # sign_trick = np.sign(np.abs(grad_Z1_x) - np.abs(grad_Z2_x))
    # max_grad_x = grad_Z1_x * (sign_trick+1)/2
    # max_grad_x += grad_Z2_x * np.abs(sign_trick-1)/2
    # sign_trick = np.sign(np.abs(grad_Z1_y) - np.abs(grad_Z2_y))
    # max_grad_y = grad_Z1_y * (sign_trick+1)/2
    # max_grad_y += grad_Z2_y * np.abs(sign_trick-1)/2

    # p_x = np.zeros(np.shape(diff_grad_x))
    # p_y = np.zeros(np.shape(diff_grad_y))
    # for i in range(np.shape(diff_grad_x)[0]):
    #     for j in range(np.shape(diff_grad_x)[1]):
    #         p_x[i,j] = n.cdf(diff_grad_x[i,j])
    #         p_y[i,j] = n.cdf(diff_grad_y[i,j])

    # E_x = p_x * max_grad_x
    # E_y = p_y * max_grad_y

    # fig = plt.figure()
    # plt.contourf(x, y, Z1+Z2, levels = 1000, cmap='jet')
    # plt.quiver(x, y, diff_grad_x, diff_grad_y, density=1.5, color='k', linewidth=1, arrowsize=0.5)
    # fig = plt.figure()
    # heat = plt.contourf(x, y, p_x, levels = 1000, cmap='jet')
    # plt.quiver(x, y, -max_grad_x, np.zeros(np.shape(max_grad_x)), color='k')
    # plt.title("X component visualization")
    # plt.colorbar(heat)
    # plt.savefig("prob_viz_x.png")


    # fig = plt.figure()
    # heat = plt.contourf(x, y, p_y, levels = 1000, cmap='jet')
    # plt.quiver(x, y, np.zeros(np.shape(max_grad_y)), -max_grad_y, color='k')
    # plt.title("Y component visualization")
    # plt.colorbar(heat)
    # plt.savefig("prob_viz_y.png")

    # fig = plt.figure()
    # plt.contourf(x, y, Z1+Z2, levels = 1000, cmap='jet')
    # plt.streamplot(x, y, -E_x, -E_y, density=1.5, color='k', linewidth=1, arrowsize=0.5)
    # plt.title("Expected gradients")
    # plt.savefig("grad_expectancy.png")


    #########################################################
    ## Environment Grad visualization (Single, avg)
    #########################################################
    # fig = plt.figure()
    # plt.contourf(x, y, Z1, levels = 1000, cmap='jet')
    # plt.streamplot(x, y, -grad_Z1_x, -grad_Z1_y, density=1.5, color='k', linewidth=1, arrowsize=0.5)
    # plt.savefig("grad_env_A.png")
    # fig = plt.figure()
    # plt.contourf(x, y, Z2, levels = 1000, cmap='jet')
    # plt.streamplot(x, y, -grad_Z2_x, -grad_Z2_y, density=1.5, color='k', linewidth=1, arrowsize=0.5)
    # plt.savefig("grad_env_B.png")
    fig = plt.figure()
    plt.contourf(x, y, Z1+Z2, levels = 1000, cmap='jet')
    if strmplt:
        plt.streamplot(x, y, avg_grad_x, avg_grad_y, density=1.5, color='k', linewidth=1, arrowsize=0.5)
    else:
        plt.quiver(x, y, avg_grad_x, avg_grad_y, color='k')
    plt.title("Average gradients")
    plt.savefig("grad_avg.png")

    plt.figure()
    plt.contourf(x, y, Z1 + Z2, levels = 1000, cmap='jet')
    if strmplt:
        plt.streamplot(x, y, grad_x, grad_y, density=2, color='k', linewidth=1, arrowsize=0.5)
    else:
        plt.quiver(x, y, grad_x, grad_y, color='k')
    plt.savefig("grad_"+str(method_name)+".png")


    #########################################################
    ## Gradient descent visualization
    #########################################################
    if SGD:
        grad = ILC_paper_grad_noise
        # grad = home_hard_grad_noise

    plt.figure()
    # plt.contourf(x, y, np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2)), levels = 1000, cmap='jet')
    plt.contourf(x, y, Z1 + Z2, levels = 1000, cmap='jet')
    if strmplt:
        plt.streamplot(x, y, grad_x, grad_y, density=2, color='k', linewidth=1, arrowsize=0.5)
    else:
        plt.quiver(x, y, grad_x, grad_y, color='k')


    x0, y0 = np.array([[-4], [4]])
    # x0, y0 = np.random.uniform(-6,6,(2,1))

    dynamical_sys(x0, y0, f, grad, method, lr, m)

    # plt.savefig("dyn_batch_noise_hard_3.png")
    plt.show()
