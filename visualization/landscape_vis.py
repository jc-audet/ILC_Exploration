import matplotlib.pyplot as plt
import numpy as np
import copy

from scipy.integrate import odeint, solve_ivp
from numpy.linalg import eig

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


def dynamical_sys(f, grad, hack):
    
    def diff(t, u):
        x, y = u

        grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y = grad(x, y)

        avg_grad_x = -grad_Z1_x - grad_Z2_x
        avg_grad_y = -grad_Z1_y - grad_Z2_y

        mask_x = ( np.abs( ( np.sign(grad_Z1_x) + np.sign(grad_Z2_x) ) / 2 ) > 0.9 ).astype(float)
        mask_y = ( np.abs( ( np.sign(grad_Z1_y) + np.sign(grad_Z2_y) ) / 2 ) > 0.9 ).astype(float)
        mask_grad_x = avg_grad_x * mask_x
        mask_grad_y = avg_grad_y * mask_y

        return [mask_grad_x, mask_grad_y]
        # return [mask_grad_x, mask_grad_y]
    
    def AND_x(t, u):
        x, y = u

        grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y = grad(x, y)

        mask_x = ( np.abs( ( np.sign(grad_Z1_x) + np.sign(grad_Z2_x) ) / 2 ) > 0.9 ).astype(float)

        return mask_x - 0.5
    AND_x.terminal = True
    AND_x.direction = -1.0

    def AND_y(t, u):
        x, y = u

        grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y = grad(x, y)

        mask_y = ( np.abs( ( np.sign(grad_Z1_y) + np.sign(grad_Z2_y) ) / 2 ) > 0.9 ).astype(float)

        return mask_y - 0.5
    AND_y.terminal = True
    AND_y.direction = -1.0


    u0 = np.random.uniform(-6,6,2)
    S = np.empty((2,0))
    i=1

    if hack:
        while True:

            sol = solve_ivp(diff, [0.,10.], u0, events=(AND_x, AND_y))

            if np.size(sol.y_events) >= 1:
                if np.size(sol.y_events[0]) !=0:
                    u0 = sol.y_events[0][0]
                else:
                    u0 = sol.y_events[1][0]
                
                x, y = u0

                grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y = grad(x, y)

                avg_grad_x = -grad_Z1_x - grad_Z2_x
                avg_grad_y = -grad_Z1_y - grad_Z2_y
                
                # if np.linalg.norm([grad_Z1_x, grad_Z1_y]) < 1e-4 and np.linalg.norm([grad_Z2_x, grad_Z2_y]) < 1e-4:
                if np.linalg.norm([grad_Z1_x, grad_Z1_y]) / np.linalg.norm([grad_Z2_x, grad_Z2_y]) < 0.05:
                    print("Grad break")
                    break
                else:
                    while AND_x(0., u0) < 0:
                        u0[0] = np.random.uniform(-6,6,1)[0]
                    while AND_y(0., u0) < 0:
                        u0[1] = np.random.uniform(-6,6,1)[0]
            else: 
                print("no stop break")
                break

            print(i)
            i+=1

            S = np.append(S, sol.y, axis = 1)
    else:
        sol = solve_ivp(diff, [0.,10.], u0)
        S = np.append(S, sol.y, axis = 1)

            
    
    ## Define space
    x = np.arange(-6, 6, 0.5)
    y = np.arange(-6, 6, 0.5)

    xx, yy = np.meshgrid(x, y)

    Z1, Z2 = f(xx,yy)
    grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y = grad(x, y)


    # plt.contourf(xx, yy, Z1+Z2, levels = 1000, cmap='jet')
    plt.plot(S[0,:], S[1,:], 'r')



        # if np.abs( ( np.sign(grad_Z1_x) + np.sign(grad_Z2_x) ) / 2 ) > 0.9:
            

        # mask_y = np.abs( ( np.sign(grad_Z1_y) + np.sign(grad_Z2_y) ) / 2 ) > 0.9
        # mask_grad_x = avg_grad_x * mask_x
        # mask_grad_y = avg_grad_y * mask_y


if __name__ == '__main__':

    ## Choose method
    hack = True
    method = 'AND' 
    # method = 'conf'

    ## Choose function
    # f = ILC_paper_f
    # grad = ILC_paper_grad
    # f = home_ez_f
    # grad = home_ez_grad
    f = home_hard_f
    grad = home_hard_grad
    hess = home_hard_hess

    ## Define space
    x = np.arange(-6, 6, 0.5)
    y = np.arange(-6, 6, 0.5)

    x, y = np.meshgrid(x, y)

    w = 0.3

    ## Function
    Z1, Z2 = f(x, y)
    grad_Z1_x, grad_Z1_y, grad_Z2_x, grad_Z2_y = grad(x, y)

    hess_Z1_xx, hess_Z1_xy, hess_Z1_yx, hess_Z1_yy, hess_Z2_xx, hess_Z2_xy, hess_Z2_yx, hess_Z2_yy = hess(x,y)

    hess_Z1 = np.stack([np.stack([hess_Z1_xx, hess_Z1_xy], axis=2), np.stack([hess_Z1_yx, hess_Z1_yy], axis=2)], axis=3)

    hess_Z1_val, hess_Z1_vec = eig(hess_Z1)
    print(np.shape(hess_Z1_vec))

    ## Gradient combination
    avg_grad_x = - grad_Z1_x - grad_Z2_x
    avg_grad_y = - grad_Z1_y - grad_Z2_y

    if method == 'AND':
        mask_x = ( np.abs( ( np.sign(grad_Z1_x) + np.sign(grad_Z2_x) ) / 2 ) > 0.9 ).astype(float)
        mask_y = ( np.abs( ( np.sign(grad_Z1_y) + np.sign(grad_Z2_y) ) / 2 ) > 0.9 ).astype(float)
        mask_grad_x = avg_grad_x * mask_x
        mask_grad_y = avg_grad_y * mask_y

    elif method == 'conf':
        mask_x = np.abs( ( np.sign(grad_Z1_x) + np.sign(grad_Z2_x) ) / 2 )
        mask_y = np.abs( ( np.sign(grad_Z1_y) + np.sign(grad_Z2_y) ) / 2 )
        mask_grad_x = avg_grad_x * mask_x
        mask_grad_y = avg_grad_y * mask_y


    fig = plt.figure()
    plt.contourf(x, y, hess_Z1_val[:,:,0], levels = 1000, cmap='jet')
    plt.streamplot(x, y, hess_Z1_vec[:,:,0,0], hess_Z1_vec[:,:,1,0], density=1.5, color='k', linewidth=1, arrowsize=0.5)
    fig = plt.figure()
    plt.contourf(x, y, hess_Z1_val[:,:,1], levels = 1000, cmap='jet')
    plt.streamplot(x, y, hess_Z1_vec[:,:,0,1], hess_Z1_vec[:,:,1,1], density=1.5, color='k', linewidth=1, arrowsize=0.5)
    # fig = plt.figure()
    # plt.contourf(x, y, Z1+Z2, levels = 1000, cmap='jet')

    fig = plt.figure()
    plt.contourf(x, y, Z1, levels = 1000, cmap='jet')
    plt.streamplot(x, y, -grad_Z1_x, -grad_Z1_y, density=1.5, color='k', linewidth=1, arrowsize=0.5)
    plt.savefig("grad_env_A.png")
    # fig = plt.figure()
    # plt.contourf(x, y, Z2, levels = 1000, cmap='jet')
    # plt.streamplot(x, y, -grad_Z2_x, -grad_Z2_y, density=1.5, color='k', linewidth=1, arrowsize=0.5)
    # plt.savefig("grad_env_B.png")
    # fig = plt.figure()
    # plt.contourf(x, y, Z1+Z2, levels = 1000, cmap='jet')
    # plt.streamplot(x, y, avg_grad_x, avg_grad_y, density=1.5, color='k', linewidth=1, arrowsize=0.5)
    # plt.savefig("grad_avg.png")
    # fig = plt.figure()
    # plt.contourf(x, y, Z1+Z2, levels = 1000, cmap='jet')
    # plt.streamplot(x, y, mask_grad_x, mask_grad_y, density=1.5, color='k', linewidth=1, arrowsize=0.5)
    # plt.savefig("grad_and.png")



    # plt.show()

    plt.figure()
    plt.contourf(x, y, Z1 + Z2, levels = 1000, cmap='jet')
    plt.streamplot(x, y, mask_grad_x, mask_grad_y, density=2, color='k', linewidth=1, arrowsize=0.5)

    # for i in range(200):
    dynamical_sys(f, grad, hack)

    plt.savefig("dyn_batch.png")
    plt.show()