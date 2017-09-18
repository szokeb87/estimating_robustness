import numpy as np
import scipy as sp
import scipy.linalg as la
from scipy.stats import lognorm
from scipy.interpolate import interp1d

import quantecon as qe
import control as cont
import matplotlib.pyplot as plt
import seaborn as sb

from robust_utils import *


class Robust(object):
    """
    Insert clever documentation here
    """

    def __init__(self, phi, kappa, sigma, beta0, beta1, alpha,\
                 xi0, xi1, xi2, delta, x0 = None, zeta = None):

        self.phi, self.kappa, self.sigma = list(map(self.convert, (phi, kappa, sigma)))
        self.beta0, self.beta1, self.alpha = list(map(self.convert, (beta0, beta1, alpha)))
        self.delta = delta

        self.N, self.K = self.sigma.shape
        self.M = self.alpha.shape[0]

        if zeta == None:
            self.zeta_0 = np.zeros((self.K, 1))
            self.zeta_1 = np.zeros((self.K, self.N))
        else:
            zeta = np.asarray(zeta)
            self.zeta_0 = zeta[:self.K].reshape(self.K, 1)
            self.zeta_1 = zeta[self.K:].reshape(self.K, self.N)

        self.phi_bar = self.phi + self.sigma @ self.zeta_0
        self.kappa_bar = self.kappa - self.sigma @ self.zeta_1
        self.beta0_bar = self.beta0 + self.alpha @ self.zeta_0
        self.beta1_bar = self.beta1 + self.alpha @ self.zeta_1

        self.xi_0, self.xi_1, xi_2 = list(map(self.convert, (xi0, xi1, xi2)))
        self.xi_2 = (.5)*(xi_2 + xi_2.T)  # make sure that it's symmetric (need for the care)

        xiA = np.hstack((self.xi_0, np.zeros((1, self.N))))
        xiB = np.hstack((2*self.xi_1, self.xi_2))
        Xi = np.vstack((xiA, xiB))
        self.Xi = (.5)*(Xi + Xi.T)        # make sure that it is symmetric (to check positive semi-d with eigenvals)

        #if not all(np.linalg.eigvals(self.Xi) >= 0):
        #    raise ValueError("The function xi() must be nonnegative, but matrix Xi is not positive semi-definite")

        #==================================================================
        # Calculate the exact stationary distributions for X:
        #==================================================================

        #------------------------------------------------------------------
        # (1) Benchmark model
        #------------------------------------------------------------------

        KK = np.kron(self.kappa, np.eye(self.N)) + np.kron(np.eye(self.N), self.kappa)
        self.SigmaX_inf = (la.inv(KK) @ (self.sigma @ self.sigma.T).flatten('F')).reshape(self.N, self.N)
        self.MuX_inf = la.inv(self.kappa) @ self.phi

        # Define the initial state
        if x0 is None:
            self.x0 = self.MuX_inf
        else:
            self.x0 = np.atleast_2d(x0).reshape(self.N, 1)

        # Calculate the elements of the value function
        self.v2, self.v_11, self.v_10, self.v_01, self.v_00, self.v_0m1, \
            self.theta, self.eta_0, self.eta_1 = self.valuefunc()

        # Calculate the parameters under the worst-case model
        self.beta0_tilde = self.beta0 + self.alpha @ self.eta_0
        self.beta1_tilde = self.beta1 + self.alpha @ self.eta_1
        self.phi_tilde = self.phi + self.sigma @ self.eta_0
        self.kappa_tilde = self.kappa - self.sigma @ self.eta_1

        #------------------------------------------------------------------
        # (2) Worst-case model
        #------------------------------------------------------------------

        KK_tilde = np.kron(self.kappa_tilde, np.eye(self.N)) + np.kron(np.eye(self.N), self.kappa_tilde)
        self.SigmaX_inf_tilde = (la.inv(KK_tilde) @ (self.sigma @ self.sigma.T).flatten('F')).reshape(self.N, self.N)
        self.MuX_inf_tilde = la.inv(self.kappa_tilde) @ self.phi_tilde

        self.MuX_inf_bar = la.inv(self.kappa_bar) @ self.phi_bar



        # For the likelihood function and the LSS, store the "discrete time" coefficients
        # ... multivariate OU-process sampled at dicrete points -> VAR(1)
        #self.phi_D, self.kappa_D, self.sigma_D, self.beta0_D, self.beta1_D, self.alpha_D, self.SigmaX_D_inf =
        #    self.from_cont_to_discrete()
        # worstcase counterparts
        #self.phi_D_tilde, self.kappa_D_tilde, self.sigma_D_tilde, self.beta0_D_tilde, self.beta1_D_tilde, \
        #    self.alpha_D_tilde, self.SigmaX_D_tilde_inf = self.from_cont_to_discrete(worstcase=1)

        # Store the yield curves as functions (of yield-to-maturity)
        #self.a, self.b1, self.b2 = self.generate_ab()
        #self.a_tilde, self.b1_tilde, self.b2_tilde = self.generate_ab(worstcase=1)

        # Construct BIG state space representation: quant-econ lss class with
        # state vector: [1 X^T logY]
        #self.lss, self.lss_tilde = self.construct_ss()


    def convert(self, x):
        """
        Convert array_like objects (lists of lists, floats, etc.) into
        well formed 2D NumPy arrays
        """
        return np.atleast_2d(np.asarray(x, dtype='float'))


    def from_cont_to_discrete(self, tau=1, worstcase = 0):
        """
        Construct the discrete-time coefficients from the original continuous time params
        tau : period length between skip sampling (quarters)
        """

        N, M, K = self.N, self.M, self.K
        sigma, alpha =  self.sigma, self.alpha

        if worstcase == 0:
            phi, kappa, beta0, beta1 = self.phi, self.kappa, self.beta0, self.beta1

        elif worstcase == 1:
            phi, kappa, beta0, beta1 = self.phi_tilde, self.kappa_tilde, self.beta0_tilde, self.beta1_tilde

        elif worstcase == 2:
            phi, kappa, beta0, beta1 = self.phi_bar, self.kappa_bar, self.beta0_bar, self.beta1_bar

        kappa_inv = la.inv(kappa)
        mu = kappa_inv @ phi
        vecSigma = (sigma @ sigma.T).flatten('F')
        KK = np.kron(kappa, np.eye(N)) + np.kron(np.eye(N), kappa)

        #=================================================================================
        # Coefficients for the Markov state process: VAR(1)
        #=================================================================================
        #       X_{t+tau} = phi_D + kappa_D X_{t} + sigma_D Z_{t+tau}

        kappa_D = la.expm(-kappa*tau)
        resolv = np.eye(N) - kappa_D
        phi_D = resolv @ mu
        sigma_D = sigma
        sigma2_D = (la.inv(KK) @ (np.eye(N*N) - la.expm(-KK*tau)) @ vecSigma).reshape(N, N)

        try:
            sigma_D = np.linalg.cholesky(sigma2_D)
        except np.linalg.LinAlgError:
            raise ValueError("Sigma2_D is not positive definite")

        # discrete time stationary covariance matrix of X
        SigmaX_D_inf = (la.inv(np.eye(N*N) - np.kron(kappa_D, kappa_D)) @ (sigma_D @ sigma_D.T).flatten('F')).reshape(N, N)


        #=================================================================================
        # Coefficients for the additive functional vector:
        #=================================================================================
        #       DlogY_{t+tau} = beta0_D + beta1_D X_{t} + alpha_D Z_{t+tau}

        beta0_D = (beta0 + beta1 @ mu)*tau - beta1 @ kappa_inv @ resolv @ mu
        beta1_D = beta1 @ kappa_inv @ resolv

#        const = beta1 @ kappa_inv @ sigma + alpha
#        integ = kappa_inv @ resolv
#        integ_transpose = (np.eye(N) - la.expm(-kappa.T*tau)) @ kappa_inv.T
#        Q_c = kappa_inv @ sigma @ sigma.T @ kappa_inv.T
#        vecQ = Q_c.flatten('F')

#        first_term = (const @ const.T)*tau
#        second_term = - const @ (sigma.T @ kappa_inv.T @ integ_transpose @ beta1.T)
#        third_term =  - (beta1 @ integ @ kappa_inv @ sigma) @ const.T
#        fourth_term = beta1 @ (la.inv(KK) @ (np.eye(N*N) - la.expm(-KK*tau)) @ vecQ).reshape(N,N) @ beta1.T

#        alpha2_D = first_term + second_term + third_term + fourth_term

#        try:
#            alpha_D = np.linalg.cholesky(alpha2_D)
#        except np.linalg.LinAlgError:
#            raise ValueError("alpha2_D is not positive definite")
        alpha_D = alpha

        return phi_D, kappa_D, sigma_D, beta0_D, beta1_D, alpha_D, SigmaX_D_inf
#        kappa_D = np.eye(N) - kappa
#        SigmaX_D_inf = (la.inv(np.eye(N*N) - np.kron(kappa_D, kappa_D)) @ (sigma @ sigma.T).flatten('F')).reshape(N, N)
#        return phi, kappa_D, sigma, beta0, beta1, alpha, SigmaX_D_inf

    def construct_ss(self, YTM):
        """
        This creates the discrete time state space representation that can be passed
        into the quantecon LSS class.
        """

        N, M, K = self.N, self.M, self.K

        nx0r, nx0c = np.zeros(N), np.zeros((N, 1))
        ny0r, ny0c = np.zeros(M), np.zeros((M, 1))
        nxy0 = np.zeros((N, M))
        nz0r = np.zeros(K)
        nyy1 = np.eye(M)

        # discrete time coefficients
        phi_D, kappa_D, sigma_D, beta0_D, beta1_D, alpha_D, SigmaX_D_inf = self.from_cont_to_discrete()
        # worstcase counterparts
        phi_D_tilde, kappa_D_tilde, sigma_D_tilde, beta0_D_tilde, beta1_D_tilde, alpha_D_tilde,\
                SigmaX_D_tilde_inf = self.from_cont_to_discrete(worstcase = 1)

        # Build A matrix for LSS: Order of states is: [1, X_t, logYt]
        A1 = np.hstack([1, nx0r, ny0r])                                  # Transition for 1
        A2 = np.hstack([phi_D, kappa_D, nxy0])                           # Transition for X_{t+1}
        A3 = np.hstack([beta0_D, beta1_D, nyy1])                         # Transition for logY_{t+1}
        Abar = np.vstack([A1, A2, A3])

        # under the worst case model
        A1_tilde = np.hstack([1, nx0r, ny0r])                            # Transition for 1
        A2_tilde = np.hstack([phi_D_tilde, kappa_D_tilde, nxy0])         # Transition for X_{t+1}
        A3_tilde = np.hstack([beta0_D_tilde, beta1_D_tilde, nyy1])       # Transition for logY_{t+1}
        Abar_tilde = np.vstack([A1_tilde, A2_tilde, A3_tilde])

        # Build B matrix for LSS
        Bbar = np.vstack([nz0r, sigma_D, alpha_D])
        Bbar_tilde = np.vstack([nz0r, sigma_D_tilde, alpha_D_tilde])

        # Build G matrix for LSS: Order of observation is: [Xt, logYt, y, y_wc, ]
        G1 = np.hstack([nx0c, np.eye(N), nxy0])                          # Selector for x_{t}
        G2 = np.hstack([ny0c, nxy0.T, np.eye(M)])                        # Selector for y_{t}
        G3 = self.yield_matrix(YTM)                                      # some zero coupon yields under the benchmark
        G4 = self.yield_matrix(YTM, worstcase = 1)                       # some zero coupon yields under the WC
        G5 = np.hstack([Bbar @ self.eta_0, Bbar @ self.eta_1,\
                        np.zeros((Bbar.shape[0], M))])[1:, :]            # the worst-case distortion B @ H
        Gbar = np.vstack([G1, G2, G3, G4, G5])

        # Build H matrix for LSS
        Hbar = np.zeros((Gbar.shape[0], K))

        # Initial conditions
        x0 = np.hstack([1, nx0r, ny0r])
        S0 = np.zeros((1 + N + M, 1 + N + M))
        #S0[1:(1+N), 1:(1+N)] = SigmaX_D_inf

        x0_tilde = np.hstack([1, (la.inv(np.eye(N) - kappa_D_tilde) @ phi_D_tilde).squeeze(), ny0r])
        S0_tilde = np.zeros((1 + N + M, 1 + N + M))
        #S0_tilde[1:(1+N), 1:(1+N)] = SigmaX_D_tilde_inf

        lss = qe.lss.LinearStateSpace(Abar, Bbar, Gbar, Hbar, mu_0 = x0, Sigma_0 = S0)
        lss_tilde = qe.lss.LinearStateSpace(Abar_tilde, Bbar_tilde, Gbar, Hbar,
                                            mu_0 = x0_tilde, Sigma_0 = S0_tilde)

        return lss, lss_tilde


    def valuefunc(self):

        N, M, K = self.N, self.M, self.K
        phi, kappa, beta0, beta1 = self.phi, self.kappa, self.beta0, self.beta1
        alpha, sigma = self.alpha, self.sigma
        xi_0, xi_1, xi_2 = self.xi_0, self.xi_1, self.xi_2
        delta = self.delta

        #try:
        # (1) Calculate v2 which gives v_2(theta) after getting multiplied by theta
        A2 = (-kappa  - (delta / 2) * np.eye(N)) #/ 100
        B2 = (np.sqrt(2) * sigma) #/ 100
        Q2 = -xi_2 * .5

        v2, f, g = cont.care(A2, B2, Q2)
        v2 = v2 #* 100

        # (2) Calculate the coefficients for v1
        L_inv = la.inv(delta * np.eye(N) - (-kappa).T + 2*(v2 @ sigma @ sigma.T))
        v_11 = L_inv @ (v2 @ phi - (.5)*xi_1 )
        v_10 = L_inv @ ( (.5)*beta1.T - v2 @ sigma @ alpha.T )[:, 0] / delta
        v_10 = np.expand_dims(v_10, 1)

        # (3) Calculate the coefficients for v0
        v_01 = np.asscalar((1/delta)*(2*phi.T @ v_11 + (.5)*np.trace(sigma.T @ v2 @ sigma) - \
                            (.5)*xi_0 - 2*v_11.T @ sigma @ sigma.T @ v_11))
        v_00 = np.asscalar((1/delta)*(2*phi.T @ v_10 + (1/delta)*(beta0 - 2*alpha @ sigma.T @ v_11)[0,:] - \
                            4*v_10.T @ sigma @ sigma.T @ v_11 ))
        v_0m1 = np.asscalar((-1/delta)*((1/(2*delta**2))*(alpha @ alpha.T)[0,0] + \
                            (2/delta)*(alpha @ sigma.T @ v_10)[0] + 2*v_10.T @ (sigma @ sigma.T) @ v_10) )

        x_0 = self.x0
        denum = v_01 + 2*v_11.T @ x_0 + x_0.T @ v2 @ x_0
        if denum != 0:
            theta2 = v_0m1/denum
        elif denum == 0:
            theta2 = np.inf
        theta = np.sqrt(theta2)

        eta_0 = -(1/theta)*(np.expand_dims(alpha[0, :], 1)/delta + 2*sigma.T @ v_10) - 2*sigma.T @ v_11
        eta_1 = - 2*sigma.T @ v2

        return v2, v_11, v_10, v_01, v_00, v_0m1, theta, eta_0, eta_1

#        except ValueError:
#            raise ValueError("Q is not positive semi-definte, so the Xi matrix cannot be too big!")


    def print_out_params(self, discrete = 0):
        '''
        Print out the parameters and other objects of interests along with their worst-case counterparts
        if discrete == 1 -> plot the discrete-time parameters
        if discrete != 1 -> plot the continuous time parameters
        '''

        if discrete != 1:
            print("True parameter values for the continuous time model")
            print("----------------------------------------------------------------------")
            print("phi   = [{: 1.3f}]    kappa = [{: 1.3f}, {: 1.3f}]    sigma = [{: 1.2f}, {: 1.2f}]".format(100*self.phi[0,0],self.kappa[0,0],self.kappa[0,1],100*self.sigma[0,0],100*self.sigma[0,1]))
            print("        [{: 1.3f}]            [{: 1.3f}, {: 1.3f}]            [{: 1.2f}, {: 1.2f}]".format(100*self.phi[1,0],self.kappa[1,0],self.kappa[1,1],100*self.sigma[1,0],100*self.sigma[1,1]))
            print("\n")
            print("beta0 = [{: 1.3f}]    beta1 = [{: 1.3f}, {: 1.3f}]    alpha = [{: 1.2f}, {: 1.2f}]".format(100*self.beta0[0,0],self.beta1[0,0],self.beta1[0,1],100*self.alpha[0,0],100*self.alpha[0,1]))
            print("        [{: 1.3f}]            [{: 1.3f}, {: 1.3f}]            [{: 1.2f}, {: 1.2f}]".format(100*self.beta0[1,0],self.beta1[1,0],self.beta1[1,1],100*self.alpha[1,0],100*self.alpha[1,1]))
            print("\n")
            print("Xi    = [{: 1.2f}, {: 1.2f}, {: 1.2f}]".format(self.Xi[0,0],self.Xi[0,1],self.Xi[0,2]))
            print("        [{: 1.2f}, {: 1.2f}, {: 1.2f}]".format(self.Xi[1,0],self.Xi[1,1],self.Xi[1,2]))
            print("        [{: 1.2f}, {: 1.2f}, {: 1.2f}]".format(self.Xi[2,0],self.Xi[2,1],self.Xi[2,2]))

            print("\n")
            print("Worst-case parameter values for the continuous time model")
            print("----------------------------------------------------------------------")
            print("phi   = [{: 1.3f}]    kappa = [{: 1.3f}, {: 1.3f}]".format(100*self.phi_tilde[0,0],self.kappa_tilde[0,0],self.kappa_tilde[0,1]))
            print("        [{: 1.3f}]            [{: 1.3f}, {: 1.3f}] ".format(100*self.phi_tilde[1,0],self.kappa_tilde[1,0],self.kappa_tilde[1,1]))
            print("\n")
            print("beta0 = [{: 1.3f}]    beta1 = [{: 1.3f}, {: 1.3f}]".format(100*self.beta0_tilde[0,0],self.beta1_tilde[0,0],self.beta1_tilde[0,1]))
            print("        [{: 1.3f}]            [{: 1.3f}, {: 1.3f}]".format(100*self.beta0_tilde[1,0],self.beta1_tilde[1,0],self.beta1_tilde[1,1]))
            print("\n")

            print("The corresponding half-life (for the detection error prob) is: {:1.2f} quarters\n".format(self.chernoff()[0]))

        else:
            phi_D, kappa_D, sigma_D, beta0_D, beta1_D, alpha_D, SigmaX_D_inf = self.from_cont_to_discrete()
            phi_D_tilde, kappa_D_tilde, sigma_D_tilde, beta0_D_tilde, beta1_D_tilde, alpha_D_tilde,\
                    SigmaX_D_tilde_inf = self.from_cont_to_discrete(worstcase = 1)

            print("Baseline parameter values arising from discrete sampling")
            print("----------------------------------------------------------------------\n")
            print("phi   = [{: 1.3f}]    kappa = [{: 1.3f}, {: 1.3f}]    sigma = [{: 1.3f}, {: 1.3f}]".format(100*phi_D[0,0],kappa_D[0,0],kappa_D[0,1],100*sigma_D[0,0],100*sigma_D[0,1]))
            print("        [{: 1.3f}]            [{: 1.3f}, {: 1.3f}]            [{: 1.3f}, {: 1.3f}]".format(100*phi_D[1,0],kappa_D[1,0],kappa_D[1,1],100*sigma_D[1,0],100*sigma_D[1,1]))
            print("\n")
            print("beta0 = [{: 1.3f}]    beta1 = [{: 1.3f}, {: 1.3f}]    alpha = [{: 1.3f}, {: 1.3f}]".format(100*beta0_D[0,0],beta1_D[0,0],beta1_D[0,1],100*alpha_D[0,0],100*alpha_D[0,1]))
            print("        [{: 1.3f}]            [{: 1.3f}, {: 1.3f}]            [{: 1.3f}, {: 1.3f}]".format(100*beta0_D[1,0],beta1_D[1,0],beta1_D[1,1],100*alpha_D[1,0],100*alpha_D[1,1]))

            print("\n")
            print("Worst-case parameters arising from discrete sampling")
            print("----------------------------------------------------------------------\n")
            print("phi   = [{: 1.3f}]    kappa = [{: 1.3f}, {: 1.3f}]    sigma = [{: 1.3f}, {: 1.3f}]".format(100*phi_D_tilde[0,0], kappa_D_tilde[0,0], kappa_D_tilde[0,1], 100*sigma_D_tilde[0,0], 100*sigma_D_tilde[0,1]))
            print("        [{: 1.3f}]            [{: 1.3f}, {: 1.3f}]            [{: 1.3f}, {: 1.3f}]".format(100*phi_D_tilde[1,0], kappa_D_tilde[1,0], kappa_D_tilde[1,1], 100*sigma_D_tilde[1,0], 100*sigma_D_tilde[1,1]))
            print("\n")
            print("beta0 = [{: 1.3f}]    beta1 = [{: 1.3f}, {: 1.3f}]    alpha = [{: 1.3f}, {: 1.3f}]".format(100*beta0_D_tilde[0,0], beta1_D_tilde[0,0], beta1_D_tilde[0,1], 100*alpha_D_tilde[0,0], 100*alpha_D_tilde[0,1]))
            print("        [{: 1.3f}]            [{: 1.3f}, {: 1.3f}]            [{: 1.3f}, {: 1.3f}]".format(100*beta0_D_tilde[1,0], beta1_D_tilde[1,0], beta1_D_tilde[1,1], 100*alpha_D_tilde[1,0], 100*alpha_D_tilde[1,1]))



    def generate_ab(self, worstcase = 0, T = 150, Ngrid = 151, real = 0):

        N, M, K = self.N, self.M, self.K
        alpha, sigma = self.alpha, self.sigma
        delta = self.delta

        if worstcase == 0:
            phi, kappa, beta0, beta1 = self.phi, self.kappa, self.beta0, self.beta1
        elif worstcase == 1:
            #phi, kappa, beta0, beta1 = self.phi_tilde, self.kappa_tilde, self.beta0_tilde, self.beta1_tilde

            # if using discrete time params
            phi, kappa, sigma, beta0, beta1, alpha, dont_need = self.from_cont_to_discrete(worstcase = 1)
        elif worstcase == 2:
            # this case is for the unrestriceted change of measure
            phi, kappa, beta0, beta1 = self.phi_bar, self.kappa_bar, self.beta0_bar, self.beta1_bar

        if real == 0:
            iota = np.ones((M, 1))
        elif real == 1:
            iota = np.zeros((M, 1))
            iota[0, 0] = 1

        if  worstcase == 2:
            rho0 = delta + iota.T @ self.beta0 - (.5) * iota.T @ alpha @ alpha.T @ iota
            rho1 = iota.T @ self.beta1
        else:
            rho0 = delta + iota.T @ beta0 - (.5)*iota.T @ alpha @ alpha.T @ iota
            rho1 = iota.T @ beta1

        t_grid = np.linspace(1e-15, T, Ngrid)
        dt = t_grid[1] - t_grid[0]
        a = np.zeros((Ngrid,))
        b = np.zeros((Ngrid, N))

        if worstcase == 2:
            for t in range(Ngrid - 1):
                b[t+1, :] = b[t, :] + dt*(b[t, :] @ (-kappa) - rho1)
                a[t+1] = a[t] + dt*(b[t, :] @ phi + (.5)*b[t, :] @ sigma @ sigma.T @ b[t, :].T - rho0)

        else:
            for t in range(Ngrid - 1):
                #b[t+1, :] = b[t, :] + dt*(b[t, :] @ (-kappa) - rho1)
                b[t+1, :] = dt*(b[t, :] @ kappa - rho1)      # if using discrete time params
                a[t+1] = a[t] + dt*(b[t, :] @ phi + (.5)*b[t, :] @ sigma @ sigma.T @ b[t, :].T - iota.T @ alpha @ sigma.T @ b[t, :].T - rho0)

        a = -a/t_grid
        a[0] = rho0
        t_grid.shape = (Ngrid, 1)
        b = -b/np.hstack((t_grid, t_grid))
        b[0, :] = rho1
        t_grid = t_grid.squeeze()

        return interp1d(t_grid, a), interp1d(t_grid, b[:, 0]), interp1d(t_grid, b[:, 1])


    def zero_coupon_yields(self, tau, worstcase = 0, real = 0):
        """
            tau : yield-to-maturity in quarters
            wc  : boolian, 0 refers to the benchmark model, 1 is for the worst-case
        """
        a, b1, b2 = self.generate_ab(worstcase = worstcase, real = real)

        return 400 * np.asarray([a(tau), b1(tau), b2(tau)])


    def yield_matrix(self, YTM, worstcase = 0):
        """
            Constructs the matrix containing the yields in the quantecon lss form
            ytm : nparray containing the yield-to-maturities for which we want to have yields
            wc  : boolian determining if yield params are under the worstcase model or not
        """
        M = self.M

        zc_yields = np.hstack([self.zero_coupon_yields(1, worstcase = worstcase, real = 0), np.zeros(M)])
        zc_yields_real = np.hstack([self.zero_coupon_yields(1, worstcase = worstcase, real = 1), np.zeros(M)])

        for ii in range(2, 121):
            zc_yields = np.vstack([zc_yields, np.hstack([self.zero_coupon_yields(ii, worstcase = worstcase, real = 0), np.zeros(M)])])
            zc_yields_real = np.vstack([zc_yields_real, np.hstack([self.zero_coupon_yields(ii, worstcase = worstcase, real = 1), np.zeros(M)])])

        yields_nom = zc_yields[np.asarray(YTM['nominal']) - 1, :]

        if YTM['real'] == []:
            yields = np.vstack([yields_nom])
        else:
            yields_real = zc_yields_real[np.asarray(YTM['real']) - 1, :]
            yields = np.vstack([yields_nom, yields_real])

        return yields


    def yield_distribution(self, T=1, wcParam=0, wcX=0, real = 0):

        abb = self.zero_coupon_yields(T, worstcase = wcParam, real = real)

        if wcX == 0:
            mean_yield = abb @ np.hstack([1, self.MuX_inf.squeeze()])
            var_yield = abb[1:] @ self.SigmaX_inf @ abb[1:]
        else:
            mean_yield = abb @ np.hstack([1, self.MuX_inf_tilde.squeeze()])
            var_yield = abb[1:] @ self.SigmaX_inf_tilde @ abb[1:]

        return sp.stats.norm(mean_yield, np.sqrt(var_yield))


    def X_CondDist(self, X, tau, worstcase=0):
        """
        X : current value of the state vector
        tau : horizon at which we want to forecast
        """
        if len(X.shape)==1:
            X = np.atleast_2d(X).T

        N, sigma = self.N, self.sigma

        if worstcase == 0:
            phi_D = self.from_cont_to_discrete(worstcase = 0)[0]
            kappa_D = self.from_cont_to_discrete(worstcase = 0)[1]
        elif worstcase == 1:
            phi_D = self.from_cont_to_discrete(worstcase = 1)[0]
            kappa_D = self.from_cont_to_discrete(worstcase = 1)[1]
        elif worstcase == 2:
            kappa, mu = self.kappa_bar, self.MuX_inf_bar

        constant = np.zeros((N, 1))
        for i in range(tau):
            constant += np.linalg.matrix_power(kappa_D, i) @ phi_D

        EX_tau = constant + np.linalg.matrix_power(kappa_D, tau) @ X

        VarX_tau = np.zeros((N, N))
        for i in range(1, tau + 1):
            VarX_tau += np.linalg.matrix_power(kappa_D, tau - i) @ sigma @ sigma.T @ np.linalg.matrix_power(kappa_D, tau - i).T

        return sp.stats.multivariate_normal(EX_tau.squeeze(), VarX_tau)

    def yield_forecast_ab(self, ytm, tau_end, tau_start=None, worstcase=1, worstcase_abb=1):
        """
        ytm       : yield-to-maturity at t + tau
        tau_end   : horizon at which we want to forecast
        tau_start : benchmark period from which we want to forecast
                    if None, standard forecast

        f(X_t) = E[y^{ytm}_{t+tau_end} - y^{ytm}_{t+tau_start} | X_t]
        the function returns E[f] and Var[f]

        """

        N, sigma = self.N, self.sigma
        m = len(ytm)

        abb = self.zero_coupon_yields(ytm, worstcase = worstcase_abb)

        if worstcase == 0:
            phi_D = self.from_cont_to_discrete(worstcase = 0)[0]
            kappa_D = self.from_cont_to_discrete(worstcase = 0)[1]
        elif worstcase == 1:
            phi_D = self.from_cont_to_discrete(worstcase = 1)[0]
            kappa_D = self.from_cont_to_discrete(worstcase = 1)[1]

        if tau_start is None:
            constant = np.zeros((N, 1))
            for i in range(tau_end):
                constant += np.linalg.matrix_power(kappa_D, i) @ phi_D
            autoreg_term = np.linalg.matrix_power(kappa_D, tau_end)

            Ey_a = abb[0, :].reshape(m, 1) + abb[1:, :].T @ constant
        else:
            constant = np.zeros((N, 1))
            for i in range(tau_start, tau_end):
                constant += np.linalg.matrix_power(kappa_D, i) @ phi_D
            autoreg_term = np.linalg.matrix_power(kappa_D, tau_end) - np.linalg.matrix_power(kappa_D, tau_start)

            Ey_a = abb[1:, :].T @ constant

        Ey_b = abb[1:, :].T @ autoreg_term

        return np.hstack([Ey_a, Ey_b])


    def Yield_forecast(self, X, tau, ytm, worstcase = 0, wc = 1):
        """
        X   : current value of the state vector
        tau : horizon at which we want to forecast
        ytm : yield-to-maturity at t+tau
        """
        if len(X.shape) == 1:
            X = np.atleast_2d(X).T

        N, sigma = self.N, self.sigma
        abb = self.zero_coupon_yields(ytm, worstcase = wc)

        X_dist = self.X_CondDist(X, tau, worstcase = worstcase)
        mean_yield = abb @ np.hstack([1, X_dist.mean])
        var_yield = abb[1:] @ X_dist.cov @ abb[1:]

        return sp.stats.norm( mean_yield, np.sqrt(var_yield) )


    def logY_forecast(self, X, tau, worstcase = 0):
        """
        X   : current value of the state vector
        tau : horizon at which we want to forecast
        """
        if len(X.shape) == 1:
            X = np.atleast_2d(X).T

        X_dist = self.X_CondDist(X, tau, worstcase = worstcase)

        phi_D, kappa_D, sigma_D, beta0_D, beta1_D, alpha_D, \
                    SigmaX_D_inf = self.from_cont_to_discrete(worstcase = worstcase)

        mean_logY = beta0_D + beta1_D @ X_dist.mean.reshape(self.M, 1)
        var_logY = beta1_D @ X_dist.cov @ beta1_D.T + alpha_D @ alpha_D.T

        return sp.stats.multivariate_normal(mean_logY.squeeze(), var_logY)


    def chernoff_objfunc(self, r):
        """
        For a given r and eta_0 and eta_1 this method calculates the psi (eignrvalue of the genreator)
        """

        # Pull out useful info
        N, M, K = self.N, self.M, self.K
        phi, kappa, sigma = self.phi, self.kappa, self.sigma
        eta_0, eta_1 = self.eta_0, self.eta_1

        kappa_tilde = kappa  - r*(sigma @ eta_1)
        phi_tilde = phi + r*(sigma @ eta_0)

        try:
        # (1) Calculate lambda2 (quadratic terms -- solving a CARE)
            A = -kappa_tilde
            B = np.sqrt(2)*sigma
            Q = (.5)*r*(r-1)*(eta_1.T @ eta_1 )
            R = - np.eye(K)
            lambda2, f, g = cont.care(A, B, Q, R = R, S = None, E = None)

        # (2) Calculate lambda1 (linear terms)
            denom = la.inv( kappa_tilde.T - (lambda2.T + lambda2) @ (sigma @ sigma.T) )
            numer = r*(r-1)*(eta_1.T @ eta_0) + (lambda2.T + lambda2) @ phi_tilde
            lambda1 = denom @ numer

        # (2) Calculate psi (constant terms)
            minus_psi = (.5)*r*(r-1)*(eta_0.T @ eta_0) + lambda1.T @ phi_tilde + (.5)*np.trace( (lambda2.T + lambda2 + lambda1 @ lambda1.T) @ (sigma @ sigma.T) )

            return np.asscalar(minus_psi)

        except ValueError:
            raise ValueError("Something wrong with the CARE")


    def chernoff(self):

        res = sp.optimize.minimize_scalar(self.chernoff_objfunc, bounds = (0, 1), method = 'bounded')
        if res.fun !=0:
            HL = np.log(2) / (-res.fun)
        else:
            HL = np.inf
        # half-life and Chernoff entropy
        return HL, -res.fun


    def autocov_model(self, nn, worstcase = 0):

        N, M, K = self.N, self.M, self.K

        phi, kappa, sigma, beta0, beta1, alpha, SigmaX_inf = self.from_cont_to_discrete(worstcase = worstcase)
        sigma, alpha = sigma*100, alpha*100

        cov_array = np.zeros((nn + 1, M, M))
        Exx, G, L = cont.dare(kappa.T, np.zeros((N, N)), sigma @ sigma.T, np.eye(N))
        cov_array[0, :, :] = beta1 @ Exx @ beta1.T  +  alpha @ alpha.T

        for j in range(1, nn + 1):
            cov_array[j, :, :] =  beta1 @ np.linalg.matrix_power(kappa, j-1) @ (kappa @ Exx @ beta1.T  + sigma @ alpha.T)

        return cov_array, Exx


    def autocov_plot(self, data, nn, NW_lags, models = [0]):
        '''
        models:
            0 : rational expectation beliefs
            1 : worst-case beliefs
            2 : unrestricted risk-neutral beliefs
        '''
        acov_model = self.autocov_model(nn, worstcase = 0)[0]

        if np.any(np.asarray(models)==1):
            acov_model_wc = self.autocov_model(nn, worstcase = 1)[0]

        if np.any(np.asarray(models)==2):
            acov_model_uc = self.autocov_model(nn, worstcase = 2)[0]

        # =====================================
        # Sample autocov function and GMM se
        # =====================================
        T, M = data.shape
        data = np.asarray(data - data.mean(0))

        store_acov = np.zeros((nn + 1, M, M))
        store_se = np.zeros((nn + 1, M, M))

        for m in range(M):
            for k in range(M):
                for j in range(nn + 1):
                    numb_obs = T-j
                    yt_ytj = data[j:, m]*data[:T-j, k]
                    acov_gmm = yt_ytj.mean()
                    u = np.vstack([np.zeros((j, 1)), (yt_ytj - acov_gmm).reshape(T-j, 1)])

                    store_acov[j, m, k] = acov_gmm
                    R = (u.T @ u)/numb_obs

                    for i in range(1, NW_lags):
                        R_temp = (u[i:, 0].T @ u[:-i, 0])/numb_obs
                        R += 2 * (1 - i/NW_lags) * R_temp

                    store_se[j, m, k] = np.sqrt(R/numb_obs)

        LB = store_acov - 2*store_se
        UB = store_acov + 2*store_se

        fig, ax = plt.subplots(2, 2, figsize = (14, 9))
        for i in range(4):
            rn = int(np.floor(i/2))
            cn = i % 2

            ax[rn, cn].plot(np.arange(nn + 1), store_acov[:, rn, cn], lw = 3, color = sb.xkcd_rgb["faded green"])
            ax[rn, cn].fill_between(np.arange(nn + 1), LB[:, rn, cn], UB[:, rn, cn], color = sb.xkcd_rgb["faded green"], alpha = .15)
            ax[rn, cn].plot(np.arange(nn + 1), acov_model[:, rn, cn], lw = 3, color = 'k')
            if np.any(np.asarray(models)==1):
                ax[rn, cn].plot(np.arange(nn + 1), acov_model_wc[:, rn, cn], lw = 3, color = 'r')
            if np.any(np.asarray(models)==2):
                ax[rn, cn].plot(np.arange(nn + 1), acov_model_uc[:, rn, cn], lw = 3, color = 'b')
            ax[rn, cn].axhline(color='k', linestyle = '--', lw =1)

        return fig, ax


    def autocorr_plot(self, data, nn, NW_lags, models = [0], figsize = (14, 8)):
        '''
        models:
            0 : rational expectation beliefs
            1 : worst-case beliefs
            2 : unrestricted risk-neutral beliefs
        '''
        acorr_model = np.zeros((nn + 1, 2, 2))

        acov_model = self.autocov_model(nn, worstcase = 0)[0]
        sig_x, sig_y = np.sqrt(acov_model[0, 0, 0]), np.sqrt(acov_model[0, 1, 1])
        sigmas = np.asarray([[sig_x*sig_x, sig_x*sig_y],
                             [sig_y*sig_x, sig_y*sig_y]])

        for i in range(nn + 1):
            acorr_model[i, :, :] = acov_model[i, :, :]/sigmas

        if np.any(np.asarray(models)==1):
            acov_model_wc = self.autocov_model(nn, worstcase = 1)[0]
            acorr_model_wc = np.zeros((nn + 1, 2, 2))
            sig_x, sig_y = np.sqrt(acov_model_wc[0, 0, 0]), np.sqrt(acov_model_wc[0, 1, 1])
            sigmas = np.asarray([[sig_x*sig_x, sig_x*sig_y],
                                [sig_y*sig_x, sig_y*sig_y]])
            for i in range(nn + 1):
                acorr_model_wc[i, :, :] = acov_model_wc[i, :, :]/sigmas

        if np.any(np.asarray(models)==2):
            acov_model_uc = self.autocov_model(nn, worstcase = 2)[0]
            acorr_model_uc = np.zeros((nn + 1, 2, 2))
            sig_x, sig_y = np.sqrt(acov_model_uc[0, 0, 0]), np.sqrt(acov_model_uc[0, 1, 1])
            sigmas = np.asarray([[sig_x*sig_x, sig_x*sig_y],
                                [sig_y*sig_x, sig_y*sig_y]])
            for i in range(nn + 1):
                acorr_model_uc[i, :, :] = acov_model_uc[i, :, :]/sigmas

        # =====================================
        # Sample autocov function and GMM se
        # =====================================
        T, M = data.shape
        data = np.asarray(data - data.mean(0))

        store_est = np.zeros((nn + 1, 3*M))
        store_acorr = np.zeros((nn + 1, M*M))
        store_se = np.zeros((nn + 1, M*M))


        for j in range(nn + 1):
            numb_obs = T - j
            store_uu = np.zeros((3*M, numb_obs))

            u_Ex2 = data.T[0, j:]**2
            u_Ey2 = data.T[1, j:]**2
            store_est[j, 0] = u_Ex2.mean()
            store_est[j, 1] = u_Ey2.mean()

            store_uu[0, :] = u_Ex2 - store_est[j, 0]
            store_uu[1, :] = u_Ey2 - store_est[j, 1]

            for m in range(M):
                for k in range(M):
                    xt_ytj = data[j:, m] * data[:T-j, k]
                    acov_gmm = xt_ytj.mean()
                    u = (xt_ytj - acov_gmm).reshape(T-j, 1)
                    store_uu[2 + 2*m + k, :] = u.T
                    store_est[j, 2 + 2*m + k] = acov_gmm

            R = (store_uu @ store_uu.T)/numb_obs

            for i in range(1, NW_lags):
                R_temp = (store_uu[:, i:] @ store_uu[:, :-i].T)/numb_obs
                R += 2 * (1 - i/NW_lags) * R_temp

            store_acorr[j, :], G = autocorr_gradient(store_est[j, :])
            store_se[j, :] = np.sqrt(abs(np.diag((G @ R @ G.T))/numb_obs))

        LB = store_acorr - 2*store_se
        UB = store_acorr + 2*store_se

        fig, ax = plt.subplots(2, 2, figsize = figsize)
        for i in range(4):
            rn = int(np.floor(i/2))
            cn = i % 2

            if np.any(np.asarray(models)==-1):
                ax[rn, cn].plot(np.arange(nn + 1), store_acorr[:, i], lw = 3, color = sb.xkcd_rgb["faded green"])
                ax[rn, cn].fill_between(np.arange(nn + 1), LB[:, i], UB[:, i], color = sb.xkcd_rgb["faded green"], alpha = .15)
            if np.any(np.asarray(models)==0):
                ax[rn, cn].plot(np.arange(nn + 1), acorr_model[:, rn, cn], lw = 3, color = 'k')
            if np.any(np.asarray(models)==1):
                ax[rn, cn].plot(np.arange(nn + 1), acorr_model_wc[:, rn, cn], lw = 3, color = sb.xkcd_rgb['pale red'])
            if np.any(np.asarray(models)==2):
                ax[rn, cn].plot(np.arange(nn + 1), acorr_model_uc[:, rn, cn], lw = 3, color = sb.xkcd_rgb['denim blue'], alpha = .6)
            ax[rn, cn].axhline(color='k', linestyle = '--', lw =1)

        return fig, ax



#=====================================================================================================
#=====================================================================================================
#=====================================================================================================



def first_step_estimation(data, initial, model, printout = True):

    if model == "restricted_2":
        numb_params = 13
    elif model == "restricted_3":
        numb_params = 11

    data = np.asarray(data)
    result_BFGS = sp.optimize.minimize(lambda param: -loglh(param, data, model = model),
                                       initial,
                                       method = 'BFGS',
                                       options = {'maxiter' : 1e5, 'disp': printout})
    if printout:
        print("\n")
        print('L_1 and L_inf norm of the gradient vector at MLE:')
        print("   ||x||_1 = {:1.5f}      ||x||_inf={:1.5f}".format(np.sum(abs(result_BFGS.jac)),
                                                               max(abs(result_BFGS.jac)) ))
        print('Hessian negative definite at MLE?:')
        print(all(np.linalg.eig(result_BFGS.hess_inv)[0] > 0))

    param_store = extract_params(result_BFGS.x[:numb_params], model = model)
    if model == "restricted_2":
        bfgs_est = build_params(param_store[0], param_store[1], param_store[2], param_store[3],
                                param_store[4], param_store[5])
    elif model == "restricted_3":
        bfgs_est = build_params(param_store[0], param_store[1], param_store[2], data.mean(0).reshape(2, 1),
                                param_store[4], param_store[5])

    param_se = np.sqrt(np.diag(result_BFGS.hess_inv[:numb_params, :numb_params])/data.shape[0])
    se_store = extract_params(param_se, model = model)
    bfgs_est_se = build_params(se_store[0], se_store[1], se_store[2], se_store[3], np.zeros((2,2)), se_store[5])

    if printout:
        print("\n")
        print("Parameter estimates:")
        print(np.around(bfgs_est, decimals = 3))
        print("Asymptotic standard errors:")
        print(np.around(bfgs_est_se, decimals = 3))

    return bfgs_est, bfgs_est_se, result_BFGS


def second_step_estimation(x, param_mle, X_hat, data_yields, ytm, case=1, delta=.001):
    """
    This function calculates the model implied yield curves for different Xi parameters (contained in x)
    and compare them with the data by calculating the sum of squared differences.
    """

    data_yields = np.atleast_2d(data_yields)
    T, Obs = data_yields.shape

    # Pull out the first stage MLE estimates -> will go directly into Robust
    phi, kappa, sigma, beta0, beta1, alpha = extract_params(param_mle)

    # Construct the Xi matrix and the xi coeffs going into the Robust class
    if case == 1:
        xi_0 = x[0]
        xi_1 = np.zeros((2, 1))
        xi_2 = np.asarray([[x[1], 0.0], [2*x[2], x[3]]])
    elif case == 2:
        xi_0 = x[0]
        xi_1, xi_2 = np.zeros((2, 1)), np.zeros((2, 2))
    elif case == 3:
        xi_0 = 0.0
        xi_1 = np.zeros((2, 1))
        xi_2 = np.asarray([[x[0], 0.0], [2*x[1], x[2]]])

    try:
        mm = Robust(phi, kappa, sigma, beta0, beta1, alpha, xi_0, xi_1, xi_2, delta)
    except ValueError:
        return 10**8, (10**8)*np.ones_like(data_yields)

    yield_coeffs = mm.zero_coupon_yields(ytm['nominal'], worstcase = 1).T   # Obs columns are for a, b
    Xhat = np.ones((3, T))
    Xhat[1:, :] = X_hat.T
    yield_hat_N = yield_coeffs @ Xhat                              # Obs x T matrix the model implied yields

    # Check if there are TIPS yields among ytm:
    if ytm['real'] == []:
        yield_hat = np.vstack([yield_hat_N])
    else:
        yield_coeffs_real = mm.zero_coupon_yields(ytm['real'], worstcase = 1, real = 1).T
        yield_hat_R = yield_coeffs_real @ Xhat
        yield_hat = np.vstack([yield_hat_N, yield_hat_R])

    # Sum of squared error objective:
    diff = (yield_hat.T - data_yields)*10      # multiplied by 10 to make the scale well-suited for the optimizer

#    return np.mean(np.nanmean(diff**2, 0)), (yield_hat.T - data_yields)**2
    return np.nanmean(diff**2), (yield_hat.T - data_yields)**2


def second_step_estimation_survey(x, param_mle, Xhat_survey, data_survey, ytm, case=1, level=0, delta=.0002):
    """
    This function calculates the model implied yield curves for different Xi parameters (contained in x)
    and compare them with the data by calculating the sum of squared differences.
    """

    data_yields = np.atleast_2d(data_survey)
    T, Obs = data_survey.shape

    # Pull out the first stage MLE estimates -> will go directly into Robust
    phi, kappa, sigma, beta0, beta1, alpha = extract_params(param_mle)

    # Construct the Xi matrix and the xi coeffs going into the Robust class
    if case == 1:
        xi_0 = x[0]
        xi_1 = np.zeros((2, 1))
        xi_2 = np.asarray([[x[1], 0.0], [2*x[2], x[3]]])
    elif case == 2:
        xi_0 = x[0]
        xi_1, xi_2 = np.zeros((2, 1)), np.zeros((2, 2))
    elif case == 3:
        xi_0 = 0.0
        xi_1 = np.zeros((2, 1))
        xi_2 = np.asarray([[x[0], 0.0], [2*x[1], x[2]]])

    try:
        mm = Robust(phi, kappa, sigma, beta0, beta1, alpha, xi_0, xi_1, xi_2, delta)
    except ValueError:
        return 10**8, (10**8)*np.ones_like(data_yields)

    if level == 0:
        exp_yield_coeffs = mm.yield_forecast_ab(ytm = ytm, tau_end = 5, tau_start = 1, worstcase = 1)
    elif level == 1:
        exp_yield_coeffs = mm.yield_forecast_ab(ytm = ytm, tau_end = 4, worstcase = 1)

    exp_yield_hat = exp_yield_coeffs @ Xhat_survey.T

    # Sum of squared error objective:
    diff = (exp_yield_hat.T - data_survey)*100

#    return np.mean(np.nanmean(diff**2, 0)), (yield_hat.T - data_yields)**2
    return np.nanmean(diff**2), exp_yield_hat.T


def ssq_error(x, param_mle, X_hat, data_yields, ytm, delta, case = 0):
    """
    This function calculates the model implied yield curves for different Xi parameters (contained in param_xi)
    and compare them with the data by calculating the sum of squared differences.

    Arguments:
    -------------
        - params       : parameters for the ltr S matrix, where Xi=SS'                      ...  (6,)
        - param_mle    : the ML parameter estimates for (kappa,sigma,beta1,alpha, x0)       ...  (13,)
        - X_hat        : filtered state X_hat                                               ...  (T+1, N)
        - data_yields  : observed zero-coupon yields in the columns                         ...  (T, Obs)
        - ytm          : yield-to-maturities IN QUARTERS for the nominal and possibly for
                         TIPS yields contained in data_yields. Dictionary with keys
                         'nominal' and 'real'. Must be consistent with data_yields          ...  Dict
        - delta        : discount rate                                                      ...  Float
        - case         : different kinds of restrictions on the Xi matrix, integer 0-5      ...  Int
    """

    data_yields = np.atleast_2d(data_yields)
    T, Obs = data_yields.shape

    # Pull out the first stage MLE estimates -> will go directly into Robust
    phi, kappa, sigma, beta0, beta1, alpha = extract_params(param_mle)

    # Construct the Xi matrix and the xi coeffs going into Robust...subject to the restrictions
    params = xi_restrictions(x, case = case)
    Xi = make_Xi(params)
    xi_0 = Xi[0, 0]
    xi_1 = np.asarray([[Xi[1, 0]], [Xi[2, 0]]])
    xi_2 = np.asarray([[Xi[1, 1], 0.0], [2*Xi[2, 1], Xi[2, 2]] ])


    try:
        mm = Robust(phi, kappa, sigma, beta0, beta1, alpha, xi_0, xi_1, xi_2, delta, X_hat[0, :])

        yield_coeffs = mm.zero_coupon_yields(ytm['nominal'], worstcase = 1).T   # Obs columns are for a, b
        Xhat = np.ones((3, T))
        Xhat[1:, :] = X_hat.T
        yield_hat_N = yield_coeffs @ Xhat                              # Obs x T matrix the model implied yields

        # Check if there are TIPS yields among ytm:
        if ytm['real'] == []:
            yield_hat = np.vstack([yield_hat_N])
        else:
            yield_coeffs_real = mm.zero_coupon_yields(ytm['real'], worstcase = 1, real = 1).T   # Obs cols are for a,b
            yield_hat_R = yield_coeffs_real @ Xhat                         # ObsxT matrix the model implied yields
            yield_hat = np.vstack([yield_hat_N, yield_hat_R])

    # Sum of squared error objective:
        #return np.mean(np.nanmean((yield_hat.T - data_yields)**2, 0)), (yield_hat.T - data_yields)**2
        return np.nanmean((yield_hat.T - data_yields)**2), (yield_hat.T - data_yields)**2

    except ValueError:
        return 10**6, (10**6) * np.ones_like(data_yields)


def ssq_error_unconstrained(x, param_mle, X_hat, data_yields, ytm, delta):
    """
    This function calculates the model implied yield curves for different Xi parameters (contained in param_xi)
    and compare them with the data by calculating the sum of squared differences.

    Arguments:
    -------------
        - params       : parameters for the unresticted zeta0 and zeta1                     ...  (6,)
        - param_mle    : the ML parameter estimates for (kappa,sigma,beta1,alpha, x0)       ...  (13,)
        - X_hat        : filtered state X_hat                                               ...  (T+1, N)
        - data_yields  : observed zero-coupon yields in the columns                         ...  (T, Obs)
        - ytm          : yield-to-maturities IN QUARTERS for the nominal and possibly for
                         TIPS yields contained in data_yields. Dictionary with keys
                         'nominal' and 'real'. Must be consistent with data_yields          ...  Dict
        - delta        : discount rate                                                      ...  Float
    """

    data_yields = np.atleast_2d(data_yields)
    T, Obs = data_yields.shape

    phi, kappa, sigma, beta0, beta1, alpha = extract_params(param_mle)
    xi_0 = 0
    xi_1 = np.asarray([[0.0], [0.0]])
    xi_2 = np.asarray([[0.0, 0.0], [0.0, 0.0]])


    try:
        mm = Robust(phi, kappa, sigma, beta0, beta1, alpha, xi_0, xi_1, xi_2, delta, X_hat[0, :], zeta = x)

        yield_coeffs = mm.zero_coupon_yields(ytm['nominal'], worstcase = 2).T   # Obs columns are for a, b
        Xhat = np.ones((3, T))
        Xhat[1:, :] = X_hat.T
        yield_hat_N = yield_coeffs @ Xhat                              # Obs x T matrix the model implied yields

        # Check if there are TIPS yields among ytm:
        if ytm['real'] == []:
            yield_hat = np.vstack([yield_hat_N])
        else:
            yield_coeffs_real = mm.zero_coupon_yields(ytm['real'], worstcase = 2, real = 1).T   # Obs cols are for a,b
            yield_hat_R = yield_coeffs_real @ Xhat                         # ObsxT matrix the model implied yields
            yield_hat = np.vstack([yield_hat_N, yield_hat_R])

    # Sum of squared error objective:
        return np.mean(np.nanmean((yield_hat.T - data_yields)**2, 0)), (yield_hat.T - data_yields)**2

    except ValueError:
        return 10**6, (10**6) * np.ones_like(data_yields)
