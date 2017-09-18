'''
Separate functions used by the Robust class but for some reason not included
'''

import numpy as np
import scipy as sp
import scipy.linalg as la
import quantecon as qe
import pandas as pd
import control as cont
from scipy.stats import lognorm
from scipy.interpolate import interp1d



def loglh(param, data, N=2, M=2, K=2, model = 'unrestricted'):
    return kalman_filter(param, data, N=N, M=M, K=K, model = model)[0]


def kalman_filter(param, data, N=2, M=2, K=2, model = 'unrestricted'):
    '''
    Objective function (for the NOT demeaned series),
        i.e. y_t = [log C_t, log P_t]^T  without demeaning them

    Model:
        dy_{t+1} = beta0_D + beta1_D x_{t} + alpha_D w_{t+1}
        x_{t+1} = phi_D + kappa_D x_{t} + sigma_D w_{t+1}

        alpha_D w_{t+1} is the Wold error/whitened error
        Omega = alpha_D (alpha_D)^T is MxM
    '''
    data = np.atleast_2d(data)

    if model == 'restricted_1':
        phi_D, kappa_D, sigma_D, beta0_D, beta1_D, alpha_D = extract_params(param[:14], N, M, K, model = model)
        if len(param) == 14:
            MuX_inf = (np.eye(N) - kappa_D) @ phi_D
            x = MuX_inf
        elif len(param) > 14:
            x = param[-2:].reshape(N, 1)

    elif model == 'restricted_2':
        phi_D, kappa_D, sigma_D, beta0_D, beta1_D, alpha_D = extract_params(param[:13], N, M, K, model = model)
#        beta0_D = np.mean(data, 0).reshape((M, 1))
        if len(param) == 13:
            MuX_inf = (np.eye(N) - kappa_D) @ phi_D
            x = MuX_inf
        elif len(param) > 13:
            x = param[-2:].reshape(N, 1)

    elif model == 'restricted_3':
        phi_D, kappa_D, sigma_D, beta0_D, beta1_D, alpha_D = extract_params(param[:11], N, M, K, model = model)
        beta0_D = np.mean(data, 0).reshape((M, 1))
        if len(param) == 11:
            MuX_inf = (np.eye(N) - kappa_D) @ phi_D
            x = MuX_inf
        elif len(param) > 11:
            x = param[-2:].reshape(N, 1)

    Omega = alpha_D @ alpha_D.T
    T, m = data.shape

    try:
        #---------------------------------------------------------------------
        # Rule out "bad scenarios":
        #   (1) the kappa_D matrix is unstable
        #   (2) "singular" covariance matrix for the forecast error
        #   (3) covariance matrices are not positive semidefinite
        cond1 = max(abs(la.eig(kappa_D)[0])) > 1
        cond2 = la.det(Omega) < 10e-50
        cond3 = min(la.eig(Omega)[0]) < 0 or min(la.eig(sigma_D @ sigma_D.T)[0]) < 0
        bad_scenario = any([cond1, cond2, cond3])

        if bad_scenario:
            return -10**6, np.nan*np.ones((N, T)), np.nan*np.ones((T, 1))
        else:
            llh = np.zeros((T, 1))
            XX = np.zeros((N, T + 1))

            XX[:, 0] = x.squeeze()
            invomega = la.inv(Omega)
            alphainv = la.inv(alpha_D)

            for tt in range(T):
                y  = np.asarray([data[tt, :]]).T
                xL = x
                u = y - beta0_D - beta1_D @ xL
                x =  phi_D + kappa_D @ xL + sigma_D @ (alphainv @ u)
                XX[:, tt+1] = x.squeeze()
                llh[tt] = -0.5*( m * np.log(2*np.pi) + np.log(la.det(Omega)) + u.T @ invomega @ u )

            return llh.sum()/T, XX[:, :-1], llh

    except ValueError:
        return -10**6, np.nan*np.ones((N, T)), np.nan*np.ones((T, 1))


def kalman_smoother(param, data, N=2, M=2, K=2, model = 'unrestricted'):
    '''
    Objective function (for the NOT demeaned series),
        i.e. y_t = [log C_t, log P_t]^T  without demeaning them

    Model:
        dy_{t+1} = beta0_D + beta1_D x_{t} + alpha_D w_{t+1}
        x_{t+1} = phi_D + kappa_D x_{t} + sigma_D w_{t+1}

        alpha_D w_{t+1} is the Wold error/whitened error
        Omega = alpha_D (alpha_D)^T is MxM
    '''
    data = np.atleast_2d(data)

    if model == 'restricted_2':
        phi_D, kappa_D, sigma_D, beta0_D, beta1_D, alpha_D = extract_params(param[:13], N, M, K, model = model)
        if len(param) == 13:
            MuX_inf = (np.eye(N) - kappa_D) @ phi_D
            x = MuX_inf
        elif len(param) > 13:
            x = param[-2:].reshape(N, 1)

    elif model == 'restricted_3':
        phi_D, kappa_D, sigma_D, beta0_D, beta1_D, alpha_D = extract_params(param[:11], N, M, K, model = model)
        beta0_D = np.mean(data, 0).reshape((M, 1))
        if len(param) == 11:
            MuX_inf = (np.eye(N) - kappa_D) @ phi_D
            x = MuX_inf
        elif len(param) > 11:
            x = param[-2:].reshape(N, 1)

    T, m = data.shape


    # Matrices to store values
    X_now = np.zeros((N, T))                       # E[x_{t} | F_t]
    X_next = np.zeros((N, T+1))                    # E[x_{t+1} | F_t]
    P_now = np.zeros((N, N, T))                    # P_{t|t}
    P_next = np.zeros((N, N, T+1))                 # P_{t+1|t}

    Omega = alpha_D @ alpha_D.T
    invomega = la.inv(Omega)
    Q = sigma_D @ sigma_D.T

    # Initialize the Kalman filter (X_next[:, 0] is zero)
    P_next[:, :, 0] = qe.solve_discrete_lyapunov(kappa_D, Q)


    #-----------------------------------------------------
    # Kalman filter: (data: t)
    #-----------------------------------------------------
    for t in range(T):

        omega = beta1_D @ P_next[:, :, t] @ beta1_D.T + Omega
        invomega = la.inv(omega)
        y  = np.asarray([data[t, :]]).T
        a = y - beta0_D - beta1_D @ X_next[:, t].reshape(N, 1)

        K_adj = (P_next[:, :, t] @ beta1_D.T) @ invomega
        X_now[:, t] = X_next[:, t] + (K_adj @ a).flatten()
        P_now[:, :, t] = P_next[:, :, t] - K_adj @ beta1_D @ P_next[:, :, t]

        X_next[:, t+1] = kappa_D @ X_now[:, t]
        P_next[:, :, t+1] = kappa_D @ P_now[:, :, t] @ kappa_D.T + Q


    #-------------------------------------------------
    # Backward step
    #-------------------------------------------------

    X_tT = np.zeros((N, T))
    P_tT = np.zeros((N, N, T))
    X_tT[:, -1] = X_now[:, -1]
    P_tT[:, :, -1] = P_now[:, :, -1]

    for t in range(T-2):
        J = P_now[:, :, -2-t] @ kappa_D.T @ P_next[:, :, -2-t]
        X_tT[:, -2-t] = X_now[:, -2-t] + J @ (X_now[:, -1-t] - X_next[:, -2-t])
        P_tT[:, :, -2-t] = P_now[:, :, -2-t] + J @ (P_now[:, :, -1-t] - P_next[:, :, -2-t]) @ J.T

    return X_tT, P_tT



def xi_restrictions(x_vec, case = 0):

    if case == 0:
        params = x_vec
    elif case == 1:
        params = np.asarray([x_vec[0], 0.0, 0.0, 0.0, 0.0, 0.0])
    elif case == 2:
        params = np.asarray([0.0, 0.0, 0.0, x_vec[0], 0.0, x_vec[1]])
    elif case == 3:
        params = np.asarray([x_vec[0], 0.0, 0.0, x_vec[1], 0.0, x_vec[2]])
    elif case == 4:
        params = np.asarray([0.0, 0.0, 0.0, x_vec[0], x_vec[1], x_vec[2]])
    elif case == 5:
        params = np.asarray([x_vec[0], 0.0, 0.0, x_vec[1], x_vec[2], x_vec[3]])
    elif case == 6:
        params = np.asarray([x_vec[0], x_vec[1], x_vec[2], x_vec[3], -x_vec[1]*x_vec[2]/x_vec[3], x_vec[4]])

    return params

def make_Xi(params):
    '''
    Parameterize the positive semidefinite symmetric matrix by its square root
    almost Cholesky...diagonal elements should be strictly positive
    '''
    xi_sq = np.asarray([[params[0], 0, 0],
                        [params[1], params[3], 0],
                        [params[2], params[4], params[5]]])
    return xi_sq @ xi_sq.T


def extract_xi(param_xi):
    Xi = np.asarray([[  param_xi[0],         0.0,         0.0],
                     [2*param_xi[1], param_xi[3],         0.0],
                     [2*param_xi[2], param_xi[4], param_xi[5]]])

    Xi_sym = (.5)*(Xi + Xi.T)
    xi_0 = param_xi[0]
    xi_1 = np.asarray([[param_xi[1]], [param_xi[2]]])
    xi_2 = np.asarray([ [param_xi[3], 0.0], [param_xi[4], param_xi[5]] ])

    return xi_0, xi_1, xi_2, Xi_sym

def extract_params(param, N=2, M=2, K=2, model = 'unrestricted'):
    """
    (1) Unrestricted model: (length of param = N+N*N+N*K+M*N+M+M*K = 20)
        param = phi, kappa, sigma, beta0, beta1, alpha
        The first M elements are for phi, the next NxN elements are for kappa, the next NxK elements are sigma
        The next M are for beta0, the next MxN are for beta1, the remaining MxK are for alpha

    (2) Restricted 1 model: (length of param = N*N + N*K + M + M*K = 14)
        param = kappa, sigma, beta0, lt(alpha)

    (3) Restricted 2 model: (length of param = N*N + N*K + M + (M*K-1) = 13)
        param = kappa, sigma, beta0, lt(alpha)

    (4) Restricted 3 model: (length of param =  N*N + (N*K-1) + (M*K-1) = 10)
        param = kappa, sigma, lt(alpha)
    """

    if model == 'unrestricted' and len(param) == N+N*N+N*K+M*N+M+M*K:
        phi = param[:N].reshape(N, 1)
        kappa = param[N:N+N*N].reshape(N, N)
        sigma = param[N+N*N:N+N*N+N*K].reshape(N, K)

        beta0 = param[N+N*N+N*K:N+N*N+N*K+M].reshape(M, 1)
        beta1 = param[N+N*N+N*K+M:N+N*N+N*K+M*N+M].reshape(M, N)
        alpha = param[N+N*N+N*K+M*N+M:N+N*N+N*K+M*N+M+M*K].reshape(M, K)

        return phi, kappa, sigma, beta0, beta1, alpha

    elif model == 'restricted_1' and len(param) == N*N+N*K+M+M*K:
        phi = np.zeros((N, 1))
        kappa = param[:N*N].reshape(N, N)
        sigma = param[N*N:N*N+N*K].reshape(N, K)

        beta0 = param[N*N+N*K:N*N+N*K+M].reshape(M, 1)
        beta1 = np.eye(M)
        alpha = param[N*N+N*K+M:N*N+N*K+M+M*K].reshape(M, K)

        return phi, kappa, sigma, beta0, beta1, alpha

    elif model == 'restricted_2' and len(param) == N*N+N*K+M+M*K-1:
        phi = np.zeros((N, 1))
        kappa = param[:N*N].reshape(N, N)
        sigma = param[N*N:N*N+N*K].reshape(N, K)

#        resolv = np.eye(N) - kappa
#        kappa_C = -la.logm(kappa)
#        kappa_inv = la.inv(kappa_C)
#        beta1 = kappa_inv @ resolv

        beta0 = param[N*N+N*K:N*N+N*K+M].reshape(M, 1)
        beta1 = np.eye(M)
        aa = param[N*N+N*K+M:N*N+N*K+M+M*K-1]
        alpha = np.asarray([[aa[0], 0.0],[aa[1], aa[2]]])

        return phi, kappa, sigma, beta0, beta1, alpha

    elif model =='restricted_3' and len(param) == N*N+N*K+M*K-1:
        phi = np.zeros((N, 1))
        kappa = param[:N*N].reshape(N, N)
        sigma = param[N*N:N*N+N*K].reshape(N, K)

        beta0 = np.zeros((M, 1))
        beta1 = np.eye(M)
        aa = param[N*N+N*K:N*N+N*K+M*K-1]
        alpha = np.asarray([[aa[0], 0.0],[aa[1], aa[2]]])

        return phi, kappa, sigma, beta0, beta1, alpha

    else:
        raise ValueError("The specification is not consistent with the length of param!")


def build_params(phi, kappa, sigma, beta0, beta1, alpha):
    """
    The inverse of extract_params
    """
    N, K = sigma.shape
    M = phi.shape[0]

    params = np.hstack([np.reshape(phi, (N,) ),
                        np.reshape(kappa, (N*N,) ),
                        np.reshape(sigma, (N*K,) ),
                        np.reshape(beta0, (M,) ) ,
                        np.reshape(beta1, (M*N,) ),
                        np.reshape(alpha, (M*K,) )])

    return params


def from_discrete_to_cont(phi_D, kappa_D, sigma_D, beta0_D, beta1_D, alpha_D):

    tau = 1
    N, K = sigma_D.shape
    M = beta1_D.shape[0]

    kappa = -la.logm(kappa_D) / tau
    if type(kappa[0,0])== np.complex128:
        raise ValueError("kappa is complex")

    kappa_inv = la.inv(kappa)

    vecSigma = (sigma_D @ sigma_D.T).flatten('F')
    KK = np.kron(kappa, np.eye(N)) + np.kron(np.eye(N), kappa)
    sigma2 = np.reshape( la.inv( np.eye(N * N) - la.expm(-KK * tau) ) @ KK  @ vecSigma, (N, N))

    try:
        sigma = np.linalg.cholesky(sigma2)
    except np.linalg.LinAlgError:
        raise ValueError("Sigma2 is not positive definite")

    resolv = np.eye(N) - kappa_D
    mu = resolv @ phi_D
    phi = kappa @ mu

    beta1 = beta1_D @ la.inv(resolv) @ kappa
    beta0 = beta0_D/tau - (beta1 @ (np.eye(N) - kappa_inv @ resolv ) @ mu)

    #def finding_alpha(param):
    #    alpha = np.asarray([[param[0], 0], [param[1], param[2]]])
    #    vecSigma = (sigma @ sigma.T).flatten('F')

    #    const = beta1 @ kappa_inv @ sigma + alpha
    #    integ = kappa_inv @ resolv
    #    integ_transpose = (np.eye(N) - la.expm(-kappa.T*tau)) @ kappa_inv.T
    #    Q_c = kappa_inv @ sigma @ sigma.T @ kappa_inv.T
    #    vecQ = Q_c.flatten('F')

    #    first_term = (const @ const.T)*tau
    #    second_term = - const @ ( sigma.T @ kappa_inv.T @ integ_transpose @ beta1.T   )
    #    third_term =  - ( beta1 @ integ @ kappa_inv @ sigma ) @ const.T
    #    fourth_term = beta1 @ np.reshape( la.inv( KK ) @ (np.eye(N*N) - la.expm(-KK*tau))  @ vecQ ,(N,N)) @ beta1.T

    #    alpha2_alter = first_term + second_term + third_term + fourth_term

    #    return (alpha_D - np.linalg.cholesky(alpha2_alter)).flatten()[[0,2,3]]

    #try:
    #    sol = sp.optimize.root(finding_alpha, (alpha_D - np.eye(N)/1000).flatten()[[0, 2, 3]] )
    #    alpha_p = sol.x
    #    alpha = np.asarray([[alpha_p[0], 0], [alpha_p[1], alpha_p[2]]])
    #except TypeError:
    #    raise ValueError("Problem with the solver")

#    return phi, kappa, sigma_D, beta0, beta1, alpha_D
    return phi, kappa, sigma, beta0, beta1, alpha_D


def NBER_Shade(ax, start_date, date_file):
    """
    This function adds NBER recession bands to a Matplotlib Figure object.
    ax         : axis
    start_date : start date for the sample, form: yyyy-mm-dd
    """

    # load the NBER recession dates
    NBER_Dates = pd.read_csv(date_file)
    sample_1 = pd.Timestamp(start_date) <= pd.DatetimeIndex(NBER_Dates['Peak'])
    sample_2 = pd.Timestamp(start_date) <= pd.DatetimeIndex(NBER_Dates['Trough'])
    NBER_Dates = NBER_Dates[sample_1 + sample_2]

    # for loop generates recession bands!
    for i in NBER_Dates.index:
        ax.axvspan(NBER_Dates['Peak'][i], NBER_Dates['Trough'][i],
                    facecolor='grey', alpha=0.15)


def quadratic(x1, x2, xi0, xi1, xi2):
    val = xi0 + 2*(xi1[0]*x1 + xi1[1]*x2) + (x1**2)*xi2[0] + (x2**2)*xi2[2] + (x1*x2)*xi2[1]
    val[val < 0] = np.nan

    return val


def autocorr_gradient(estimates):
    Ex2, Ey2, Exxj, Exyj, Eyxj, Eyyj = estimates

    sig_x = np.sqrt(Ex2)
    sig_y = np.sqrt(Ey2)
    der_x = -(.5)/(sig_x**3 * sig_y)
    der_y = -(.5)/(sig_x * sig_y**3)

    ACF = np.asarray([Exxj/Ex2,
                      Exyj/(sig_x * sig_y),
                      Eyxj/(sig_x * sig_y),
                      Eyyj/Ey2])

    G = np.asarray([[-Exxj/(Ex2**2),              0, 1/Ex2,               0,               0,     0],
                    [  Exyj * der_x,   Exyj * der_y,     0, 1/(sig_x*sig_y),               0,     0],
                    [  Eyxj * der_x,   Eyxj * der_y,     0,               0, 1/(sig_x*sig_y),     0],
                    [             0, -Eyyj/(Ey2**2),     0,               0,               0, 1/Ey2]])
    return ACF, G



def NW_se(data, k):
    """
    This function computes GMM standard errors for the mean and standard deviation estimators
    """

    data = np.atleast_2d(data)
    data = data[np.isfinite(data)]
    T = max(data.shape)
    data = data.reshape(T, 1)

    gmm_mean = np.nanmean(data)
    gmm_stdev = np.nanstd(data)

    d_hat = np.asarray([[-1, 0], [0, -2*gmm_stdev]])

    f1 = data - np.ones((T, 1))*gmm_mean
    f2 = (data - np.ones((T, 1))*gmm_mean)**2 - np.ones((T, 1))*gmm_mean**2
    f = np.hstack([f1, f2])

    R = (f.T @ f)/T

    for i in range(1, k):
        R_temp = (f[i:, :].T @ f[:-i, :])/T
        R += 2 * (1 - i/k) * R_temp

    S = R

    V = np.linalg.inv(d_hat.T @ np.linalg.inv(S) @ d_hat)

    return np.asarray([gmm_mean, gmm_stdev]), np.sqrt(np.diag(V)/T)

def NW_corr(data, k):
    """
    This function computes GMM standard errors for the mean and standard deviation estimators
    """

    data = np.atleast_2d(data)
    T = max(data.shape)
    data = data.reshape(T, 2)
    data = data - data.mean(0)

    Ex2 = np.mean(data[:, 0]**2)
    Ey2 = np.mean(data[:, 1]**2)
    Exy = np.mean(data[:, 0] * data[:, 1])

    f1 = (data[:, 0]**2 - Ex2).reshape(T, 1)
    f2 = (data[:, 1]**2 - Ey2).reshape(T, 1)
    f3 = (data[:, 0] * data[:, 1] - Exy).reshape(T, 1)

    f = np.hstack([f1, f2, f3])
    R = (f.T @ f)/T

    for i in range(1, k):
        R_temp = (f[i:, :].T @ f[:-i, :])/T
        R += 2 * (1 - i/k) * R_temp

    V = R

    d_hat = np.asarray([[-0.5*Exy/Ex2**(3/2)/np.sqrt(Ey2), -0.5*Exy/np.sqrt(Ex2)/Ey2**(3/2), 1/np.sqrt(Ex2)/np.sqrt(Ey2)]])


    return Exy/np.sqrt(Ex2)/np.sqrt(Ey2), np.sqrt(d_hat @ V @ d_hat.T/T)[0]


def autocorrelation(data, nn, NW_lags):
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

    return store_acorr, LB, UB

def autocorr_model(kappa, sigma, model2, isnn):

    N, M, K = 2, 2, 2

    b = model2.zero_coupon_yields(tau=4, worstcase=1)[1:].reshape(2, 1)
    db = model2.zero_coupon_yields(tau=20, worstcase=1)[1:].reshape(2, 1) - model2.zero_coupon_yields(tau=4, worstcase=1)[1:].reshape(2, 1)

    beta1 = np.vstack([b.T, db.T]) @ kappa
    sigma = sigma*100
    alpha = np.vstack([b.T, db.T]) @ sigma

    cov_array = np.zeros((nn + 1, M, M))
    Exx, G, L = cont.dare(kappa.T, np.zeros((N, N)), sigma @ sigma.T, np.eye(N))
    cov_array[0, :, :] = beta1 @ Exx @ beta1.T  +  alpha @ alpha.T

    for j in range(1, nn + 1):
        cov_array[j, :, :] =  beta1 @ np.linalg.matrix_power(kappa, j-1) @ (kappa @ Exx @ beta1.T  + sigma @ alpha.T)

    acorr_model = np.zeros((nn + 1, 2, 2))

    acov_model = cov_array
    sig_x, sig_y = np.sqrt(acov_model[0, 0, 0]), np.sqrt(acov_model[0, 1, 1])
    sigmas = np.asarray([[sig_x*sig_x, sig_x*sig_y],
                         [sig_y*sig_x, sig_y*sig_y]])

    for i in range(nn + 1):
        acorr_model[i, :, :] = acov_model[i, :, :]/sigmas

    return acorr_model


def FB_bootstrap(yy, xx):
    """
     This function implements the so called "bootstrapping" method to turn par yields
     into zero coupon yields.

     Arguments:
         yy    :    list of par yields with different maturities (in quarters)
                    ascending order for a given date t
         xx    :    list of corresponding maturities (in quarters!)

     NOTE: We need this bacause the survey questions refer to
     Treasury par (CMT) yields. But the affine term structure model is affine only in
     zero-coupon yields (not par yields). They are actually quite close, so hopefully
     the approximation error is small.
    """
    yy = np.asarray(yy)

    #---------------------------------
    # Take care of the NaNs
    #---------------------------------
    if np.isnan(yy)[-1]==True:
        yy[-1] = yy[-2] #+ (yy[-2] - yy[-3])
    nan = np.isnan(yy)
    nonan = np.array([not e for e in nan])

    #---------------------------------
    # "Bootstrapping" Fama and Bliss
    #---------------------------------

    par = interp1d(xx[nonan], yy[nonan])
    z_rate = 1 + yy/200
    zeroc = np.asarray(par(2))

    for ytm in np.arange(2, 61)*2:
        ind = int(ytm/2)
        cc = (par(ytm)/2) * np.ones(ind)
        cc[-1] += 100
        new_zero = ((cc[-1]/(100 - sum((1 + zeroc/200)**(-np.arange(1, ind)) * cc[:-1])))**(1/ind) - 1)*200
        zeroc = np.hstack([zeroc, new_zero])

    return zeroc


def nan_interpolator(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """
    y = np.asarray(y)

    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    return y
