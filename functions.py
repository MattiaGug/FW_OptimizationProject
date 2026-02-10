import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.linalg import norm
from scipy.sparse.linalg import svds 


def function_loss(R: np.ndarray, P: np.ndarray) -> float:
    known_indices = R.nonzero()
    R_known = R[known_indices]
    P_predicted = P[known_indices]
    
    N = len(R_known)
    if N == 0:
        return 0.0
    
    mse = np.sum((R_known - P_predicted) ** 2) / N
    return mse


def gradient(R: np.ndarray, P: np.ndarray) -> np.ndarray:
    '''
    Computes gradient of MSE loss w.r.t. P (only on observed entries)
    '''
    known_indices = R.nonzero()
    N = len(known_indices[0])
    if N == 0:
        return np.zeros_like(P)
    
    grad = np.zeros_like(P, dtype=float)
    grad[known_indices] = 2.0 * (P[known_indices] - R[known_indices]) / N   
    return grad


def LMO(grad: np.ndarray, delta: float) -> np.ndarray:
    ''' 
    LMO - Linear Minimization Oracle
    Finds the S matrix that minimizes <gradient, S> under the 
    constraint ||S||_* <= delta.

    Parameters: 
    - grad: gradient matrix
    - delta: radius of the nuclear norm ball
    '''

    # Computing only the largest singular value (k=1)
    u, s, vt = svds(grad, k=1, which='LM')
    
    # Reshaping for proper matrix multiplication
    u = u.reshape(-1, 1)
    vt = vt.reshape(1, -1)
    
    S = -delta * np.dot(u, vt)
    
    return S


def FW_standard(R: np.ndarray, delta: float, max_iter: int = 200,
                tol: float = 1e-6, init_type: str = 'zeros',
                init_with_lmo: bool = False, verbose: bool = False) -> tuple:
    '''
    Standard Frank-Wolfe algorithm with classic step size gamma = 2/(k+1).
    '''
    m, n = R.shape

    if init_with_lmo:
        initial_grad = gradient(R, np.zeros((m, n)))
        P = LMO(initial_grad, delta)
    else:
        P = np.zeros((m, n), dtype=float)

    history = [function_loss(R, P)]
    gap_history = []

    for k in range(max_iter):
        grad = gradient(R, P)
        S = LMO(grad, delta)
        D = S - P

        gap = np.sum(grad * (P - S))
        gap_history.append(gap)

        if gap < tol:
            if verbose:
                print(f"Converged at iteration {k} (gap={gap:.2e})")
            break

        gamma = 2.0 / (k + 1)

        P = P + gamma * D

        loss = function_loss(R, P)
        history.append(loss)

        if verbose and k % 20 == 0:
            print(f"Iter {k}: loss={loss:.4f}, gap={gap:.2e}, gamma={gamma:.4f}")

    return P, history, gap_history


def FW_line_search(R: np.ndarray, delta: float, max_iter: int = 200,
                   tol: float = 1e-6, verbose: bool = False,
                   init_with_lmo: bool = False) -> tuple:
    m, n = R.shape

    if init_with_lmo:
        initial_grad = gradient(R, np.zeros((m, n)))
        P = LMO(initial_grad, delta)
    else:
        P = np.zeros((m, n), dtype=float)

    history = [function_loss(R, P)]
    gap_history = []

    known_indices = R.nonzero()
    R_obs = R[known_indices]

    for k in range(max_iter):
        grad = gradient(R, P)
        S = LMO(grad, delta)
        D = S - P

        gap = np.sum(grad * (P - S))
        gap_history.append(gap)

        if gap < tol:
            break

        # line search on observed entries
        P_obs = P[known_indices]
        D_obs = D[known_indices]

        numerator = np.sum((R_obs - P_obs) * D_obs)
        denominator = np.sum(D_obs ** 2)

        if denominator > 1e-12:
            gamma = np.clip(numerator / denominator, 0.0, 1.0)
        else:
            gamma = 2.0 / (k + 2)

        P = P + gamma * D

        loss = function_loss(R, P)
        history.append(loss)

    return P, history, gap_history



def FW_pairwise(R: np.ndarray, delta: float, max_iter: int = 500,
                tol: float = 1e-6, verbose: bool = False, init_with_lmo: bool = True) -> tuple:
    '''
    Pairwise Frank-Wolfe per Matrix Completion.
    Mantiene una decomposizione esplicita P = sum(alpha_i * S_i).
    
    Versione corretta:
    - Segno giusto nella line-search
    - No normalizzazione lambda → vincolo rispettato
    - Check similarità per evitare duplicati atomi
    - Stampa ||P||_* per monitorare vincolo
    - Fallback gamma minimo se denominator piccolo
    '''
    m, n = R.shape

    # Pre-calcolo indici osservati per efficienza
    rows_obs, cols_obs = R.nonzero()
    n_obs = len(rows_obs)
    if n_obs == 0:
        return np.zeros((m, n)), [0.0], [0.0]

    # Helper per prodotto scalare solo su observed
    def dot_obs(A, B):
        return np.dot(A[rows_obs, cols_obs], B[rows_obs, cols_obs])

    # Inizializzazione
    grad0 = gradient(R, np.zeros((m, n)))
    S0 = LMO(grad0, delta)

    if init_with_lmo:
        P = S0.copy()
        atoms = [S0]
        alphas = [1.0]
    else:
        # P parte da zero
        P = np.zeros((m, n), dtype=float)
        atoms = [S0]
        alphas = [0.0]

    history = [function_loss(R, P)]
    gap_history = []

    for k in range(max_iter):
        grad = gradient(R, P)

        # Forward atom
        S_fw = LMO(grad, delta)

        #Away atom
        idx_away = -1
        max_val = -np.inf

        for i, atom in enumerate(atoms):
            val = dot_obs(grad, atom)
            if val > max_val:
                max_val = val
                idx_away = i

        S_away = atoms[idx_away]
        alpha_away = alphas[idx_away]

        # Direzione pairwise
        D = S_fw - S_away

        # Dual Gap
        val_fw = dot_obs(grad, S_fw)
        val_P = dot_obs(grad, P)
        gap = val_P - val_fw
        gap_history.append(gap)

        if gap < tol:
            if verbose:
                print(f"Converged at iteration {k} (gap={gap:.2e})")
            break

        # Line Search 
        P_minus_R_obs = P[rows_obs, cols_obs] - R[rows_obs, cols_obs]
        D_obs = D[rows_obs, cols_obs]

        numerator = np.dot(P_minus_R_obs, D_obs) 
        denominator = np.dot(D_obs, D_obs)

        if denominator < 1e-14:
            gamma = 0.0
        else:
            gamma = -numerator / denominator

        gamma_max = alpha_away
        gamma = np.clip(gamma, 0.0, gamma_max)

        # Fallback minimo se gamma troppo piccolo
        if gamma < 1e-10:
            gamma = 1e-6

        # Aggiornamento P
        if gamma > 1e-14:
            P = P + gamma * D

            # Aggiornamento pesi
            alphas[idx_away] -= gamma

            # Aggiungo S_fw 
            found_fw = False
            norm_fw = np.linalg.norm(S_fw, 'fro')
            for i, atom in enumerate(atoms):
                norm_atom = np.linalg.norm(atom, 'fro')
                if norm_fw > 1e-10 and norm_atom > 1e-10:
                    corr = np.abs(np.dot(S_fw.ravel(), atom.ravel())) / (norm_fw * norm_atom)
                    if corr > 0.999:
                        alphas[i] += gamma
                        found_fw = True
                        break

            if not found_fw:
                atoms.append(S_fw)
                alphas.append(gamma)


            if alphas[idx_away] <= 1e-10 and len(alphas) > 1:
                atoms.pop(idx_away)
                alphas.pop(idx_away)

        loss = function_loss(R, P)
        history.append(loss)

        if verbose and k % 20 == 0:
            nuc_norm = np.linalg.norm(P, 'nuc')
            print(f"Iter {k}: loss={loss:.4f}, gap={gap:.2e}, atoms={len(atoms)}, "
                  f"gamma={gamma:.4f}, ||P||_* = {nuc_norm:.2f}/{delta}")

    return P, history, gap_history
