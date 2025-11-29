import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.linalg import norm
from scipy.sparse.linalg import svds

'''
def function_loss(R: np.ndarray, P: np.ndarray) -> float:
    
    Computes the Mean Squared Error (MSE) between the non-zero elements
    of R and the corresponding elements in P.

    Parameters:
    - R: The actual matrix.
    - P: The predicted matrix.

    Returns:
    - The MSE (actual matrix - predicted one)^2.
    
    # Find the indices of known ratings (non-zero in R)
    known_indices = R.nonzero()
    
    # Extract only the known ratings and corresponding predictions
    R_known = R[known_indices]
    P_predicted = P[known_indices]

    # Calculating the Squared Error
    squared_error = np.square(R_known - P_predicted)
    
    # MSE
    N = len(R_known)
    if N == 0:
        return 0.0  # Avoiding division by zero if R is all zeros
    
    mse = np.sum(squared_error) / N
    
    # Returning Mean Squared Error
    return mse 

'''

def function_loss(R: np.ndarray, P: np.ndarray) -> float:
    known_indices = R.nonzero()
    R_known = R[known_indices]
    P_predicted = P[known_indices]
    
    N = len(R_known)
    if N == 0:
        return 0.0
    
    mse = np.sum((R_known - P_predicted) ** 2) / N  # <-- divisione per N!
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
    # Compute only the largest singular value (k=1)
    u, s, vt = svds(grad, k=1, which='LM')
    
    # Reshape for proper matrix multiplication
    u = u.reshape(-1, 1)
    vt = vt.reshape(1, -1)
    
    # Solution is -delta * u * v.T
    S = -delta * np.dot(u, vt)
    
    return S


def FW_standard(R: np.ndarray, delta: float, max_iter: int = 200, 
                tol: float = 1e-6, init_type: str = 'zeros', 
                init_with_lmo = False, verbose: bool = False) -> tuple:
    '''
    Standard Frank-Wolfe algorithm with classic step size gamma = 2/(k+2).
    
    Parameters:
        R           : observed rating matrix (zeros = missing)
        delta       : nuclear norm budget ||P||_* <= delta
        max_iter    : maximum number of iterations
        tol         : stop when duality gap < tol
        init_type   : 'zeros' or 'random' (projected random init)
        verbose     : print progress
        
    Returns:
        P_final, loss_history, gap_history
    '''
    
    m, n = R.shape
    
    if init_with_lmo:
        # Stessa inizializzazione del Pairwise
        initial_grad = gradient(R, np.zeros((m, n)))
        P = LMO(initial_grad, delta)
    else:
        P = np.zeros((m, n), dtype=float)

    history = []
    gap_history = []

    for k in range(max_iter):
        # Gradient
        grad = gradient(R, P)
        
        # LMO → extreme point S
        S = LMO(grad, delta)
        
        # Direction
        D = S - P
        
        # Duality gap
        gap = np.sum(grad * (P - S))
        gap_history.append(gap)
        
        # Stop if converged
        if gap < tol:
            if verbose:
                print(f"Converged at iteration {k} (gap={gap:.2e})")
            break
        
        # Standard FW step size: gamma = 2/(k+2)
        gamma = 2.0 / (k + 2)
        
        # Update
        P = P + gamma * D
        
        loss = function_loss(R, P)
        history.append(loss)
        
        if verbose and k % 20 == 0:
            print(f"Iter {k}: loss={loss:.4f}, gap={gap:.2e}, gamma={gamma:.4f}")
    
    return P, history, gap_history


def FW_pairwise(R: np.ndarray, delta: float, max_iter: int = 500, 
                tol: float = 1e-6) -> tuple:
    '''
    TRUE Pairwise Frank-Wolfe with explicit active set tracking.
    Maintains X = sum(lambda_i * S_i) where S_i are atoms and lambda_i >= 0.
    
    Parameters:
        R           : observed rating matrix
        delta       : nuclear norm budget
        max_iter    : maximum iterations
        tol         : convergence tolerance
        
    Returns:
        P_final, loss_history, gap_history
    '''
    m, n = R.shape
    
    # FIX 1: Initialize with a proper atom from LMO (not zeros!)
    initial_grad = gradient(R, np.zeros((m, n)))
    S_init = LMO(initial_grad, delta)
    
    P = S_init.copy()
    history = []
    gaps = []
    
    # Active set: list of atoms and their coefficients
    S_atoms = [S_init.copy()]
    lambdas = [1.0]

    for k in range(1, max_iter + 1):
        G = gradient(R, P)
        
        # 1. Forward step: find best vertex via LMO
        S_fw = LMO(G, delta)
        gap_fw = np.sum(G * (P - S_fw))
        
        # 2. Away step: find worst atom in active set
        dot_products = [np.sum(G * atom) for atom in S_atoms]
        id_away = np.argmax(dot_products)
        S_away = S_atoms[id_away]
        lambda_away = lambdas[id_away]
        
        # 3. Pairwise direction
        d_pairwise = S_fw - S_away
        
        # 4. Duality gap (use FW gap for stopping)
        gap = gap_fw
        gaps.append(gap)
        
        if gap < tol:
            break
        
        # 5. Maximum step size (can't exceed lambda_away)
        gamma_max = lambda_away
        
        # 6. FIX 2: Correct line search for MSE on observed entries
        known_indices = R.nonzero()
        R_obs = R[known_indices]
        P_obs = P[known_indices]
        D_obs = d_pairwise[known_indices]
        
        numerator = np.sum((R_obs - P_obs) * D_obs)
        denominator = np.sum(D_obs ** 2)
        
        if denominator > 1e-12:
            gamma = np.clip(numerator / denominator, 0.0, gamma_max)
        else:
            gamma = min(2.0 / (k + 2), gamma_max)
        
        # 7. Update active set
        # FIX 3: Better atom comparison using correlation
        s_fw_index = -1
        norm_fw = np.linalg.norm(S_fw, 'fro')
        for i, atom in enumerate(S_atoms):
            norm_atom = np.linalg.norm(atom, 'fro')
            if norm_fw > 1e-10 and norm_atom > 1e-10:
                correlation = np.abs(np.sum(S_fw * atom)) / (norm_fw * norm_atom)
                if correlation > 0.999:
                    s_fw_index = i
                    break
        
        if s_fw_index != -1:
            lambdas[s_fw_index] += gamma
        else:
            S_atoms.append(S_fw.copy())
            lambdas.append(gamma)
        
        # Decrease coefficient of away atom
        lambdas[id_away] -= gamma
        
        # Remove atoms with zero coefficient
        if lambdas[id_away] <= 1e-10:
            del S_atoms[id_away]
            del lambdas[id_away]
        
        # FIX 4: Normalize coefficients
        total = sum(lambdas)
        if total > 0:
            lambdas[:] = [l / total for l in lambdas]
        
        # 8. Update P
        P = P + gamma * d_pairwise
        
        loss = function_loss(R, P)
        history.append(loss)

    return P, history, gaps


def FW_line_search(R: np.ndarray, delta: float, max_iter: int = 200,
                   tol: float = 1e-6, verbose: bool = False) -> tuple:
    m, n = R.shape
    P = np.zeros((m, n), dtype=float)

    history = []
    gap_history = []

    for k in range(max_iter):
        grad = gradient(R, P)
        S = LMO(grad, delta)
        D = S - P
        
        gap = np.sum(grad * (P - S))
        gap_history.append(gap)
        
        if gap < tol:
            break
        
        # QUESTA È LA PARTE CRITICA - line search corretto
        known_indices = R.nonzero()
        R_obs = R[known_indices]
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