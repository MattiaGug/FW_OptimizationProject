import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.linalg import norm
from scipy.sparse.linalg import svds




def function_loss(R: np.ndarray, P: np.ndarray) -> float:
    '''
    Computes the  Mean Squared Error (MSE) between the non-zero elements
    of R and the corresponding elements in P.

    Parameters:
    - R: The actual matrix.
    - P: The predicted matrix.

    Returns:
    - The MSE (actual matrix - predicted one)^2.
    '''
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
        return 0.0 # Avoiding division by zero if R is all zeros
    
    mse = np.sum(squared_error) / N
    
    # Returning Mean Squared Error
    return mse 


def gradient(R: np.ndarray, P: np.ndarray) -> np.ndarray:
    known_indices = R.nonzero()
    N = len(known_indices[0])
    if N == 0:
        return np.zeros_like(P)
    
    grad = np.zeros_like(P, dtype=float)
    grad[known_indices] = 2.0 * (P[known_indices] - R[known_indices]) / N
    return grad


def LMO (grad: np.ndarray, delta: float) -> np.ndarray:
    '''
    LMO - Linear Minimization Oracle
    it finds the S matrix that minimizes the scalar product <gradient , S> under the 
    limit ||S||_* < delta.

    Param: 
    -   grad: gradient matrix
    -   delta: radius of the nuclear norm
     '''
    # Calcoliamo solo il primo valore singolare (k=1) più grande ('LM')
    # Questo è molto più efficiente di una SVD completa
    u, s, vt = svds(grad, k=1, which='LM')
    
    # Ricostruiamo la matrice di rango 1
    # u e vt sono vettori, li rimodelliamo per fare il prodotto matriciale corretto
    u = u.reshape(-1, 1)
    vt = vt.reshape(1, -1)
    
    # La soluzione è -delta * u * v.T
    S = -delta * np.dot(u, vt)
    
    return S




def gamma_line_search(R: np.ndarray, P: np.ndarray, D: csr_matrix, 
                      f_loss, gamma_max: float = 1.0) -> float:
    '''
    Performs a line search to find the best step size gamma.
    '''
    P_csr = csr_matrix(P) if not issparse(P) else P
    gammas = np.linspace(0, gamma_max, 50)
    
    P_start = P_csr + 0.0 * D
    best_loss = f_loss(R, P_start)
    best_gamma = 0.0
    
    for gamma in gammas[1:]: 
        print("Testing gamma:", gamma)
        P_new_csr = P_csr + gamma * D
        gamma_loss = f_loss(R, P_new_csr.toarray() if issparse(P_new_csr) else P_new_csr)
        
        if gamma_loss < best_loss:
            print("Better gamma loss found:", gamma_loss)
            best_loss = gamma_loss
            best_gamma = gamma
    print("Chosen gamma:", best_gamma)   
    return best_gamma


def armijoRule(R: np.ndarray, P: np.ndarray, f_loss, 
                         grad: np.ndarray, D: np.ndarray, gamma_max: float = 1.0) -> float:
    '''
    Efficient Armijo backtracking line search using precomputed gradient and direction.
    
    Returns a step size gamma that satisfies the Armijo sufficient decrease condition.
    '''
    alpha = gamma_max          # initial step size
    beta  = 0.5                # backtracking factor
    sigma = 1e-4               # Armijo constant (0 < sigma << 1)
    min_alpha = 1e-10          # safety threshold

    current_loss = f_loss(R, P)
    grad_dot_D = np.sum(grad * D)               # directional derivative

    # If not a descent direction → reject step (numerical safety)
    if grad_dot_D >= 0:
        return 0.0

    # Backtracking loop
    while alpha >= min_alpha:
        P_test = P + alpha * D
        new_loss = f_loss(R, P_test)
        
        # Armijo condition: sufficient decrease
        if new_loss <= current_loss + sigma * alpha * grad_dot_D:
            return alpha                                    
        
        alpha *= beta                                       # reduce step

    return 0.0  # step became too small → reject move


def FW_standard(R: np.ndarray, delta: float, max_iter: int = 200, tol: float = 1e-6,
                init_type: str = 'zeros', ls_method: str = 'armijo') -> tuple:
    '''
    Clean and correct Frank-Wolfe for nuclear norm constrained matrix completion.
    
    Parameters:
        R           : observed rating matrix (zeros = missing)
        delta       : nuclear norm budget ||P||_* <= delta
        max_iter    : maximum number of iterations
        tol         : stop when duality gap < tol
        init_type   : 'zeros' or 'random' (projected random init)
        ls_method   : 'armijo' (fast) or 'grid' (your original slow search)
        
    Returns:
        P_final, loss_history, gap_history
    '''
    m, n = R.shape

    # --- 1. Initialization ---
    if init_type == 'random':
        X = np.random.randn(m, n)
        U, s, Vt = svd(X, full_matrices=False)
        s = delta * s / (np.sum(s) + 1e-15)
        P = U @ np.diag(s) @ Vt
    else:
        P = np.zeros((m, n), dtype=float)

    history = []
    gap_history = []

    for k in range(max_iter):
        # 2. Gradient
        grad = gradient(R, P)
        
        # 3. LMO → extreme point S
        S = LMO(grad, delta)
        
        # 4. Direction
        D = S - P
        
        # 5. DUALITY GAP
        # gap = <grad, P - S> ≥ 0, measures distance to optimality
        gap = np.sum(grad * (P - S))
        gap_history.append(gap)
        
        # Stop if converged
        if gap < tol:
            break
        
        # 6. Line search
        if ls_method == 'armijo':
            gamma = armijoRule(R, P, function_loss, grad, D, gamma_max=1.0)
        elif ls_method == 'grid':
            D_sparse = csr_matrix(D)
            gamma = gamma_line_search(R, P, D_sparse, function_loss, gamma_max=2.0)
        else:
            raise ValueError("LS must be 'armijo' or 'grid'")
        
        if gamma <= 0.0:
            raise ValueError("Optimal reached or numerical issue found")
            break 
            
        
        P = P + gamma * D
        
        loss = function_loss(R, P)
        history.append(loss)
    
    return P, history, gap_history





