import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.linalg import norm

def function_loss(R: np.ndarray, P: np.ndarray) -> float:
    '''
    Computes the Root Mean Squared Error (RMSE) between the non-zero elements
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
    '''
    Computes the gradient of the loss
    with respect to the matrix P.

    Parameters:
    - R: The original matrix.
    - P: The predicted matrix.

    Returns:
    - The gradient of the MSE loss with respect to P.
    '''
    # Finding the indices of known ratings (non-zero in R)
    known_indices = R.nonzero()
    
    # N is the number of known ratings
    N = len(known_indices[0]) 
    if N == 0:
        return np.zeros_like(R)

    # Computing the difference (P - R) only for known entries
    # The gradient is 2/N * (P - R) for the known entries, 0 otherwise.
    grad = np.zeros_like(R, dtype=float)
    grad[known_indices] = (P[known_indices] - R[known_indices]) * 2 / N

    return grad


def gamma_line_search(R: np.ndarray, P: np.ndarray, D: csr_matrix, 
                      f_loss, gamma_max: float) -> float:
    '''
    Performs a line search to find the best step size gamma.
    '''

    # Converting P to CSR once for efficient sparse addition
    P_csr = csr_matrix(P) if not issparse(P) else P
    
    # Linearly spaced test gammas, including gamma=0
    gammas = np.linspace(0, gamma_max, 50) #Can be adjusted for finer search 
    
    
    # Loss at gamma=0 (the current point P)
    P_start = P_csr + 0.0 * D # P_start is just P_csr
    best_loss = f_loss(R, P_start)
    best_gamma = 0.0
    
    # Iterative Line Search (Starting from the second point)
    for gamma in gammas[1:]: 
        
        print("Testing gamma:", gamma)
        # Calculate P_new efficiently in the sparse domain
        P_new_csr = P_csr + gamma * D
        
        # Calculate the loss for this specific step size (gamma)
        gamma_loss = f_loss(R, P_new_csr)
        
        # Updating Best Result
        if gamma_loss < best_loss:
            print("Better gamma loss found:", gamma_loss, "Old gamma loss was: ")
            best_loss = gamma_loss
            best_gamma = gamma
    print(best_gamma)   
    return best_gamma



'''
Definition of Armijo Rule for gamma search -> less computational cost 
'''
def armijoRule(R: np.ndarray, P: np.ndarray, f_loss, 
               gradient, gamma_max: float = 1.0) -> float:
    
    # --- ConstantsArmijo ---
    alpha = gamma_max  
    min_alpha = 1e-10  
    sigma = 1e-4        
    beta = 0.5        

    # --- Pre-calculation ---
    current_loss = f_loss(R, P)
    grad = gradient(R, P)
    
    # Since Direction D = -grad, the slope is: grad dot -grad = -||grad||^2 (FROBENIUS NORM)
    grad_norm_sq = np.sum(np.square(grad))
    
    # --- Backtracking Loop ---
    while True:
        # Update Rule: Steepest Descent (P - alpha * grad)
        P_test = P - alpha * grad
        
        # New loss at the test point
        new_loss = f_loss(R, P_test)
        
        # Armijo Condition:
        target_loss = current_loss - sigma * alpha * grad_norm_sq
        
        if new_loss <= target_loss:
            return alpha
            
        # Shrink step size
        alpha *= beta
        
        # Safety break to prevent infinite loops
        if alpha < min_alpha:
            return 0.0
