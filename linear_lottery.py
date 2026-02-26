import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional

def run_linear_lottery(u: np.ndarray, L: float, k: int, 
                          tol: float = 1e-9, max_iter: int = 1000) -> np.ndarray:
    """
    Compute allocation probabilities using direct definition with binary search for tau.
    
    Parameters: u (utilities), L (Lipschitz constant), k (budget), tol, max_iter
    Returns: p (allocation probabilities)
    """
    n = len(u)
    
    # Binary search for tau
    tau_min = L * np.min(u) - 1.0
    tau_max = L * np.max(u) + 1.0
    
    for _ in range(max_iter):
        tau = (tau_min + tau_max) / 2.0
        p = np.clip(L * u - tau, 0, 1)
        sum_p = np.sum(p)
        
        if abs(sum_p - k) < tol:
            return p
        elif sum_p > k:
            tau_min = tau
        else:
            tau_max = tau
    
    # Final adjustment to ensure exact constraint
    p = np.clip(L * u - tau, 0, 1)
    return p


def fairness_lp(u: np.ndarray, L: float, k: int) -> Tuple[np.ndarray, dict]:
    """
    Compute allocation probabilities using linear programming formulation.
    
    Solves: max p^T u  s.t. |p_i - p_j| <= L|u_i - u_j|, sum(p) = k, p in [0,1]^n
    
    Parameters: u (utilities), L (Lipschitz constant), k (budget)
    Returns: (p, result_info)
    """
    n = len(u)
    
    # We want to maximize p^T u, so minimize -p^T u
    c = -u
    
    # Build Lipschitz constraints: |p_i - p_j| <= L * |u_i - u_j|
    # This is equivalent to:
    #   p_i - p_j <= L * |u_i - u_j|
    #   p_j - p_i <= L * |u_i - u_j|
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(i + 1, n):
            diff_u = abs(u[i] - u[j])
            
            # Constraint: p_i - p_j <= L * |u_i - u_j|
            constraint = np.zeros(n)
            constraint[i] = 1
            constraint[j] = -1
            A_ub.append(constraint)
            b_ub.append(L * diff_u)
            
            # Constraint: p_j - p_i <= L * |u_i - u_j|
            constraint = np.zeros(n)
            constraint[j] = 1
            constraint[i] = -1
            A_ub.append(constraint)
            b_ub.append(L * diff_u)
    
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    
    # Equality constraint: sum(p) = k
    A_eq = np.ones((1, n))
    b_eq = np.array([k])
    
    # Bounds: 0 <= p_i <= 1
    bounds = [(0, 1) for _ in range(n)]
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method='highs')
    
    if not result.success:
        raise ValueError(f"LP optimization failed: {result.message}")
    
    return result.x, {
        'success': result.success,
        'objective': -result.fun,  # Convert back to maximization
        'message': result.message
    }

def run_comparison():
    """
    Compare the two methods for computing allocation probabilities.
    """

    def compare_methods(u: np.ndarray, L: float, k: int) -> Tuple[bool, Tuple[np.ndarray, np.ndarray]]:
        """
        Compare the two methods for computing allocation probabilities.
        
        Parameters: u (utilities), L (Lipschitz constant), k (budget)
        """
        p_direct = run_linear_lottery(u, L, k)
        p_lp, _ = fairness_lp(u, L, k)
        
        # check if results are close
        if np.allclose(p_direct, p_lp, atol=1e-6):
            return True, (p_direct, p_lp)
        else:
            return False, (p_direct, p_lp)

    # Check that methods match on random utilities
    np.random.seed(42)
    n = 100
    L_values = [0.1, 0.25, 0.5, 0.75]
    k = 10
    trials = 100

    for L in L_values:
        consistent_count = 0
        first_mismatch_shown = False
        for t in range(trials):
            u = np.random.rand(n)
            consistent, results = compare_methods(u, L, k)
            if consistent:
                consistent_count += 1
            elif not first_mismatch_shown:
                p_direct, p_lp = results
                print(f"Mismatch on trial {t} (L={L}):")
                print("p_direct[:10] =", p_direct[:10])
                print("p_lp[:10]     =", p_lp[:10])
                first_mismatch_shown = True
        print(f"L={L}: {consistent_count}/{trials} trials consistent.")
    

if __name__ == "__main__":
    run_comparison()