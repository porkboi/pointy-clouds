import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import defaultdict
import networkx as nx

class ADMMSolver:
    """
    ADMM solver for the multi-objective optimization problem:
    min_{x, z} f(x) + g(z)
    subject to: x = z
    
    where:
    - f(x) = sum_i TD(x_i)
    - g(z) = g_1(z_1) + g_2(z_2) + g_3(z_3)
    """
    
    def __init__(self, n_nodes, edge_list, weights, lambda_param=0.1, mu=0.1, rho=1.0, verbose=True):
        """
        Initialize ADMM solver.
        
        Args:
            n_nodes: Number of nodes
            edge_list: List of edges (tuples)
            weights: Node weights w(i)
            lambda_param: Regularization parameter for g_2
            mu: Regularization parameter for g_3
            rho: ADMM penalty parameter
            verbose: Print progress
        """
        self.n_nodes = n_nodes
        self.edge_list = edge_list
        self.weights = weights
        self.lambda_param = lambda_param
        self.mu = mu
        self.rho = rho
        self.verbose = verbose
        
        # Build adjacency structure
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(n_nodes))
        self.graph.add_edges_from(edge_list)
        self.adjacency = {i: list(self.graph.neighbors(i)) for i in range(n_nodes)}
        
        # Initialize variables with random values for non-trivial problem
        self.x = np.random.randn(n_nodes) * 0.5
        self.z1 = np.random.randn(n_nodes) * 0.5
        self.z2 = np.random.randn(n_nodes) * 0.5
        self.z3 = np.random.randn(n_nodes) * 0.5
        self.u1 = np.zeros(n_nodes)
        self.u2 = np.zeros(n_nodes)
        self.u3 = np.zeros(n_nodes)
        
        # History tracking
        self.history = {
            'f_x': [],
            'g1_z1': [],
            'g2_z2': [],
            'g3_z3': [],
            'total_objective': [],
            'constraint_violation': [],
            'dual_residual': [],
            'primal_residual': [],
            'x_norm': [],
            'z_norm': []
        }
    
    def TD(self, x):
        """
        Total delay function: TD(x) = sum_i w_in(i) * w_out(i) * d(i)
        Approximated as non-convex function for realistic behavior
        """
        # Add a more realistic non-convex component
        return np.sum(x**2 + 0.5 * np.sin(2*np.pi*x))
    
    def grad_TD(self, x):
        """Gradient of TD function"""
        return 2 * x + np.cos(2*np.pi*x)
    
    def f(self, x):
        """Objective function f(x) = sum_i TD(x_i)"""
        return self.TD(x)
    
    def g1(self, z1):
        """
        Path-based regularization term
        g_1(z_1) = (1/2) * sum_j (w(i)*w(j)/N^2) * sum_k ||z_{1,k} - z_{1,k+1}||_2
        Approximated as smoothness penalty
        """
        penalty = 0.0
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                w_prod = self.weights[i] * self.weights[j] / (self.n_nodes ** 2)
                penalty += 0.5 * w_prod * np.linalg.norm(z1[i] - z1[j]) ** 2
        return penalty
    
    def g2(self, z2):
        """L2 regularization: g_2(z_2) = lambda * ||z_2||_2^2"""
        return self.lambda_param * np.linalg.norm(z2) ** 2
    
    def g3(self, z3):
        """
        Graph Laplacian regularization: g_3(z_3) = mu * sum_{(i,j) in E} ||z_{3,i} - z_{3,j}||_2
        """
        penalty = 0.0
        for i, j in self.edge_list:
            penalty += self.mu * np.linalg.norm(z3[i] - z3[j])
        return penalty
    
    def g(self, z1, z2, z3):
        """Total penalty term"""
        return self.g1(z1) + self.g2(z2) + self.g3(z3)
    
    def prox_f(self, v, stepsize):
        """
        Proximal operator of f: prox_{f,stepsize}(v)
        For smooth f, use gradient descent step: v - stepsize * grad_f(v)
        """
        return v - stepsize * self.grad_TD(v)
    
    def prox_g2(self, v, stepsize):
        """
        Proximal operator of g_2(z) = lambda * ||z||_2^2
        prox_{g_2, stepsize}(v) = v / (1 + 2*stepsize*lambda)
        """
        return v / (1.0 + 2.0 * stepsize * self.lambda_param)
    
    def prox_g3(self, v, stepsize):
        """
        Proximal operator of g_3 (graph Laplacian)
        Solved via iterative refinement (gradient descent on augmented problem)
        """
        z3 = v.copy()
        # Few iterations of gradient descent
        for _ in range(5):
            grad = np.zeros(self.n_nodes)
            for i, j in self.edge_list:
                diff = z3[i] - z3[j]
                norm_diff = np.linalg.norm(diff) + 1e-8
                grad[i] += self.mu * diff / norm_diff
                grad[j] -= self.mu * diff / norm_diff
            z3 = z3 - 0.1 * stepsize * grad
        return z3
    
    def prox_g1(self, v, stepsize):
        """
        Proximal operator of g_1 (smoothness)
        Approximated via gradient descent on smoothness penalty
        """
        z1 = v.copy()
        for _ in range(5):
            grad = np.zeros(self.n_nodes)
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        w_prod = self.weights[i] * self.weights[j] / (self.n_nodes ** 2)
                        diff = z1[i] - z1[j]
                        norm_diff = np.linalg.norm(diff) + 1e-8
                        grad[i] += 0.5 * w_prod * diff / norm_diff
            z1 = z1 - 0.1 * stepsize * grad
        return z1
    
    def update_x(self):
        """x-update: x^{k+1} = prox_{f, rho}(z - u)"""
        v = self.z1 - self.u1/self.rho
        self.x = self.prox_f(v, 1.0/self.rho)
    
    def update_z1(self):
        """z_1-update: z_1^{k+1} = prox_{g_1, rho}(x + u_1/rho)"""
        v = self.x + self.u1/self.rho
        self.z1 = self.prox_g1(v, 1.0/self.rho)
    
    def update_z2(self):
        """z_2-update: z_2^{k+1} = prox_{g_2, rho}(x + u_2/rho)"""
        v = self.x + self.u2/self.rho
        self.z2 = self.prox_g2(v, 1.0/self.rho)
    
    def update_z3(self):
        """z_3-update: z_3^{k+1} = prox_{g_3, rho}(x + u_3/rho)"""
        v = self.x + self.u3/self.rho
        self.z3 = self.prox_g3(v, 1.0/self.rho)
    
    def update_duals(self):
        """Dual variable updates"""
        self.u1 = self.u1 + self.rho * (self.x - self.z1)
        self.u2 = self.u2 + self.rho * (self.x - self.z2)
        self.u3 = self.u3 + self.rho * (self.x - self.z3)
    
    def compute_residuals(self, prev_z1, prev_z2, prev_z3, prev_x):
        """Compute primal and dual residuals for convergence checking"""
        # Primal residuals (constraint violation)
        r1 = np.linalg.norm(self.x - self.z1)
        r2 = np.linalg.norm(self.x - self.z2)
        r3 = np.linalg.norm(self.x - self.z3)
        primal_residual = r1 + r2 + r3
        
        # Dual residuals
        s1 = self.rho * np.linalg.norm(self.z1 - prev_z1)
        s2 = self.rho * np.linalg.norm(self.z2 - prev_z2)
        s3 = self.rho * np.linalg.norm(self.z3 - prev_z3)
        dual_residual = s1 + s2 + s3
        
        return primal_residual, dual_residual
    
    def solve(self, max_iter=100, tol=1e-4):
        """
        Solve the ADMM problem
        """
        for k in range(max_iter):
            # Store previous values for residual computation
            prev_x = self.x.copy()
            prev_z1 = self.z1.copy()
            prev_z2 = self.z2.copy()
            prev_z3 = self.z3.copy()
            
            # ADMM updates
            self.update_x()
            self.update_z1()
            self.update_z2()
            self.update_z3()
            self.update_duals()
            
            # Compute objective and residuals
            f_val = self.f(self.x)
            g_val = self.g(self.z1, self.z2, self.z3)
            total_obj = f_val + g_val
            
            constraint_violation = np.linalg.norm(self.x - self.z1) + \
                                  np.linalg.norm(self.x - self.z2) + \
                                  np.linalg.norm(self.x - self.z3)
            
            primal_residual, dual_residual = self.compute_residuals(
                prev_z1, prev_z2, prev_z3, prev_x
            )
            
            # Store history
            self.history['f_x'].append(f_val)
            self.history['g1_z1'].append(self.g1(self.z1))
            self.history['g2_z2'].append(self.g2(self.z2))
            self.history['g3_z3'].append(self.g3(self.z3))
            self.history['total_objective'].append(total_obj)
            self.history['constraint_violation'].append(constraint_violation)
            self.history['primal_residual'].append(primal_residual)
            self.history['dual_residual'].append(dual_residual)
            self.history['x_norm'].append(np.linalg.norm(self.x))
            self.history['z_norm'].append(np.linalg.norm(self.z1) + np.linalg.norm(self.z2) + np.linalg.norm(self.z3))
            
            if self.verbose and k % 10 == 0:
                print(f"Iteration {k:3d}: Obj={total_obj:.6f} | " + 
                      f"Constraint={constraint_violation:.6e} | " +
                      f"Primal={primal_residual:.6e} | Dual={dual_residual:.6e}")
            
            # Check convergence
            if constraint_violation < tol and primal_residual < tol and dual_residual < tol:
                if self.verbose:
                    print(f"\nConverged at iteration {k}")
                break
        
        return self.x, self.z1, self.z2, self.z3
    
    def plot_progress(self):
        """Visualize convergence progress"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Total objective
        axes[0, 0].semilogy(self.history['total_objective'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Total Objective')
        axes[0, 0].set_title('Objective Function Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Individual components
        axes[0, 1].semilogy(self.history['f_x'], 'b-', label='f(x)', linewidth=2)
        axes[0, 1].semilogy(self.history['g1_z1'], 'r-', label='g₁(z₁)', linewidth=2)
        axes[0, 1].semilogy(self.history['g2_z2'], 'g-', label='g₂(z₂)', linewidth=2)
        axes[0, 1].semilogy(self.history['g3_z3'], 'm-', label='g₃(z₃)', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Objective Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Constraint violation
        axes[0, 2].semilogy(self.history['constraint_violation'], 'r-', linewidth=2)
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('||x - z||')
        axes[0, 2].set_title('Constraint Violation')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Primal and dual residuals
        axes[1, 0].semilogy(self.history['primal_residual'], 'b-', label='Primal', linewidth=2)
        axes[1, 0].semilogy(self.history['dual_residual'], 'r-', label='Dual', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].set_title('Primal & Dual Residuals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Variable norms
        axes[1, 1].semilogy(self.history['x_norm'], 'b-', label='||x||', linewidth=2)
        axes[1, 1].semilogy(self.history['z_norm'], 'r-', label='||z||', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Norm')
        axes[1, 1].set_title('Variable Norms')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Solution heatmap
        z_matrix = np.array([self.z1, self.z2, self.z3])
        im = axes[1, 2].imshow(z_matrix, aspect='auto', cmap='viridis')
        axes[1, 2].set_xlabel('Node Index')
        axes[1, 2].set_ylabel('z₁, z₂, z₃')
        axes[1, 2].set_title('Final Solution')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Create a small network
    np.random.seed(42)
    n_nodes = 20
    
    # Create random graph
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < 0.3:  # 30% edge probability
                edges.append((i, j))
    
    # Random weights
    weights = np.random.uniform(0.5, 2.0, n_nodes)
    
    # Initialize solver
    solver = ADMMSolver(
        n_nodes=n_nodes,
        edge_list=edges,
        weights=weights,
        lambda_param=0.1,
        mu=0.5,
        rho=1.0,
        verbose=True
    )
    
    # Solve
    print("=" * 80)
    print("ADMM Optimization Progress")
    print("=" * 80)
    x_opt, z1_opt, z2_opt, z3_opt = solver.solve(max_iter=150, tol=1e-5)
    
    # Print final solution
    print("\n" + "=" * 80)
    print("Final Solution Summary")
    print("=" * 80)
    print(f"Final objective value: {solver.history['total_objective'][-1]:.6f}")
    print(f"Final constraint violation: {solver.history['constraint_violation'][-1]:.6e}")
    print(f"Primal residual: {solver.history['primal_residual'][-1]:.6e}")
    print(f"Dual residual: {solver.history['dual_residual'][-1]:.6e}")
    print(f"\nOptimal x (first 10): {x_opt[:10]}")
    print(f"Optimal z₁ (first 10): {z1_opt[:10]}")
    print(f"Optimal z₂ (first 10): {z2_opt[:10]}")
    print(f"Optimal z₃ (first 10): {z3_opt[:10]}")
    
    # Plot progress
    fig = solver.plot_progress()
    plt.savefig('/tmp/admm_progress.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to /tmp/admm_progress.png")
    plt.show()