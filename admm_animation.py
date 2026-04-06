import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
from admm_solver import ADMMSolver
import warnings
warnings.filterwarnings('ignore')

class AnimatedADMMSolver(ADMMSolver):
    """Extended ADMM solver with animation-friendly iteration tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_snapshots = []
        
    def store_snapshot(self):
        """Store current state for animation"""
        snapshot = {
            'iteration': len(self.iteration_snapshots),
            'x': self.x.copy(),
            'z1': self.z1.copy(),
            'z2': self.z2.copy(),
            'z3': self.z3.copy(),
            'u1': self.u1.copy(),
            'u2': self.u2.copy(),
            'u3': self.u3.copy(),
            'objective': self.history['total_objective'][-1] if self.history['total_objective'] else 0,
            'constraint_violation': self.history['constraint_violation'][-1] if self.history['constraint_violation'] else 0,
            'primal_residual': self.history['primal_residual'][-1] if self.history['primal_residual'] else 0,
            'dual_residual': self.history['dual_residual'][-1] if self.history['dual_residual'] else 0,
        }
        self.iteration_snapshots.append(snapshot)
    
    def solve_animated(self, max_iter=100, tol=1e-4, skip_frames=2):
        """Solve with snapshot storage for animation (every skip_frames iterations)"""
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
            
            # Store snapshot for animation every skip_frames iterations
            if k % skip_frames == 0:
                self.store_snapshot()
            
            if self.verbose and k % 10 == 0:
                print(f"Iteration {k:3d}: Obj={total_obj:.6f} | Constraint={constraint_violation:.6e}")
            
            # Check convergence
            if constraint_violation < tol and primal_residual < tol and dual_residual < tol:
                if self.verbose:
                    print(f"Converged at iteration {k}")
                self.store_snapshot()  # Store final state
                break
        
        # Ensure final state is stored
        if (k % skip_frames != 0):
            self.store_snapshot()
        
        return self.x, self.z1, self.z2, self.z3


class ADMMAnimator:
    """Create animations of ADMM algorithm progress"""
    
    def __init__(self, solver):
        self.solver = solver
        self.snapshots = solver.iteration_snapshots
        
    def create_animation(self, output_file='admm_animation.gif', fps=5):
        """Create animated visualization of ADMM convergence"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        ax_graph = fig.add_subplot(gs[0:2, 0:2])
        ax_node_values = fig.add_subplot(gs[0, 2:4])
        ax_objective = fig.add_subplot(gs[1, 2])
        ax_constraint = fig.add_subplot(gs[1, 3])
        ax_x_vs_z = fig.add_subplot(gs[2, 0:2])
        ax_residuals = fig.add_subplot(gs[2, 2])
        ax_stats = fig.add_subplot(gs[2, 3])
        
        # Precompute node positions for consistent layout
        pos = nx.spring_layout(self.solver.graph, seed=42, k=2, iterations=50)
        
        def update_frame(frame_idx):
            # Clear all axes
            for ax in [ax_graph, ax_node_values, ax_objective, ax_constraint, 
                      ax_x_vs_z, ax_residuals, ax_stats]:
                ax.clear()
            
            if frame_idx >= len(self.snapshots):
                frame_idx = len(self.snapshots) - 1
            
            snapshot = self.snapshots[frame_idx]
            iteration = snapshot['iteration']
            x = snapshot['x']
            z1 = snapshot['z1']
            z2 = snapshot['z2']
            z3 = snapshot['z3']
            
            # ========== Graph Visualization ==========
            ax_graph.set_title(f'Network Graph & Node Values\n(Iteration {iteration})', 
                              fontsize=12, fontweight='bold')
            
            # Draw network
            nx.draw_networkx_edges(self.solver.graph, pos, ax=ax_graph, 
                                  alpha=0.2, width=0.5, edge_color='gray')
            
            # Color nodes by x values
            node_colors = x
            nodes = nx.draw_networkx_nodes(self.solver.graph, pos, ax=ax_graph,
                                          node_color=node_colors, node_size=500,
                                          cmap='RdBu_r', vmin=-1.5, vmax=1.5)
            
            # Draw node labels
            labels = {i: f"{i}" for i in range(self.solver.n_nodes)}
            nx.draw_networkx_labels(self.solver.graph, pos, labels, ax=ax_graph,
                                   font_size=7, font_weight='bold')
            
            ax_graph.set_xlim(-1.3, 1.3)
            ax_graph.set_ylim(-1.3, 1.3)
            ax_graph.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(nodes, ax=ax_graph, fraction=0.046, pad=0.04)
            cbar.set_label('x values', rotation=270, labelpad=15)
            
            # ========== Node Value Evolution ==========
            ax_node_values.set_title('z₁, z₂, z₃ Values by Node', fontsize=11, fontweight='bold')
            
            x_pos = np.arange(self.solver.n_nodes)
            width = 0.25
            
            ax_node_values.bar(x_pos - width, z1, width, label='z₁', alpha=0.8, color='#e74c3c')
            ax_node_values.bar(x_pos, z2, width, label='z₂', alpha=0.8, color='#3498db')
            ax_node_values.bar(x_pos + width, z3, width, label='z₃', alpha=0.8, color='#2ecc71')
            
            ax_node_values.set_xlabel('Node Index', fontsize=10)
            ax_node_values.set_ylabel('Value', fontsize=10)
            ax_node_values.legend(fontsize=9, loc='upper right')
            ax_node_values.grid(True, alpha=0.3, axis='y')
            ax_node_values.set_xlim(-0.5, self.solver.n_nodes - 0.5)
            
            # ========== Objective Function History ==========
            ax_objective.set_title('Objective', fontsize=11, fontweight='bold')
            
            obj_history = self.solver.history['total_objective'][:iteration+1]
            ax_objective.semilogy(obj_history, 'b-', linewidth=2, marker='o', markersize=3)
            ax_objective.axvline(frame_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax_objective.set_xlabel('Frame', fontsize=9)
            ax_objective.set_ylabel('Objective', fontsize=9)
            ax_objective.grid(True, alpha=0.3)
            ax_objective.tick_params(labelsize=8)
            
            # ========== Constraint Violation ==========
            ax_constraint.set_title('Constraint Violation', fontsize=11, fontweight='bold')
            
            const_history = self.solver.history['constraint_violation'][:iteration+1]
            ax_constraint.semilogy(const_history, 'r-', linewidth=2, marker='s', markersize=3)
            ax_constraint.axvline(frame_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax_constraint.set_xlabel('Frame', fontsize=9)
            ax_constraint.set_ylabel('||x - z||', fontsize=9)
            ax_constraint.grid(True, alpha=0.3)
            ax_constraint.tick_params(labelsize=8)
            
            # ========== x vs z Coupling ==========
            ax_x_vs_z.set_title('Variable Coupling: x vs z (should converge to identity)', 
                                fontsize=11, fontweight='bold')
            
            z_avg = (z1 + z2 + z3) / 3
            
            ax_x_vs_z.scatter(x, z_avg, s=100, alpha=0.6, color='purple', edgecolors='black', linewidth=1)
            
            # Plot diagonal for reference
            lim = max(np.abs(x).max(), np.abs(z_avg).max()) * 1.1
            ax_x_vs_z.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=2, label='x = z')
            
            ax_x_vs_z.set_xlabel('x', fontsize=10)
            ax_x_vs_z.set_ylabel('mean(z)', fontsize=10)
            ax_x_vs_z.grid(True, alpha=0.3)
            ax_x_vs_z.set_xlim(-lim, lim)
            ax_x_vs_z.set_ylim(-lim, lim)
            ax_x_vs_z.legend(fontsize=9)
            ax_x_vs_z.set_aspect('equal')
            
            # Add node labels to scatter
            for i, (xi, zi) in enumerate(zip(x, z_avg)):
                if i % 3 == 0:  # Label every 3rd point to avoid clutter
                    ax_x_vs_z.annotate(str(i), (xi, zi), fontsize=7, alpha=0.5)
            
            # ========== Residuals ==========
            ax_residuals.set_title('Residuals', fontsize=11, fontweight='bold')
            
            prim_history = self.solver.history['primal_residual'][:iteration+1]
            dual_history = self.solver.history['dual_residual'][:iteration+1]
            
            ax_residuals.semilogy(prim_history, 'b-', linewidth=2, marker='o', 
                                 markersize=3, label='Primal', alpha=0.7)
            ax_residuals.semilogy(dual_history, 'r-', linewidth=2, marker='s', 
                                 markersize=3, label='Dual', alpha=0.7)
            ax_residuals.axvline(frame_idx, color='green', linestyle='--', alpha=0.5, linewidth=2)
            ax_residuals.set_xlabel('Frame', fontsize=9)
            ax_residuals.set_ylabel('Residual', fontsize=9)
            ax_residuals.legend(fontsize=8)
            ax_residuals.grid(True, alpha=0.3)
            ax_residuals.tick_params(labelsize=8)
            
            # ========== Statistics Panel ==========
            ax_stats.axis('off')
            
            stats_text = f"""
ITERATION STATISTICS

Iteration: {iteration}

Objective: {snapshot['objective']:.6f}

Constraint Violation:
  {snapshot['constraint_violation']:.6e}

Primal Residual:
  {snapshot['primal_residual']:.6e}

Dual Residual:
  {snapshot['dual_residual']:.6e}

||x||: {np.linalg.norm(x):.4f}
||z₁||: {np.linalg.norm(z1):.4f}
||z₂||: {np.linalg.norm(z2):.4f}
||z₃||: {np.linalg.norm(z3):.4f}

Max ||x - z₁||:
  {np.linalg.norm(x - z1):.6e}
Max ||x - z₂||:
  {np.linalg.norm(x - z2):.6e}
Max ||x - z₃||:
  {np.linalg.norm(x - z3):.6e}

Progress:
  {frame_idx + 1}/{len(self.snapshots)}
"""
            
            ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            return [ax_graph, ax_node_values, ax_objective, ax_constraint, 
                   ax_x_vs_z, ax_residuals, ax_stats]
        
        # Create animation
        anim = FuncAnimation(fig, update_frame, frames=len(self.snapshots),
                            interval=200, blit=False, repeat=True)
        
        # Save animation
        print(f"Creating animation with {len(self.snapshots)} frames...")
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
        
        return anim, fig
    
    def create_comparison_animation(self, output_file='admm_comparison.gif', fps=5):
        """Create side-by-side comparison of network and convergence metrics"""
        
        fig = plt.figure(figsize=(18, 10))
        
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        ax_graph = fig.add_subplot(gs[:, 0])
        ax_history = fig.add_subplot(gs[0, 1:])
        ax_dual_space = fig.add_subplot(gs[1, 1])
        ax_info = fig.add_subplot(gs[1, 2])
        
        # Precompute positions
        pos = nx.spring_layout(self.solver.graph, seed=42, k=2, iterations=50)
        
        def update_frame(frame_idx):
            if frame_idx >= len(self.snapshots):
                frame_idx = len(self.snapshots) - 1
            
            snapshot = self.snapshots[frame_idx]
            x = snapshot['x']
            z1 = snapshot['z1']
            z2 = snapshot['z2']
            z3 = snapshot['z3']
            iteration = snapshot['iteration']
            
            # ========== Graph ==========
            ax_graph.clear()
            ax_graph.set_title(f'Network Graph\n(Iteration {iteration})', 
                              fontsize=13, fontweight='bold')
            
            nx.draw_networkx_edges(self.solver.graph, pos, ax=ax_graph, 
                                  alpha=0.15, width=1, edge_color='gray')
            
            node_colors = x
            nodes = nx.draw_networkx_nodes(self.solver.graph, pos, ax=ax_graph,
                                          node_color=node_colors, node_size=800,
                                          cmap='RdBu_r', vmin=-1.5, vmax=1.5,
                                          edgecolors='black', linewidths=1.5)
            
            labels = {i: f"{i}" for i in range(self.solver.n_nodes)}
            nx.draw_networkx_labels(self.solver.graph, pos, labels, ax=ax_graph,
                                   font_size=8, font_weight='bold')
            
            ax_graph.set_xlim(-1.4, 1.4)
            ax_graph.set_ylim(-1.4, 1.4)
            ax_graph.axis('off')
            
            cbar = plt.colorbar(nodes, ax=ax_graph, fraction=0.046, pad=0.04)
            cbar.set_label('x values', rotation=270, labelpad=15, fontsize=10)
            
            # ========== Convergence History ==========
            ax_history.clear()
            ax_history.set_title('Convergence Metrics Over Time', fontsize=12, fontweight='bold')
            
            iterations = np.arange(iteration + 1)
            obj_history = self.solver.history['total_objective'][:iteration+1]
            const_history = self.solver.history['constraint_violation'][:iteration+1]
            
            ax_history.semilogy(iterations, obj_history, 'b-', linewidth=2.5, 
                               label='Objective', marker='o', markersize=4, markevery=max(1, iteration//10))
            ax_history.semilogy(iterations, const_history, 'r-', linewidth=2.5, 
                               label='Constraint Violation', marker='s', markersize=4, markevery=max(1, iteration//10))
            
            ax_history.axvline(frame_idx, color='green', linestyle='--', alpha=0.7, linewidth=2.5, label='Current Frame')
            ax_history.set_xlabel('Iteration', fontsize=11)
            ax_history.set_ylabel('Value (log scale)', fontsize=11)
            ax_history.legend(fontsize=10, loc='best')
            ax_history.grid(True, alpha=0.3, which='both')
            
            # ========== Dual Space (z1 vs z2 vs z3) ==========
            ax_dual_space.clear()
            ax_dual_space.set_title('z₁ vs z₂ vs z₃ Distribution', fontsize=12, fontweight='bold')
            
            z_avg = (z1 + z2 + z3) / 3
            ax_dual_space.scatter(z1, z2, s=120, alpha=0.5, color='red', label='z₁ vs z₂', edgecolors='darkred', linewidth=1)
            ax_dual_space.scatter(z2, z3, s=120, alpha=0.5, color='blue', label='z₂ vs z₃', edgecolors='darkblue', linewidth=1)
            ax_dual_space.scatter(z1, z3, s=80, alpha=0.3, color='green', label='z₁ vs z₃', edgecolors='darkgreen', linewidth=1)
            
            ax_dual_space.set_xlabel('z₁', fontsize=11)
            ax_dual_space.set_ylabel('z₂, z₃', fontsize=11)
            ax_dual_space.legend(fontsize=9, loc='best')
            ax_dual_space.grid(True, alpha=0.3)
            
            # ========== Info Panel ==========
            ax_info.clear()
            ax_info.axis('off')
            
            coupling_error = np.linalg.norm(x - z1) + np.linalg.norm(x - z2) + np.linalg.norm(x - z3)
            
            info_text = f"""
╔════════════════════════════╗
║   ADMM STATUS REPORT       ║
╚════════════════════════════╝

Frame: {frame_idx + 1} / {len(self.snapshots)}
Iteration: {iteration}

OBJECTIVE:
  Current: {snapshot['objective']:.4f}
  Init:    {self.snapshots[0]['objective']:.4f}
  Change:  {snapshot['objective'] - self.snapshots[0]['objective']:.4f}

FEASIBILITY:
  x-z₁ coupling: {np.linalg.norm(x - z1):.2e}
  x-z₂ coupling: {np.linalg.norm(x - z2):.2e}
  x-z₃ coupling: {np.linalg.norm(x - z3):.2e}
  Total error:   {coupling_error:.2e}

CONVERGENCE:
  Primal res:  {snapshot['primal_residual']:.2e}
  Dual res:    {snapshot['dual_residual']:.2e}

NORMS:
  ||x||:  {np.linalg.norm(x):.4f}
  ||z₁||: {np.linalg.norm(z1):.4f}
  ||z₂||: {np.linalg.norm(z2):.4f}
  ||z₃||: {np.linalg.norm(z3):.4f}
"""
            
            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Create animation
        anim = FuncAnimation(fig, update_frame, frames=len(self.snapshots),
                            interval=200, blit=False, repeat=True)
        
        # Save animation
        print(f"Creating comparison animation with {len(self.snapshots)} frames...")
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"Comparison animation saved to {output_file}")
        
        return anim, fig


if __name__ == "__main__":
    # ========== SETUP ==========
    print("=" * 80)
    print("ANIMATED ADMM SOLVER")
    print("=" * 80)
    
    # Create random problem
    np.random.seed(np.random.randint(0, 10000))  # Use new seed each time
    
    n_nodes = 25
    
    # Create random graph
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < 0.25:  # 25% edge probability
                edges.append((i, j))
    
    print(f"\nNetwork Configuration:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {len(edges)}")
    print(f"  Density: {2*len(edges)/(n_nodes*(n_nodes-1)):.2%}")
    
    # Random weights
    weights = np.random.uniform(0.5, 2.0, n_nodes)
    
    # Initialize solver
    print(f"\nInitializing ADMM Solver...")
    solver = AnimatedADMMSolver(
        n_nodes=n_nodes,
        edge_list=edges,
        weights=weights,
        lambda_param=0.1,
        mu=0.5,
        rho=1.0,
        verbose=True
    )
    
    # Solve with snapshots every 2 iterations
    print(f"\nSolving ADMM problem...")
    x_opt, z1_opt, z2_opt, z3_opt = solver.solve_animated(max_iter=120, tol=1e-5, skip_frames=2)
    
    print(f"\nGenerated {len(solver.iteration_snapshots)} animation frames")
    
    # Create animator
    animator = ADMMAnimator(solver)
    
    # Create main animation
    print(f"\n{'='*80}")
    print("GENERATING ANIMATIONS")
    print(f"{'='*80}")
    
    anim1, fig1 = animator.create_animation(
        output_file='/mnt/user-data/outputs/admm_animation_detailed.gif', 
        fps=5
    )
    
    # Create comparison animation
    anim2, fig2 = animator.create_comparison_animation(
        output_file='/mnt/user-data/outputs/admm_animation_comparison.gif', 
        fps=5
    )
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Final Objective: {solver.history['total_objective'][-1]:.6f}")
    print(f"Final Constraint Violation: {solver.history['constraint_violation'][-1]:.6e}")
    print(f"Total Iterations: {len(solver.history['total_objective'])}")
    print(f"Animation Frames: {len(solver.iteration_snapshots)}")
    print(f"\nAnimations saved to /mnt/user-data/outputs/")
    print(f"  - admm_animation_detailed.gif (detailed 4-panel view)")
    print(f"  - admm_animation_comparison.gif (network + metrics)")
    
    # Close figures to save memory
    plt.close('all')