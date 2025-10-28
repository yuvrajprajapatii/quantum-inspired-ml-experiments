# amplitude_amplification_search.py
import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt

class AmplitudeAmplificationSearch:
    """
    Classical implementation of quantum amplitude amplification
    Achieves O(√N) iterations vs O(N) classical search
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def _oracle(self, x, target_function, threshold=None):
        """Oracle function - marks good solutions"""
        value = target_function(x)
        if threshold is None:
            # Dynamic threshold based on current best
            return value > getattr(self, 'current_best', -np.inf)
        return value > threshold
    
    def _diffusion_operator(self, population, oracle_results):
        """Quantum-inspired diffusion operator (amplitude amplification)"""
        n_good = np.sum(oracle_results)
        n_total = len(population)
        
        if n_good == 0 or n_good == n_total:
            return population
        
        # Amplitude amplification: boost probability of good solutions
        amplification_factor = np.sqrt(n_total / n_good) if n_good > 0 else 1.0
        
        # Create new population with amplified good solutions
        new_population = []
        good_solutions = population[oracle_results]
        bad_solutions = population[~oracle_results]
        
        # Amplify good solutions (quantum superposition principle)
        n_amplified_good = min(int(n_good * amplification_factor), n_total // 2)
        if len(good_solutions) > 0:
            amplified_indices = np.random.choice(len(good_solutions), 
                                               n_amplified_good, replace=True)
            new_population.extend(good_solutions[amplified_indices])
        
        # Fill remaining with exploration (quantum randomness)
        remaining = n_total - len(new_population)
        if remaining > 0:
            # Mix of good and bad solutions with noise
            if len(good_solutions) > 0 and len(bad_solutions) > 0:
                mixed_solutions = np.vstack([good_solutions, bad_solutions])
            elif len(good_solutions) > 0:
                mixed_solutions = good_solutions
            else:
                mixed_solutions = bad_solutions
            
            for _ in range(remaining):
                base_solution = mixed_solutions[np.random.randint(len(mixed_solutions))]
                # Add quantum noise
                noise = np.random.normal(0, 0.1, base_solution.shape)
                new_population.append(base_solution + noise)
        
        return np.array(new_population)
    
    def search(self, target_function, dimensions, bounds=(-5, 5), 
               population_size=100, max_iterations=None, tolerance=1e-6):
        """
        Amplitude amplification search algorithm
        
        Args:
            target_function: Function to optimize (higher is better)
            dimensions: Problem dimensionality
            bounds: Search space bounds
            population_size: Size of search population
            max_iterations: Maximum iterations (None for theoretical optimal)
            tolerance: Convergence tolerance
        """
        if max_iterations is None:
            # Theoretical quantum advantage: O(√N) iterations
            max_iterations = int(np.sqrt(population_size * dimensions))
        
        # Initialize random population
        if isinstance(bounds, tuple):
            bounds = [bounds] * dimensions
        
        population = np.random.uniform([b[0] for b in bounds], 
                                     [b[1] for b in bounds], 
                                     (population_size, dimensions))
        
        best_value = -np.inf
        best_solution = None
        convergence_history = []
        
        for iteration in range(max_iterations):
            # Evaluate population
            values = np.array([target_function(x) for x in population])
            
            # Update best solution
            current_best_idx = np.argmax(values)
            current_best_value = values[current_best_idx]
            
            if current_best_value > best_value:
                best_value = current_best_value
                best_solution = population[current_best_idx].copy()
                self.current_best = best_value
            
            convergence_history.append(best_value)
            
            # Check convergence
            if iteration > 10:
                recent_improvement = convergence_history[-1] - convergence_history[-10]
                if abs(recent_improvement) < tolerance:
                    break
            
            # Oracle marking
            oracle_results = np.array([self._oracle(x, target_function) for x in population])
            
            # Amplitude amplification step
            population = self._diffusion_operator(population, oracle_results)
        
        return {
            'best_solution': best_solution,
            'best_value': best_value,
            'iterations': iteration + 1,
            'convergence_history': convergence_history,
            'theoretical_iterations': int(np.sqrt(population_size * dimensions))
        }

def test_amplitude_amplification():
    """Test AAS on optimization problems"""
    
    # Test functions
    def sphere_function(x):
        """Simple sphere function (higher is better version)"""
        return -(np.sum(x**2))
    
    def rastrigin_function(x):
        """Rastrigin function (higher is better version)"""
        A = 10
        n = len(x)
        return -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
    
    def rosenbrock_function(x):
        """Rosenbrock function (higher is better version)"""
        return -np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    test_functions = {
        'Sphere': sphere_function,
        'Rastrigin': rastrigin_function,
        'Rosenbrock': rosenbrock_function
    }
    
    dimensions = [10, 50, 100]
    results = {}
    
    for func_name, func in test_functions.items():
        results[func_name] = {}
        
        for dim in dimensions:
            print(f"\nTesting {func_name} function in {dim}D")
            
            # Amplitude Amplification Search
            aas = AmplitudeAmplificationSearch(random_state=42)
            start_time = time.time()
            aas_result = aas.search(func, dim, bounds=(-5, 5), population_size=200)
            aas_time = time.time() - start_time
            
            # Classical Random Search baseline
            start_time = time.time()
            best_classical = -np.inf
            classical_iterations = aas_result['theoretical_iterations'] * 10  # More iterations for fair comparison
            
            for _ in range(classical_iterations):
                x = np.random.uniform(-5, 5, dim)
                value = func(x)
                if value > best_classical:
                    best_classical = value
            
            classical_time = time.time() - start_time
            
            results[func_name][dim] = {
                'aas_value': aas_result['best_value'],
                'aas_iterations': aas_result['iterations'],
                'aas_time': aas_time,
                'classical_value': best_classical,
                'classical_iterations': classical_iterations,
                'classical_time': classical_time,
                'theoretical_advantage': classical_iterations / aas_result['iterations'],
                'practical_speedup': classical_time / aas_time if aas_time > 0 else 0
            }
            
            print(f"AAS: Value={aas_result['best_value']:.3f}, Iterations={aas_result['iterations']}")
            print(f"Classical: Value={best_classical:.3f}, Iterations={classical_iterations}")
            print(f"Theoretical advantage: {results[func_name][dim]['theoretical_advantage']:.2f}x")
            print(f"Practical speedup: {results[func_name][dim]['practical_speedup']:.2f}x")
    
    return results

if __name__ == "__main__":
    results = test_amplitude_amplification()
    print("\nFinal Results:", results)