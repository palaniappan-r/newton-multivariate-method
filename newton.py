import autograd.numpy as np
import autograd

class multivariate_newton:
    
    def __init__(self, func_f, initial_guess, tolerance, max_iterations=1000 , verbose=False):
        self.func_f = func_f
        self.initial_guess = initial_guess
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iterations = 0
        self.solution = None
        self.error = None
        self.jac = autograd.jacobian(func_f)
        self.verbose = verbose
                    
    def one_iteration(self, x_n):
        try:
            jac_inv = np.linalg.inv(self.jac(x_n))
            x_n1 = x_n - jac_inv.dot(self.func_f(x_n))
            return x_n1
        except np.linalg.LinAlgError:
            print("Error: Singular Jacobian")
            return None
    
    def solve(self):
        x_n = self.initial_guess
        while self.iterations < self.max_iterations:
            x_n1 = self.one_iteration(x_n)
            
            if x_n1 is None:
                return None
            
            self.error = np.linalg.norm(x_n1 - x_n)
            if self.error < self.tolerance:
                self.solution = x_n1
                break
            x_n = x_n1
            
            if self.verbose:
                print(f"Iteration: {self.iterations} \tX: {x_n} \tError: {self.get_error()}")
                
            self.iterations += 1
        return self.solution
    
    def get_error(self):
        return self.error
    
    def get_iterations(self):                
        return self.iterations
    
def f(x):
    return np.array([(x[0] - 1)**2 + (x[1] + 2)**2 + (x[2])**2 - 25,
                    (x[0] + 2) ** 2 + (x[1] - 2) ** 2 + (x[2] + 1) ** 2 - 25, 
                    (x[0] - 4) ** 2 + (x[1] + 2) ** 2 + (x[2] - 3) ** 2 - 25])    

# x0 = np.array([0.1, 0.2, 0.3])

# solver = multivariate_newton(f, x0, 1e-10, verbose=True)
# solver.solve()

