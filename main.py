import autograd.numpy as np
import pandas as pd
from newton import multivariate_newton
    
def f(x):
    return np.array([(x[0] - 1)**2 + (x[1] + 2)**2 + (x[2])**2 - 25,
                    (x[0] + 2) ** 2 + (x[1] - 2) ** 2 + (x[2] + 1) ** 2 - 25, 
                    (x[0] - 4) ** 2 + (x[1] + 2) ** 2 + (x[2] - 3) ** 2 - 25])    

df = pd.DataFrame(columns=['x0', 'root', 'iterations', 'error'])
x0 = np.array([0.1, 0.2, 0.3])

solver = multivariate_newton(f, x0, 1e-10, verbose=True)
solver.solve()

def find_roots():
    roots = []
    for i in np.arange(0,5,0.1):
        for j in np.arange(0,5,0.1):
            for k in np.arange(0,5,0.1):
                x0 = np.array([i, j, k])
                solver = multivariate_newton(f, x0, 1e-10)
                root = solver.solve()
                
                print(f"Trying with x0 = {x0} ; Root = {root} ; No. of iterations = {solver.get_iterations()} ; Error = {solver.get_error()}")
    
                if root is not None:
                    roots.append(root)
                    df.loc[len(df)] = [x0, root, solver.get_iterations(), solver.get_error()]
    return roots



find_roots()
print(df)
df.to_excel('roots.xlsx')