# newton-multivariate-method

# Newton Multivariate Method in Python

This project implements the Newton multivariate method in Python. The Newton multivariate method is a root-finding algorithm that produces successively better approximations to the roots (or zeroes) of a real-valued function.

## Requirements

- Python 3.6+
- Autograd

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/palaniappan-r/newton-multivariate-method.git
cd newton-multivariate-method
pip install -r requirements.txt
```

## Example Usage

```python
from newton import multivariate_newton

def f(x):
    return np.array([(x[0] - 1)**2 + (x[1] + 2)**2 + (x[2])**2 - 25,
                    (x[0] + 2) ** 2 + (x[1] - 2) ** 2 + (x[2] + 1) ** 2 - 25, 
                    (x[0] - 4) ** 2 + (x[1] + 2) ** 2 + (x[2] - 3) ** 2 - 25])    

x0 = np.array([0.1, 0.2, 0.3])

solver = multivariate_newton(f, x0, 1e-10, verbose=True)
root = solver.solve()
```