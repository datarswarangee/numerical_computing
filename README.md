# MATLAB Numerical Computing – Algorithm Design & Debugging

## Conceptual Intuition
Think of this project as solving mathematical problems through structured computation.
Each computation step refines the result.
Intermediate values guide the final outcome.
The algorithm acts like a pipeline transforming inputs into stable outputs.
This project focuses on how numerical methods behave in practice, especially when errors and instability occur.

## Project Structure
File_1: Core numerical computation and matrix-based operations
File_2: Iterative parameter estimation and stability handling


## Q1: Numerical Computation & Matrix Operations

### Objective
Perform structured computations using matrix and vector operations.

### Method
- Load or define input data
- Apply mathematical transformations
- Ensure dimensional consistency across operations

### Key Concept
Efficient computation depends on correct handling of matrix dimensions.

### Mathematical View
If data is represented as:
f ∈ ℝⁿ

### Operations transform it into:
g = T(f)
Where T represents a computational mapping.

## Q2: Iterative Parameter Estimation

### Objective
Estimate parameters using an iterative update rule.

### Method
- Compute numerator (num) and denominator (den)
- Update parameter using ratio-based formula
- Repeat until convergence

### Key Concept
Each iteration improves parameter accuracy while maintaining numerical stability.

### Mathematical Form
Update rule:
μₖ = num / (den + ε)

### Where:
μₖ → parameter at iteration k
ε → small constant to prevent division by zero



## Core Challenge
Ensuring num and den remain scalar values during computation.


## Issues Encountered & Fixes

### 1. Dimension Mismatch Error
   Unable to perform assignment because the left and right sides have a different number of elements

✔ Cause
num or den computed as vectors instead of scalars

✔ Fix
Reduce expressions using sum() or proper indexing
Validate dimensions using size()


### 2. Numerical Instability

✔ Cause
Division by very small values

✔ Fix
Add epsilon:
den + ε


### 3. Indexing Errors

✔ Cause
Incorrect loop indexing leading to mismatched assignments

✔ Fix
Ensure consistent indexing across all variables


## To Run
- Open MATLAB
- Set working directory to project folder
- Run:
  "file_1"
  "file_2"

## Output
- Computed numerical results
- Updated parameter values across iterations
- Validated stable outputs without runtime errors

## Technologies Used
- MATLAB
- Linear Algebra
- Numerical Methods
- Iterative Algorithms

## Key Takeaways
- Numerical problems require strict dimensional consistency
- Small implementation errors can cause major runtime failures
- Iterative methods require stability safeguards
- Debugging is a critical part of numerical computing

## Future Improvements
- Convert scripts into modular functions
- Add convergence criteria for iterations
- Visualize parameter evolution
- Optimize using vectorized operations

*ⓘ This project is intended for academic and learning purposes.*
