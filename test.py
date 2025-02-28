import numpy as np

# Example arrays
p_1 = np.array([[1, 2], [3, 4]])  # Shape (2, 2)
p_3 = np.array([[5, 6], [7, 8]])  # Shape (2, 2)
np.set_printoptions(precision=3, suppress=True)
# Adding new axes
p_1_expanded = p_1[:, np.newaxis, :]  # Shape (2, 2, 1)
p_3_expanded = p_3[ np.newaxis, :, :]  # Shape (2, 1, 2)
print(p_1_expanded)
print(p_3_expanded)

# Broadcasting and element-wise multiplication
product = p_1_expanded * p_3_expanded  # Shape (2, 2, 2)
print(product)
# Calculating the mean along the third axis
mean_product = np.mean(product, axis=2)  # Shape (2, 2)
print(mean_product)

vector1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
vector2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(np.mean(vector1, axis=1))