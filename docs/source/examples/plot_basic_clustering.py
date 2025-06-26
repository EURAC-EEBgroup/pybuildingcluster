"""
Basic Building Clustering
=========================

This example demonstrates basic clustering of buildings using PyBuildingCluster.
"""

import numpy as np
import matplotlib.pyplot as plt

print("PyBuildingCluster basic example")

# Generate sample data
np.random.seed(42)
x = np.random.normal(1000, 300, 100)
y = np.random.normal(50, 15, 100)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6)
plt.xlabel('Floor Area (m²)')
plt.ylabel('Energy Consumption (kWh/m²/year)')
plt.title('Building Data Example')
plt.show()