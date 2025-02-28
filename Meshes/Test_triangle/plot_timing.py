import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_timing_data():
  
    file_path = os.path.join(os.path.dirname(__file__), 'timing_triangle.csv')

    df = pd.read_csv(file_path)
    
    num_points = df['NumPoints']
    times = df['TimeElapsed(ms)']  
    
    plt.figure(figsize=(8, 6))
    plt.loglog(num_points, times, label='Computational Time', marker='o')

    n = np.array(num_points)
    times = np.array(times)

    n_min = n[0]
    time_min = times[0]

    scaling_factor_nlogn = time_min / (n_min * np.log(n_min))
    scaling_factor_n = time_min / n_min
    scaling_factor_n2 = time_min / (n_min ** 2)
    scaling_factor_n3 = time_min / (n_min ** 2*np.log(n_min))

    plt.figure(figsize=(8, 6))
    plt.loglog(n, scaling_factor_nlogn * n * np.log(n), label=r'$\mathcal{O}(n \log n)$', linestyle='dashed')
    plt.loglog(n, scaling_factor_n * n, label=r'$\mathcal{O}(n)$', linestyle='dashed')
    plt.loglog(n, scaling_factor_n2 * n**2, label=r'$\mathcal{O}(n^2)$', linestyle='dashed')
    plt.loglog(n, scaling_factor_n3 * n**2*np.log(n), label=r'$\mathcal{O}(n^2 \log n)$', linestyle='dashed')
    plt.loglog(n, times, 'b-', label='Times')

    plt.title('Computational Time vs Number of Points')
    plt.xlabel('Number of Points')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig("Meshes/Test_triangle/timing_triangle.png", dpi=300)

    print("Plot saved as Meshes/Test_triangle/timing_triangle.png")


if __name__ == "__main__":
    plot_timing_data()

