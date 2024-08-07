# Bruce A. Maxwell
# <Min Ren, Jinda Zhang, Qingyi Tian, Lan Wang>
# Summer 2024
# CS 5800 Algorithms
# Final Project - K-Nearest Neighbors for continuous variables using a K-D tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


# Create a class for the nodes in the tree
class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point = point  # The point stored in this node
        self.left = left    # Left child node
        self.right = right  # Right child node


# Function to build the KD-Tree from the dataset
def build_kd_tree(points, depth=0):
    if not points:
        return None

    k = len(points[0])  # Dimension of points (assume all points have the same dimension)
    axis = depth % k    # Select axis based on depth so that axis cycles through all valid values

    # Sort point list and choose median as pivot element
    points.sort(key=lambda x: x[axis])
    median = len(points) // 2  # Find the median

    # Create node and construct subtrees
    return KDNode(
        point=points[median],
        left=build_kd_tree(points[:median], depth + 1),
        right=build_kd_tree(points[median + 1:], depth + 1)
    )


# Function to search for the k-nearest neighbors in the KD-Tree
def kd_tree_search(node, point, k, depth=0):
    if node is None:
        return []

    k = len(node.point)
    axis = depth % k  # Select axis based on depth

    next_branch = None  # Next branch to explore
    opposite_branch = None  # Opposite branch to explore

    if point[axis] < node.point[axis]:
        next_branch = node.left
        opposite_branch = node.right
    else:
        next_branch = node.right
        opposite_branch = node.left

    best = kd_tree_search(next_branch, point, k, depth + 1)  # Explore the next branch
    best.append(node.point)
    best.sort(key=lambda x: distance(x, point))  # Sort by distance to target point

    if len(best) > k:
        best = best[:k]  # Keep only the k closest points

    if len(best) < k or abs(point[axis] - node.point[axis]) < distance(best[-1], point):
        best.extend(kd_tree_search(opposite_branch, point, k, depth + 1))
        best.sort(key=lambda x: distance(x, point))
        if len(best) > k:
            best = best[:k]  # Keep only the k closest points

    return best


# Function to calculate Euclidean distance between two points
def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Function to plot the KNN search process using animation
def plot_knn(points, target, neighbors):
    fig, ax = plt.subplots()
    scat = ax.scatter(*zip(*points), label='Data Points')
    target_scat = ax.scatter(*target, color='green', label='Target')
    neighbor_scat = ax.scatter([], [], color='red', label='Neighbors')
    plt.legend()

    # Update function for animation
    def update(frame):
        if frame < len(neighbors):
            neighbor_scat.set_offsets(neighbors[:frame + 1])
        return neighbor_scat,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(neighbors), blit=True, repeat=False)
    plt.show()


# Main function to execute the KD-Tree creation, KNN search, and plotting
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))  # Get the parent directory
    data_path = os.path.join(parent_dir, 'data', 'data.csv')  # Construct the path to the CSV file

    try:
        df = pd.read_csv(data_path)
        print("File read successfully.")
    except FileNotFoundError:
        print("File not found. Please check the path.")

    points = df.values.tolist()  # Convert data to list of points
    kd_tree = build_kd_tree(points)  # Build KD-Tree from points

    target = [5, 5, 5]  # Define a target point for KNN search
    neighbors = kd_tree_search(kd_tree, target, k=3)  # Perform KNN search
    plot_knn(points, target, neighbors)  # Plot the KNN search process


if __name__ == "__main__":
    main()
