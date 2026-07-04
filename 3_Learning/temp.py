import torch

file_path = './graphs_multi_a11y/www.tamu.edu_a11y-tree.pt'

# Explicitly set weights_only=False to load all custom graph attributes
data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)

print(data)

