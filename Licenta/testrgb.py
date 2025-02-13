import torch
import matplotlib.pyplot as plt

# Simulăm un tensor cu 5 imagini RGB de 32x32
images = torch.randint(0, 256, (5, 32, 32, 3), dtype=torch.uint8)

# Afișăm cele 5 imagini
fig, axes = plt.subplots(1, 5, figsize=(15, 5))  # 1 rând, 5 coloane
for i, ax in enumerate(axes):
    print(images[i].numpy())
    ax.imshow(images[i].numpy())  # Convertim din tensor în array
    ax.axis('off')  # Ascundem axele

plt.show()