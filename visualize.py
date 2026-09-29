import matplotlib.pyplot as plt
from data import sample_data

z = sample_data(1000)

print(z.shape)  

points = z.detach().cpu().numpy()

plt.scatter(points[:, 0], points[:, 1], s=8, alpha=0.4)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")  
plt.grid(alpha=0.3)
plt.show()