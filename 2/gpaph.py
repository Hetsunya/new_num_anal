import numpy as np
import matplotlib.pyplot as plt

# --- Система и функционал ---
def F(x):
    x1, x2 = x
    f1 = np.sinh(x1 + 0.2*x2 + np.tan(0.1*x1*x2)) - 0.8
    f2 = np.sinh(0.6*x1 - 0.1*x2 + np.tan(0.2*x1*x2)) - 0.1
    return np.array([f1, f2])

def Phi(x):
    f = F(x)
    return 0.5 * np.dot(f, f)

def grad_Phi(x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = h
        g[i] = (Phi(x+dx) - Phi(x-dx))/(2*h)
    return g

# --- Градиентный спуск с сохранением траектории ---
def gradient_descent_path(x0, alpha=0.1, eps=1e-6, max_iter=100000):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_iter):
        grad = grad_Phi(x)
        x_new = x - alpha * grad
        path.append(x_new.copy())
        if np.linalg.norm(x_new - x) < eps:
            break
        x = x_new
    return np.array(path), x

# --- Запуск ---
x_start = [0.0, 0.0]
path, solution = gradient_descent_path(x_start, alpha=0.1, eps=1e-6)

# --- Сетка для графиков ---
x1_vals = np.linspace(solution[0]-0.5, solution[0]+0.5, 100)
x2_vals = np.linspace(solution[1]-0.5, solution[1]+0.5, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.array([[Phi([x1,x2]) for x1 in x1_vals] for x2 in x2_vals])

# --- 3D-график поверхности ---
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('Phi(x1,x2)')
ax1.set_title('3D поверхность Phi(x1,x2)')

# --- Контурный график с траекторией ---
ax2 = fig.add_subplot(1,2,2)
CS = ax2.contour(X1, X2, Z, levels=30, cmap='viridis')
ax2.clabel(CS, inline=1, fontsize=8)
ax2.plot(path[:,0], path[:,1], 'ro-', markersize=3, label='Траектория градиентного спуска')
ax2.plot(solution[0], solution[1], 'b*', markersize=12, label='Минимум')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Контур Phi(x1,x2) с траекторией')
ax2.legend()

plt.tight_layout()
plt.show()
