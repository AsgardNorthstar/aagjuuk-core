import deepxde as dde
import numpy as np
import tensorflow as tf

# 1. Define 3D Spatial and Temporal Domains
geom = dde.geometry.Cuboid([0, 0, 0], [1, 1, 1])
timedomain = dde.geometry.TimeDomain(0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. Define the Coupled Thermo-Mechanical PDE System
def multi_physics_pde(x, y):
    # Inputs: x[:, 0] = x, x[:, 1] = y, x[:, 2] = z, x[:, 3] = t
    # Outputs: y[:, 0] = T (Temp), y[:, 1] = u, y[:, 2] = v, y[:, 3] = w (Displacements)
    
    T, u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    
    # Extract coordinates for gradient operations
    X, Y, Z, t = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
    
    # --- 1. Thermal Dissipation Component ---
    dT_t = dde.grad.jacobian(y, x, i=0, j=3)
    dT_xx = dde.grad.hessian(y, x, i=0, j=0)
    dT_yy = dde.grad.hessian(y, x, i=0, j=1)
    dT_zz = dde.grad.hessian(y, x, i=0, j=2)
    alpha = 0.05
    thermal_residual = dT_t - alpha * (dT_xx + dT_yy + dT_zz)
    
    # --- 2. Structural Mechanical Stress Components (Navier-Cauchy) ---
    # Material Constants for Silicon Substrate / Metals
    lmbda = 1.0  # Lame constant
    mu = 1.0     # Shear modulus
    gamma = 0.02 # Thermoelastic coupling coefficient (CTE expansion impact)
    
    # First derivatives of displacements
    du_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=2, j=1)
    dw_z = dde.grad.jacobian(y, x, i=3, j=2)
    
    # Volumetric strain (dilation tensor)
    dilation = du_x + dv_y + dw_z
    
    # Spatial gradients of thermal expansion forces
    dT_x = dde.grad.jacobian(y, x, i=0, j=0)
    dT_y = dde.grad.jacobian(y, x, i=0, j=1)
    dT_z = dde.grad.jacobian(y, x, i=0, j=2)
    
    # Second derivatives for Navier-Cauchy equations
    du_xx = dde.grad.hessian(y, x, i=1, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)
    du_zz = dde.grad.hessian(y, x, i=1, j=2)
    
    dv_xx = dde.grad.hessian(y, x, i=2, j=0)
    dv_yy = dde.grad.hessian(y, x, i=2, j=1)
    dv_zz = dde.grad.hessian(y, x, i=2, j=2)
    
    dw_xx = dde.grad.hessian(y, x, i=3, j=0)
    dw_yy = dde.grad.hessian(y, x, i=3, j=1)
    dw_zz = dde.grad.hessian(y, x, i=3, j=2)
    
    # Navier-Cauchy Residual Forces combining mechanical stress and thermal loading
    stress_x_residual = (lmbda + mu) * dde.grad.jacobian(dilation, x, i=0, j=0) + mu * (du_xx + du_yy + du_zz) - gamma * dT_x
    stress_y_residual = (lmbda + mu) * dde.grad.jacobian(dilation, x, i=0, j=1) + mu * (dv_xx + dv_yy + dv_zz) - gamma * dT_y
    stress_z_residual = (lmbda + mu) * dde.grad.jacobian(dilation, x, i=0, j=2) + mu * (dw_xx + dw_yy + dw_zz) - gamma * dT_z
    
    return [thermal_residual, stress_x_residual, stress_y_residual, stress_z_residual]

# 3. Boundary Conditions (Fixed boundaries, zero displacement and zero temp at borders)
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

# Initial thermal localized spike at t=0
def init_thermal(x):
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2]) * np.sin(np.pi * x[:, 2:3])

ic_thermal = dde.icbc.IC(geomtime, init_thermal, lambda _, on_initial: on_initial, component=0)

# Combine datasets and equations
data = dde.data.TimePDE(
    geomtime, 
    multi_physics_pde, 
    [bc, ic_thermal], 
    num_domain=6000, 
    num_boundary=400, 
    num_initial=400
)

# 4. Neural Network Scaling
# Input: [x, y, z, t] -> Output: [T, u, v, w]
net = dde.nn.FNN([4] + [64] * 5 + [4], "tanh", "Glorot normal")

# 5. Multitask Model Compilation
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

# 6. Train the Multi-Physics Brain
losshistory, train_state = model.train(iterations=5000)
model.save("asgard_multiphysics_3d_model")
print("3D Multi-Physics Solver successfully completed.")
