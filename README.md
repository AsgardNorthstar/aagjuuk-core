# Aagjuuk: Continuous 3D Thermo-Mechanical Edge Solver for Anisotropic Silicon Interfaces

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aagjuuk.streamlit.app/)

Aagjuuk is a coordinate-based, real-time (sub-5ms) numerical solver designed to compute continuous 3D coupled thermo-mechanical stress fields in microelectronic substrates. 

By replacing traditional grid-discretization methods (mesh-bound Finite Element Analysis) with multi-scale coordinate projections, the engine evaluates stress, thermal, and displacement fields over continuous domains. This architecture enables online, closed-loop actuator calibration and non-destructive defect localization directly on edge compute nodes.

---

## 1. Mathematical Formulation

Aagjuuk models the direct physical coupling between the **3D Fourier Heat Conduction Equation** and the **Anisotropic Navier-Cauchy Equations of Elasticity** for cubic crystalline lattices (e.g., single-crystal silicon).

### 1.1 Thermal Domain
The temperature field $T(\mathbf{x}, t)$ across the continuous domain is governed by the transient heat equation:

$$\rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (\kappa \nabla T) + Q$$

where:
* $\rho$ represents material density ($\text{kg/m}^3$).
* $C_p$ is the specific heat capacity ($\text{J/kg}\cdot\text{K}$).
* $\kappa$ represents the second-rank symmetric thermal conductivity tensor ($\text{W/m}\cdot\text{K}$).
* $Q$ is the volumetric heat source term representing localized laser absorption ($\text{W/m}^3$).

### 1.2 Structural Mechanics Domain
The resulting displacement field $\mathbf{u} = [u, v, w]^T$ is resolved using the Navier-Cauchy equations of motion under static equilibrium conditions:

$$\frac{\partial \sigma_{ij}}{\partial x_j} + F_i = 0$$

The second-rank stress tensor $\sigma_{ij}$ is coupled to the thermal strain field through the generalized Duhamel-Neumann relation:

$$\sigma_{ij} = C_{ijkl} \left( \varepsilon_{kl} - \alpha_{kl} \Delta T \right)$$

where:
* $C_{ijkl}$ is the fourth-rank anisotropic stiffness tensor (configured with cubic symmetry parameters $C_{11}$, $C_{12}$, and $C_{44}$).
* $\varepsilon_{kl} = \frac{1}{2} \left( \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right)$ is the infinitesimal strain tensor.
* $\alpha_{kl}$ represents the thermal expansion tensor ($\text{K}^{-1}$).
* $\Delta T = T - T_{\text{reference}}$ is the temperature offset.

---

## 2. Core Architectural Components

The repository is partitioned into isolated, modular subsystems optimized for low-latency execution:
### 2.1 Multi-Scale Fourier Feature Projection (`aagjuuk_fourier_encoder.py`)
Standard neural approximations suffer from spectral bias, rendering them incapable of resolving high-frequency spatial gradients at material interfaces. Aagjuuk maps the coordinate tensor $\mathbf{x} \in \mathbb{R}^3$ into a high-dimensional sinusoidal manifold prior to evaluation:

$$\gamma(\mathbf{x}) = \left[ \cos(2\pi \mathbf{B}\mathbf{x}), \sin(2\pi \mathbf{B}\mathbf{x}) \right]^T$$

where the projection matrix $\mathbf{B} \sim \mathcal{N}(0, \sigma^2)$ is scaled to match the physical frequencies of the material boundary conditions.

### 2.2 Inverse Defect Localization (`aagjuuk_inverse_solver.py`)
Resolves the inverse boundary value problem. By monitoring thermal telemetry anomalies at discrete surface sensor nodes, the inverse solver computes localized physical gradients to map sub-surface material discontinuities without destructive cross-sectioning.

### 2.3 Closed-Loop Actuator Regulation (`aagjuuk_control_loop.py`)
Integrates a continuous proportional-integral-derivative (PID) loop designed to dynamically regulate laser power input based on computed peak structural stress. Includes integral back-calculation (anti-windup) to prevent actuator saturation during high-amplitude transient thermal states.

---

## 3. Operational Performance Benchmarks

Inference profiling is conducted locally using hardware system telemetry bindings.

| Computational Metric | Benchmark Performance | Hardware Target |
| :--- | :--- | :--- |
| **Inference Latency** | $\le 3.84\text{ ms}$ (Average) | Intel Core i7 / AMD Ryzen 5 (Single Core) |
| **Dynamic RAM Footprint** | $\approx 42.10\text{ MB}$ (Stable) | Embedded System Memory Limits |
| **Edge Recalibration Epoch** | $\approx 120\text{ ms}$ (15 iterations) | Browser Runtime / Edge Node WebAssembly |

---

## 4. Setup and Local Execution

### 4.1 Prerequisites
The package requires a Python environment ($\ge 3.9$). 

### 4.2 Installation
To clone and install dependencies in an isolated virtual environment:

### 4.3 Running the Interactive Diagnostic Console
Execute the visualization and hardware telemetry interface:

### 4.4 Programmatic API Integration
To query the numerical solver within an active data-acquisition pipeline:

## 5. Licensing and Terms of Usage

This project is licensed under the terms of the **MIT License**. For complete terms, consult the `LICENSE` file in the root directory.
