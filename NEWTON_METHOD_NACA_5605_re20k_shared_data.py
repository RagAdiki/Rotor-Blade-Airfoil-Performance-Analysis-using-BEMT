import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import newton

# -----------------------------
# Inputs
# -----------------------------
sigma = 0.1035

alpha_polar = np.linspace(-2, 16, 10)  # deg
cl_values = np.array([
0.19462399,
0.39948897,
0.593872771,
0.777963477,
0.948276409,
1.08364461,
1.04952965,
0.894670539,
1.230259346,
1.017724079,

])


cd_values = np.array([
0.030663059,
0.030678833,
0.033158328,
0.037493554,
0.04417412,
0.06391161,
0.117564544,
0.193019895,
0.18963496,
0.283863224,

])




# Interpolators
cl_interp = interp1d(alpha_polar, cl_values, kind='linear', fill_value='extrapolate')
cd_interp = interp1d(alpha_polar, cd_values, kind='linear', fill_value='extrapolate')

# -----------------------------
# Solve for lambda
# -----------------------------
def solve_lambda(theta_deg, r):
    theta_rad = np.deg2rad(theta_deg)

    def f(lam):
        alpha_deg = np.rad2deg(theta_rad - lam/r + 0.052)
        CL = float(cl_interp(alpha_deg))
        return 4*lam**2 - 0.5*sigma*CL*r

    try:
        lam = newton(f, x0=0.05, tol=1e-10, maxiter=100)
    except RuntimeError:
        lam = 1e-6
    if lam <= 0:
        CL = float(cl_interp(theta_deg))
        lam = (0.5*sigma*CL*r)**0.5
    return lam

# -----------------------------
# Compute CT and CP
# -----------------------------
def compute_CT_CP(theta_deg, r_vals):
    lam_r = np.array([solve_lambda(theta_deg, r) for r in r_vals])
    phi = lam_r / r_vals
    alpha_eff = np.rad2deg(np.deg2rad(theta_deg) - phi + 0.052 )

    CL = cl_interp(alpha_eff)
    CD = cd_interp(alpha_eff)

    integrand_ct = (CL*np.cos(phi) - CD*np.sin(phi)) * r_vals**2
    integrand_cp = (CL*np.sin(phi) + CD*np.cos(phi)) * r_vals**3

    CT = 0.5 * sigma * np.trapezoid(integrand_ct, r_vals)
    CP = 0.5 * sigma * np.trapezoid(integrand_cp, r_vals)

    # corrected CP (user formula)
    CP_corr = CP 

    return CT, CP, CP_corr

# -----------------------------
# Sweep theta
# -----------------------------
r_vals = np.linspace(0.05, 1.0, 200)
thetas = np.arange(-2, 16, 2)

CT_list = []
CP_list = []
CP_corr_list = []

for th in thetas:
    CT, CP, CP_corr = compute_CT_CP(th, r_vals)
    CT_list.append(CT)
    CP_list.append(CP)
    CP_corr_list.append(CP_corr)

CT_list = np.array(CT_list)
CP_list = np.array(CP_list)
CP_corr_list = np.array(CP_corr_list)

# -----------------------------
# Plotting
# -----------------------------
# -----------------------------
plt.figure()
plt.plot(thetas, CT_list, '-o')
plt.xlabel("Theta (deg)")
plt.ylabel("CT")
plt.title("CT vs Theta")
plt.grid(True)

plt.figure()
plt.plot(thetas, CP_list, '-s', label='CP')
plt.plot(thetas, CP_corr_list, '-^', label='CP corrected')
plt.xlabel("Theta (deg)")
plt.ylabel("CP")
plt.title("CP and Corrected CP vs Theta")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(thetas, CT_list/CP_list, '-o')
plt.xlabel("Theta (deg)")
plt.ylabel("CT/CP")
plt.title("CT/CP vs Theta")
plt.grid(True)

plt.figure()
plt.plot(thetas, CT_list/CP_corr_list, '-s')
plt.xlabel("Theta (deg)")
plt.ylabel("CT/CP corrected")
plt.title("CT / Corrected CP vs Theta")
plt.grid(True)



plt.show()

# Plot CL vs Alpha
plt.figure(figsize=(10,5))
plt.plot(alpha_polar, cl_values, 'b-', linewidth=2, label='$C_L$ vs $\\alpha$')
plt.xlabel('Angle of Attack, α (deg)')
plt.ylabel('Lift Coefficient, $C_L$')
plt.title('Lift Coefficient vs Angle of Attack')
plt.grid(True)
plt.legend()
plt.show()

# Plot CD vs Alpha
plt.figure(figsize=(10,5))
plt.plot(alpha_polar, cd_values, 'r-', linewidth=2, label='$C_D$ vs $\\alpha$')
plt.xlabel('Angle of Attack, α (deg)')
plt.ylabel('Drag Coefficient, $C_D$')
plt.title('Drag Coefficient vs Angle of Attack')
plt.grid(True)
plt.legend()
plt.show()