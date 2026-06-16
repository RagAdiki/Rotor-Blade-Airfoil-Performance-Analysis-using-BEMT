import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import newton

# -----------------------------
# Inputs
# -----------------------------
sigma = 0.1035

alpha_polar = np.linspace(0, 16, 81)  # deg
cl_values = np.array([
0.2734,0.3067,0.3162,0.3335,0.3492,0.3642,0.3789,0.3932,0.4072,0.4213,
0.4574,0.4910,0.5239,0.5577,0.5833,0.6060,0.6303,0.6590,0.6815,0.6987,
0.7211,0.7485,0.7593,0.7782,0.8075,0.8140,0.8300,0.8559,0.8653,0.8772,
0.8969,0.9225,0.9279,0.9385,0.9532,0.9694,0.9905,1.1291,1.1421,1.1813,
1.2388,1.2712,1.2877,1.3035,1.3174,1.3285,1.3395,1.3423,1.3293,1.3185,
1.3109,1.3204,1.2932,1.2663,1.2396,1.2127,1.1853,1.1578,1.1333,1.1283,
1.0984,1.0961,1.0718,1.3070,1.0265,1.0277,1.0337,1.0411,1.0329,1.0520,
1.0419,1.0568,1.0531,1.0671,1.0623,1.0827,1.0721,1.0908,1.0832,1.0957,
1.0957
])

cd_values = np.array([
0.02981,0.02831,0.02673,0.02749,0.02830,0.02915,0.03005,0.03099,0.03197,0.03299,
0.03426,0.03550,0.03672,0.03792,0.03902,0.04013,0.04127,0.04243,0.04356,0.04473,
0.04593,0.04711,0.04839,0.04968,0.05089,0.05232,0.05373,0.05502,0.05656,0.05812,
0.05958,0.06091,0.06267,0.06437,0.06596,0.06748,0.06870,0.03900,0.04128,0.04376,
0.04717,0.05081,0.05359,0.05682,0.06046,0.06431,0.06861,0.07204,0.07515,0.07892,
0.08297,0.08821,0.09088,0.09396,0.09766,0.10209,0.10740,0.11383,0.12131,0.12740,
0.13824,0.14484,0.15623,0.17779,0.18098,0.18464,0.18883,0.19343,0.19436,0.20056,
0.20077,0.20608,0.20713,0.21264,0.21325,0.22046,0.21938,0.22594,0.22563,0.23026,
0.23212
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
        alpha_deg = np.rad2deg(theta_rad - lam/r + 0.070)
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
    alpha_eff = np.rad2deg(np.deg2rad(theta_deg) - phi + 0.070 )

    CL = cl_interp(alpha_eff)
    CD = cd_interp(alpha_eff)

    integrand_ct = (CL*np.cos(phi) - CD*np.sin(phi)) * r_vals**2
    integrand_cp = (CL*np.sin(phi) + CD*np.cos(phi)) * r_vals**3

    CT = 0.5 * sigma * np.trapezoid(integrand_ct, r_vals)
    CP = 0.5 * sigma * np.trapezoid(integrand_cp, r_vals)

    # corrected CP (user formula)
    CP_corr = CP
    FM = ((CT)**1.5)/(CP*1.414)
    
    return CT, CP, CP_corr , FM

# -----------------------------
# Sweep theta
# -----------------------------
r_vals = np.linspace(0.05, 1.0, 200)
thetas = np.arange(0, 16.1, 0.5)

CT_list = []
CP_list = []
CP_corr_list = []
FM_list = []

for th in thetas:
    CT, CP, CP_corr ,FM = compute_CT_CP(th, r_vals)
    CT_list.append(CT)
    CP_list.append(CP)
    CP_corr_list.append(CP_corr)
    FM_list.append(FM)

CT_list = np.array(CT_list)
CP_list = np.array(CP_list)
CP_corr_list = np.array(CP_corr_list)
FM_list = np.array(FM_list)
# -----------------------------
# Plotting
# -----------------------------
plt.figure()
plt.plot(thetas, CT_list, '-o')
plt.xlabel("Theta (deg)")
plt.ylabel("CT")
plt.title("CT vs Theta")
plt.grid(True)

# -----------------------------
plt.figure()
plt.plot(thetas, CP_list, '-o')
plt.xlabel("Theta (deg)")
plt.ylabel("CP")
plt.title("CP vs Theta")
plt.grid(True)


plt.figure()
plt.plot(thetas, CT_list/CP_list, '-o')
plt.xlabel("Theta (deg)")
plt.ylabel("CT/CP")
plt.title("CT/CP vs Theta")
plt.grid(True)
"""
plt.figure()
plt.plot(CT_list, CP/CP_list, '-o')
plt.xlabel("ct")
plt.ylabel("FM")
plt.title("ct ")
plt.grid(True)"""

plt.figure()
plt.plot(thetas, FM_list , '-s')
plt.xlabel("Theta (deg)")
plt.ylabel("FM")
plt.title("FM vs Theta")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(CT_list, FM_list , '-s')
plt.xlabel("ct")
plt.ylabel("FM")
plt.title("FM vs ct")
plt.grid(True)


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