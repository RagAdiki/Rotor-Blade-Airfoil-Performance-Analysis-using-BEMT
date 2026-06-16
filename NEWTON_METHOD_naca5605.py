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
    0.3526,0.3730,0.3930,0.4126,0.4320,0.4511,0.4699,0.4885,0.5069,0.5250,
    0.5429,0.5606,0.5780,0.5951,0.6120,0.6286,0.6449,0.6609,0.6766,0.6919,
    0.7068,0.7213,0.7353,0.7488,0.7618,0.7742,0.7860,0.7971,0.8078,1.1464,
    1.1756,1.2019,1.2182,1.2365,1.2492,1.2622,1.2715,1.2752,1.2811,1.2788,
    1.2687,1.2400,1.2175,1.1925,1.1679,1.1377,1.1051,0.9900,0.9667,0.9396,
    1.0437,1.0394,1.0401,1.0610,1.0630,1.0651,1.0673,1.0685,1.0695,1.0694,
    1.0706,1.0733,1.0794,1.0896,1.0996,1.0958,1.0940,1.1008,1.1105,1.1214,
    1.1116,1.1215,1.1388,1.1274,1.1408,1.1381,1.1412,1.1523,1.1496,1.1701,
    1.1632,
])

cd_values = np.array([
    0.02525,0.02562,0.02601,0.02642,0.02684,0.02728,0.02774,0.02822,0.02872,0.02924,
    0.02979,0.03036,0.03095,0.03158,0.03224,0.03293,0.03366,0.03443,0.03525,0.03612,
    0.03704,0.03803,0.03909,0.04022,0.04144,0.04275,0.04417,0.04569,0.04733,0.04392,
    0.04728,0.05126,0.05349,0.05747,0.06065,0.06449,0.06858,0.07291,0.07784,0.08320,
    0.08886,0.09266,0.09769,0.10246,0.10895,0.11480,0.12505,0.10989,0.11679,0.12360,
    0.14907,0.15208,0.15559,0.16125,0.16448,0.16770,0.17086,0.17384,0.17679,0.17958,
    0.18252,0.18568,0.18926,0.19349,0.19766,0.19915,0.20142,0.20505,0.20920,0.21363,
    0.21354,0.21792,0.22426,0.22317,0.22835,0.22962,0.23178,0.23720,0.23742,0.24629,
    0.24465
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

plt.figure()
plt.plot(thetas, FM_list , '-s')
plt.xlabel("Theta (deg)")
plt.ylabel("FM")
plt.title("FM vs Theta")
plt.grid(True)
plt.figure()
plt.plot(CT_list, FM_list , '-s')
plt.xlabel("ct")
plt.ylabel("FM")
plt.title("FM vs ct")
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