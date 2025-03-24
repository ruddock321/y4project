# Given parameters
eta = 0.123  # Efficiency factor
M_star_solar = 0.838  # Mass in solar units
T_eff = 4788  # Effective temperature in Kelvin
L_star_solar = 37.87  # Luminosity in solar units
nu_max = 38.72  # in microHz
delta_nu = 38.72  # in microHz

# Solar reference values
nu_max_sun = 3090  # microHz
delta_nu_sun = 135  # microHz
T_eff_sun = 5777  # K
g_sun = 2.74e4  # cm/s^2

# Compute Radius using scaling relation
R_star_solar = (nu_max / nu_max_sun) * (delta_nu / delta_nu_sun) ** (-2) * (T_eff / T_eff_sun) ** 0.5

# Compute Surface gravity
g_star = g_sun * (M_star_solar / R_star_solar**2)

# Compute Mass loss rate using the given equation
Mdot = (eta * (L_star_solar * R_star_solar / M_star_solar) *
        (T_eff / 4000) ** 3.5 * (1 + (g_sun / (4300 * g_star))))

# Print results
print(f"Radius (R*): {R_star_solar:.4f} R_sun")
print(f"Surface Gravity (g*): {g_star:.4f} cm/s^2")
print(f"Mass Loss Rate (Mdot): {Mdot:.4e} M_sun/year")

