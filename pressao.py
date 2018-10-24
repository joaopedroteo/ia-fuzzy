import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * distanciaity and velocidadeice on subjective ranges [0, 10]
#   * pressao has a range of [0, 25] in units of percentage points
x_distancia = np.arange(0, 101, 1)
# x_distancia = x_distancia[::-1]
# print(x_distancia)
x_velocidade = np.arange(0, 101, 1)
x_pressao  = np.arange(0, 101, 1)


# Generate fuzzy membership functions
distancia_lo = fuzz.trimf(x_distancia, [0, 0, 40])
distancia_md = fuzz.trimf(x_distancia, [25, 50, 75])
distancia_hi = fuzz.trimf(x_distancia, [60, 100, 100])
velocidade_lo = fuzz.trimf(x_velocidade, [0, 0, 40])
velocidade_md = fuzz.trimf(x_velocidade, [25, 50, 75])
velocidade_hi = fuzz.trimf(x_velocidade, [60, 100, 100])
pressao_lo = fuzz.trimf(x_pressao, [0, 0, 40])
pressao_md = fuzz.trimf(x_pressao, [25, 50, 75])
pressao_hi = fuzz.trimf(x_pressao, [60, 100, 100])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_distancia, distancia_lo, 'b', linewidth=1.5, label='Longe')
ax0.plot(x_distancia, distancia_md, 'g', linewidth=1.5, label='Próximo')
ax0.plot(x_distancia, distancia_hi, 'r', linewidth=1.5, label='Muito próximo')
ax0.set_title('Distancia')
ax0.legend()
ax0.invert_xaxis()

ax1.plot(x_velocidade, velocidade_lo, 'b', linewidth=1.5, label='Devagar')
ax1.plot(x_velocidade, velocidade_md, 'g', linewidth=1.5, label='Normal')
ax1.plot(x_velocidade, velocidade_hi, 'r', linewidth=1.5, label='Rápido')
ax1.set_title('Velocidade')
ax1.legend()

ax2.plot(x_pressao, pressao_lo, 'b', linewidth=1.5, label='Baixa')
ax2.plot(x_pressao, pressao_md, 'g', linewidth=1.5, label='Média')
ax2.plot(x_pressao, pressao_hi, 'r', linewidth=1.5, label='Alta')
ax2.set_title('Pressao no freio')
ax2.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

plt.show()


# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!
distancia = input("Distancia:")
distancia_level_lo = fuzz.interp_membership(x_distancia, distancia_lo, distancia)
distancia_level_md = fuzz.interp_membership(x_distancia, distancia_md, distancia)
distancia_level_hi = fuzz.interp_membership(x_distancia, distancia_hi, distancia)

velocidade = input("Velocidade:")
velocidade_level_lo = fuzz.interp_membership(x_velocidade, velocidade_lo, velocidade)
velocidade_level_md = fuzz.interp_membership(x_velocidade, velocidade_md, velocidade)
velocidade_level_hi = fuzz.interp_membership(x_velocidade, velocidade_hi, velocidade)

# Now we take our rules and apply them. Rule 1 concerns bad food OR service.
# The OR operator means we take the maximum of these two.
active_rule1 = np.fmax(distancia_level_hi, velocidade_level_lo)

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
pressao_activation_lo = np.fmin(active_rule1, pressao_lo)  # removed entirely to 0

# For rule 2 we connect acceptable service to medium tiping
pressao_activation_md = np.fmin(velocidade_level_md, pressao_md)

# For rule 3 we connect high service OR high food with high pressaoping
active_rule3 = np.fmax(distancia_level_lo, velocidade_level_hi)
pressao_activation_hi = np.fmin(active_rule3, pressao_hi)
pressao0 = np.zeros_like(x_pressao)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_pressao, pressao0, pressao_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_pressao, pressao_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_pressao, pressao0, pressao_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_pressao, pressao_md, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_pressao, pressao0, pressao_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_pressao, pressao_hi, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# plt.show()

# Aggregate all three output membership functions together
aggregated = np.fmax(pressao_activation_lo,
                     np.fmax(pressao_activation_md, pressao_activation_hi))

# Calculate defuzzified result
pressao = fuzz.defuzz(x_pressao, aggregated, 'centroid')
pressao_activation = fuzz.interp_membership(x_pressao, aggregated, pressao)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_pressao, pressao_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_pressao, pressao_md, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_pressao, pressao_hi, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_pressao, pressao0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([pressao, pressao], [0, pressao_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

print(pressao)

plt.show()