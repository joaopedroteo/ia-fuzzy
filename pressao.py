import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


x_distancia = np.arange(0, 101, 1)      # distancia vai de 0 a 100 variando em 1
x_velocidade = np.arange(0, 101, 1)     # velocidade vai de 0 a 100 variando em 1
x_pressao  = np.arange(0, 101, 1)       # pressao vai de 0 a 100 variando em 1


# atribuindo valores às funções fuzzy
distancia_vlo = fuzz.gaussmf(x_distancia, 0, 9)
distancia_lo = fuzz.gaussmf(x_distancia, 25, 9)
distancia_md = fuzz.gaussmf(x_distancia, 50, 9)
distancia_hi = fuzz.gaussmf(x_distancia, 75, 9)
distancia_vhi = fuzz.gaussmf(x_distancia, 100, 9)
velocidade_vlo = fuzz.gaussmf(x_velocidade,  0, 9)
velocidade_lo = fuzz.gaussmf(x_velocidade,  25, 9)
velocidade_md = fuzz.gaussmf(x_velocidade,  50, 9)
velocidade_hi = fuzz.gaussmf(x_velocidade,  75, 9)
velocidade_vhi = fuzz.gaussmf(x_velocidade,  100, 9)
pressao_vlo = fuzz.gaussmf(x_pressao, 0, 9)
pressao_lo = fuzz.gaussmf(x_pressao, 25, 9)
pressao_md = fuzz.gaussmf(x_pressao, 50, 9)
pressao_hi = fuzz.gaussmf(x_pressao, 75, 9)
pressao_vhi = fuzz.gaussmf(x_pressao, 100, 9)

# gera os graficos
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_distancia, distancia_vlo, 'b', linewidth=1.5, label='Muito Longe')
ax0.plot(x_distancia, distancia_lo, 'b', linewidth=1.5, label='Longe')
ax0.plot(x_distancia, distancia_md, 'g', linewidth=1.5, label='+ ou -')
ax0.plot(x_distancia, distancia_hi, 'r', linewidth=1.5, label='Próximo')
ax0.plot(x_distancia, distancia_vhi, 'r', linewidth=1.5, label='Muito próximo')
ax0.set_title('Distancia')
ax0.legend()
ax0.invert_xaxis()

ax1.plot(x_velocidade, velocidade_vlo, 'b', linewidth=1.5, label='Muito Devagar')
ax1.plot(x_velocidade, velocidade_lo, 'b', linewidth=1.5, label='Devagar')
ax1.plot(x_velocidade, velocidade_md, 'g', linewidth=1.5, label='Normal')
ax1.plot(x_velocidade, velocidade_hi, 'r', linewidth=1.5, label='Rápido')
ax1.plot(x_velocidade, velocidade_vhi, 'r', linewidth=1.5, label='Muito Rápido')
ax1.set_title('Velocidade')
ax1.legend()

ax2.plot(x_pressao, pressao_vlo, 'b', linewidth=1.5, label='Muito Baixa')
ax2.plot(x_pressao, pressao_lo, 'b', linewidth=1.5, label='Baixa')
ax2.plot(x_pressao, pressao_md, 'g', linewidth=1.5, label='Média')
ax2.plot(x_pressao, pressao_hi, 'r', linewidth=1.5, label='Alta')
ax2.plot(x_pressao, pressao_vhi, 'r', linewidth=1.5, label='Muito Alta')
ax2.set_title('Pressao no freio')
ax2.legend()

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# plt.show()      # mostra os 3 primeiros graficos


# recebe os valores de distancia e velocidade e calcula o quanto ele ativa as funções
distancia = input("Distancia:")
distancia_level_vlo = fuzz.interp_membership(x_distancia, distancia_vlo, distancia)
distancia_level_lo = fuzz.interp_membership(x_distancia, distancia_lo, distancia)
distancia_level_md = fuzz.interp_membership(x_distancia, distancia_md, distancia)
distancia_level_hi = fuzz.interp_membership(x_distancia, distancia_hi, distancia)
distancia_level_vhi = fuzz.interp_membership(x_distancia, distancia_vhi, distancia)

velocidade = input("Velocidade:")
velocidade_level_vlo = fuzz.interp_membership(x_velocidade, velocidade_vlo, velocidade)
velocidade_level_lo = fuzz.interp_membership(x_velocidade, velocidade_lo, velocidade)
velocidade_level_md = fuzz.interp_membership(x_velocidade, velocidade_md, velocidade)
velocidade_level_hi = fuzz.interp_membership(x_velocidade, velocidade_hi, velocidade)
velocidade_level_vhi = fuzz.interp_membership(x_velocidade, velocidade_vhi, velocidade)


# aplica as regras.

# Regra 0: Se a distância é alta ou a velocidade é baixa, a pressão é baixa
regra0 = np.fmax(distancia_level_vhi, velocidade_level_vlo)
ativacao_pressao_vlo = np.fmin(regra0, pressao_vlo)

# Regra 1: Se a distância é alta ou a velocidade é baixa, a pressão é baixa
regra1 = np.fmax(distancia_level_hi, velocidade_level_lo)
ativacao_pressao_lo = np.fmin(regra1, pressao_lo)

# Regra 2: Se a distância é média ou a velocidade é média, pressão é média
regra2 = np.fmin(distancia_level_md, velocidade_level_md)
ativacao_pressao_md = np.fmin(regra2, pressao_md)


# Regra 3: Se a distancia é média e a velocidade é alta, pressao alta
regra3 = np.fmax(np.fmax(distancia_level_lo, velocidade_level_hi), np.fmax(distancia_level_lo, velocidade_level_lo))
ativacao_pressao_hi = np.fmin(regra3, pressao_hi)

# Regra 4: Se a distância é curta ou a velocidade é alta, pressão é muito alta
regra4 = np.fmax(np.fmax(np.fmax(np.fmax(distancia_level_vlo, velocidade_level_vhi), np.fmax(distancia_level_vlo, velocidade_level_vhi)), np.fmax(distancia_level_lo, velocidade_level_hi)), np.fmax(distancia_level_vlo, velocidade_level_md))
ativacao_pressao_vhi = np.fmin(regra4, pressao_vhi)

pressao0 = np.zeros_like(x_pressao)

# gera os gráficos de ativação
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_pressao, pressao0, ativacao_pressao_vlo, facecolor='b', alpha=0.7)
ax0.plot(x_pressao, pressao_vlo, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_pressao, pressao0, ativacao_pressao_lo, facecolor='b', alpha=0.7)
ax0.plot(x_pressao, pressao_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_pressao, pressao0, ativacao_pressao_md, facecolor='g', alpha=0.7)
ax0.plot(x_pressao, pressao_md, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_pressao, pressao0, ativacao_pressao_hi, facecolor='r', alpha=0.7)
ax0.plot(x_pressao, pressao_hi, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_pressao, pressao0, ativacao_pressao_vhi, facecolor='r', alpha=0.7)
ax0.plot(x_pressao, pressao_vhi, 'r', linewidth=0.5, linestyle='--')

ax0.set_title('Nível de ativação')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# plt.show()

# Agrega os três níveis de pressão gerados
aggregated = np.fmax(ativacao_pressao_vlo,
                    np.fmax(ativacao_pressao_lo,
                    np.fmax(ativacao_pressao_md, 
                    np.fmax(ativacao_pressao_hi,ativacao_pressao_vhi))))

# "desfuzzifica" o resultado
pressao = fuzz.defuzz(x_pressao, aggregated, 'centroid')
ativacao_pressao = fuzz.interp_membership(x_pressao, aggregated, pressao)  # for plot

# gera o gráfico do resultado
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_pressao, pressao_vlo, 'b', linewidth=0.5, linestyle='--')
ax0.plot(x_pressao, pressao_lo, 'b', linewidth=0.5, linestyle='--')
ax0.plot(x_pressao, pressao_md, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_pressao, pressao_hi, 'r', linewidth=0.5, linestyle='--')
ax0.plot(x_pressao, pressao_vhi, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_pressao, pressao0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([pressao, pressao], [0, 1], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Ativação agregada e Resultado (linha preta)')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

print("Pressão aplicada: " + str(pressao))

plt.show()