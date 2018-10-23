import numpy as np
import skfuzzy as fuzz
import time
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions
distancia = ctrl.Antecedent(np.arange(0, 101, 1), 'distancia')
velocidade = ctrl.Antecedent(np.arange(0, 101, 1), 'velocidade')
pressao = ctrl.Consequent(np.arange(0, 101, 1), 'pressao')

# Auto-membership function population is possible with .automf(3, 5, or 7)
distancia.automf(3)
velocidade.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
pressao['low'] = fuzz.trimf(pressao.universe, [0, 0, 40])
pressao['medium'] = fuzz.trimf(pressao.universe, [20, 50, 80])
pressao['high'] = fuzz.trimf(pressao.universe, [60, 100, 100])

# You can see how these look with .view()
distancia['average'].view()

velocidade.view()

pressao.view()

regra1 = ctrl.Rule(velocidade['good'] | distancia['poor'], pressao['high'])
regra2 = ctrl.Rule(velocidade['good'] & distancia['good'], pressao['medium'])
regra3 = ctrl.Rule(velocidade['poor'] | distancia['good'], pressao['low'])
regra4 = ctrl.Rule(velocidade['poor'] & distancia['poor'], pressao['medium'])
regra5 = ctrl.Rule(velocidade['good'] & distancia['good'], pressao['high'])


tipping_ctrl = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5])

tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
tipping.input['distancia'] = 10
tipping.input['velocidade'] = 10 

# Crunch the numbers
tipping.compute()

print(tipping.output['pressao'])
pressao.view(sim=tipping)

