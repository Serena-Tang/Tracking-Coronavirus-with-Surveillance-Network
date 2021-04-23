import numpy as np
import pandas as pd
import sys

OD = pd.read_excel(io = sys.argv[1], sheet_name = "Sheet1")
alpha = float(sys.argv[2]) #amount of public movement

# initialize population vector
pop_size = np.abs(np.diagonal(OD) + OD.sum(axis = 0) - OD.sum(axis = 1))
print(pop_size)
locs_len = len(pop_size) # number of locations
SIR = np.zeros(shape=(locs_len, 3))
SIR[:,0] = pop_size  # initialize the S group with the respective populations

first_infections = np.where(SIR[:, 0] <= 10000, SIR[:, 0] // 20, 0)

SIR[:, 0] = SIR[:, 0] - first_infections
SIR[:, 1] = SIR[:, 1] + first_infections

row_sums = SIR.sum(axis = 1)
SIR_normalized = SIR / row_sums[:, np.newaxis]

beta = 1.4 

gamma = 0.04 
R0 = beta/gamma
beta_vec = np.random.gamma(1.6, 2, locs_len)
gamma_vec = np.full(locs_len, gamma)
alpha_vec = np.full(locs_len, alpha)

s_normalized = []
i_normalized = []
r_normalized = []

for time_step in range(30):

    infected = np.array([SIR_normalized[:, 1], ] * locs_len).transpose()
    OD_infected = np.round(OD * infected)
    current_infected = OD_infected.sum(axis = 0)
    current_infected = np.round(current_infected * alpha_vec)

    print('current infected: ', current_infected.sum())

    new_infected = beta_vec * SIR[:, 0] * current_infected / (pop_size + OD.sum(axis = 0))
    new_recovered = gamma_vec * SIR[:, 1]
    new_infected = np.where(new_infected > SIR[:, 0], SIR[:, 0], new_infected)

    SIR[:, 0] = SIR[:, 0] - new_infected
    SIR[:, 1] = SIR[:, 1] + new_infected - new_recovered
    SIR[:, 2] = SIR[:, 2] + new_recovered
    SIR = np.where(SIR < 0, 0, SIR)

    row_sums = SIR.sum(axis=1)
    SIR_normalized = SIR / row_sums[:, np.newaxis]

    S = SIR[:, 0].sum() / pop_size.sum()
    I = SIR[:, 1].sum() / pop_size.sum()
    R = SIR[:, 2].sum() / pop_size.sum()

    print('S =', S)
    print('I =', I)
    print('R =', R)
    print('(S+I+R)*pop_size = ',(S + I + R) * pop_size.sum())
    print('pop_size = ', pop_size.sum(), '\n')

    s_normalized.append(S)
    i_normalized.append(I)
    r_normalized.append(R)
