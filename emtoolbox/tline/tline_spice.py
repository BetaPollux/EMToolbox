#!/usr/bin/python3

"""Generate SPICE models for Transmission Lines."""

import numpy as np


def node_str(num: int) -> str:
    return f'N{num:05d}'


def model_str(name: str, num: int, value: float, node0: str, node1: str) -> str:
    return f'{name}{num:03d} {node0} {node1} {value:.5e}\n'


def coupling_str(num:int, i: int, j: int, val: float) -> str:
    return f'K{i}{j}_{num:03d} L{i}{i}_{num:03d} L{j}{j}_{num:03d} {val:.5e}\n'


def pi_model_2c(N: int, *, L: float, C: float,
                R: float = 0, R0: float = 0, G: float = 0,
                length: float = 1.0, name: str = 'TLINE_PI') -> str:
    assert N > 0 and N < 990
    assert R0 == 0  # TODO support R0, resistance of reference wire
    capacitors = [C * length / N] * (N + 1)
    capacitors[0] *= 0.5
    capacitors[-1] *= 0.5
    inductors = [L * length / N] * N
    if R > 0:
        resistors = [R * length / N] * N
    if G > 0:
        resistors_g = [N / (G * length)] * (N + 1)
        resistors_g[0] *= 2
        resistors_g[-1] *= 2
    nets1 = [node_str(i) for i in range(N if R == 0 else 2 * N)]
    nets1[0] = '101'
    nets1.append('201')
    nets0 = ['100']
    nets1_step = 1 if R == 0 else 2
    l_idx = list(range(0, len(nets1), nets1_step))  # Inductor left side
    c_idx = list(range(nets1_step, len(nets1), nets1_step))  # Capacitor top
    result = f'* Pi-model, 2-conductor transmission line\n'
    result += f'* L {L:.3e} H/m\n* C {C:.3e} F/m\n* R {R:.3e} Ohm/m\n* G {G:.3e} S/m\n'
    result += f'* {N} segments, {length:.3e} m length\n'
    result += f'.SUBCKT {name} 100 101 201\n'
    result += model_str('C11_', 1, capacitors[0], nets1[0], nets0[0])
    if G > 0:
        result += model_str('RG11_', 1, resistors_g[0], nets1[0], nets0[0])
    for i in range(N):
        result += model_str('L11_', i + 1, inductors[i], nets1[l_idx[i]], nets1[l_idx[i] + 1])
        if R > 0:
            result += model_str('R1_', i + 1, resistors[i], nets1[l_idx[i] + 1], nets1[l_idx[i] + 2])
        result += model_str('C11_', i + 2, capacitors[i + 1], nets1[c_idx[i]], nets0[0])
        if G > 0:
            result += model_str('RG11_', i + 2, resistors_g[i + 1], nets1[c_idx[i]], nets0[0])
    result += f'.ENDS {name}\n'
    return result


def get_inductances(L: np.ndarray, length: float = 1.0):
    '''Returns Le and Ke matrices of inductor values and coupling coefficients
    Input is per unit inductance matrix, size N x N
    Le are individual inductor values, size N
    Ke are coupling coefficients, size N x N'''
    assert L.ndim == 2
    Le = L.diagonal() * length
    N = L.shape[0]
    Ke = np.zeros_like(L)
    for i in range(N):
        for j in range(N):
            Ke[i, j] = L[i, j] / np.sqrt(L[i, i] * L[j, j])
    return Le, Ke


def get_capacitances(C: np.ndarray, length: float = 1.0):
    '''Returns Ce matrix of capacitor values
    Input is per unit capacitance matrix, size N x N
    Ce are individual capacitor values, size N+1 x N+1'''
    assert C.ndim == 2
    Ce = -length * C  # Mutual capacitances
    np.fill_diagonal(Ce, C.sum(axis=1) * length)  # Self capacitances
    return Ce


def pi_model_mtl(NS: int, *, L: np.ndarray, C: np.ndarray,
                 R: np.ndarray = None, G: float = None,
                 length: float = 1.0, name: str = 'TLINE_PI') -> str:
    assert NS == 1  # TODO support multiple segments
    assert L.shape == C.shape
    NC = L.shape[0]
    assert NC < 10  # TODO support 2 digit conductor num
    Ce = get_capacitances(C, length)
    Le, Ke = get_inductances(L, length)
    result = f'* Pi-model, {NC+1}-conductor transmission line\n'
    result += f'* {NS} segments, {NC+1} conductors, {length:.3e} m length\n'
    ports = [f'{p}{j:02d}' for p in (1, 2) for j in range(1, NC + 1)]
    result += f'.SUBCKT {name} 100 {" ".join(ports)}\n'
    for i in range(1, NC + 1):
        result += model_str(f'C{i}{i}_', 1, 0.5 * Ce[i - 1, i -1], f'1{i:02d}', '100')
        result += model_str(f'L{i}{i}_', 1, Le[i - 1], f'1{i:02d}', f'2{i:02d}')
        result += model_str(f'C{i}{i}_', 2, 0.5 * Ce[i - 1, i - 1], f'2{i:02d}', '100')
        for j in range(i + 1, NC + 1):
            result += coupling_str(1, i, j, Ke[i - 1, j - 1])
        for j in range(i + 1, NC + 1):
            result += model_str(f'C{i}{j}_', 1, 0.5 * Ce[i - 1, j - 1], f'1{i:02d}', f'1{j:02d}')
            result += model_str(f'C{i}{j}_', 2, 0.5 * Ce[i - 1, j - 1], f'2{i:02d}', f'2{j:02d}')
    result += f'.ENDS {name}\n'
    return result


if __name__ == '__main__':
    for i in range(1, 4):
        print(pi_model_2c(i, L=3e-6, C=120e-12))
    print(pi_model_2c(3, L=3e-6, C=120e-12, R=3))
    print(pi_model_2c(3, L=3e-6, C=120e-12, G=1e-9))
    L_3c = np.array([[1.11170e-6, 6.93901e-07],
                     [6.93901e-07, 1.38780e-06]])
    C_3c = np.array([[4.03439E-11, -2.01719E-11],
                     [-2.01719E-11, 2.95910E-11]])
    print(pi_model_mtl(1, L=L_3c, C=C_3c, length=0.254))
    L_4c = np.array([[5e-7, 1e-7, 2e-7],
                     [1e-7, 8e-7, 3e-7],
                     [2e-7, 3e-7, 6e-7]])
    C_4c = np.array([[5e-11, -2e-11, -3e-11],
                     [-2e-11, 7e-11, -1e-11],
                     [-3e-11, -1e-11, 9e-11]])
    print(pi_model_mtl(1, L=L_4c, C=C_4c))
