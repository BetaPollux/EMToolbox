#!/usr/bin/python3

'''Generate SPICE models for Transmission Lines'''


def node_str(num: int) -> str:
    return f'N{num:05d}'


def model_str(name: str, num: int, value: float, node0: str, node1: str) -> str:
    return f'{name}{num:03d} {node0} {node1} {value:.5e}\n'


def pi_model_2c(N: int, *, L: float, C: float,
                R: float = 0, R0: float = 0, G: float = 0,
                length: float = 1.0, name: str = 'TLINE_PI') -> str:
    assert N > 0
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


if __name__ == '__main__':
    for i in range(1, 4):
        print(pi_model_2c(i, L=3e-6, C=120e-12))
    print(pi_model_2c(3, L=3e-6, C=120e-12, R=3))
    print(pi_model_2c(3, L=3e-6, C=120e-12, G=1e-9))
