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
    capacitors = [C * length / N] * (N + 1)
    capacitors[0] *= 0.5
    capacitors[-1] *= 0.5
    inductors = [L * length / N] * N
    nets1 = [node_str(i) for i in range(1, N)]
    nets1.insert(0, '101')
    nets1.append('201')
    nets0 = ['100']
    result = f'.SUBCKT {name} 100 101 201\n'
    result += model_str('C11_', 1, capacitors[0], nets1[0], nets0[0])
    for i in range(N):
        result += model_str('L11_', i + 1, inductors[i], nets1[i], nets1[i + 1])
        result += model_str('C11_', i + 2, capacitors[i + 1], nets1[i + 1], nets0[0])
    result += f'.ENDS {name}\n'
    return result


if __name__ == '__main__':
    for i in range(1, 4):
        print(pi_model_2c(i, L=3e-6, C=120e-12))
