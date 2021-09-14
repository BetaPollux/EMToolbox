#!/usr/bin/python3

'''Generate SPICE models for Transmission Lines'''

'''
L1 N002 N001 0.4µ
C1 N001 N004 10p
C2 N002 N004 20p
C3 N003 N004 10p
L2 N003 N002 0.4µ
.backanno
.end
'''


def node_str(num: int) -> str:
    return f'N{num:05d}'


def model_str(name: str, num: int, value: float, nodes: tuple) -> str:
    return f'{name}{num:03d} {node_str(nodes[0])} {node_str(nodes[1])} {value:.5e}\n'


def get_pi_model(name: str, per_unit: dict, length: float, n_segments: int) -> str:
    inductance = per_unit.get('l', None)
    capacitance = per_unit.get('c', None)
    resistance = per_unit.get('r', None)
    conductance = per_unit.get('g', None)
    if not inductance or not capacitance:
        raise Exception('SPICE model requires inductance and capacitance')
    Ls = inductance * length / n_segments
    Cp = capacitance * length / n_segments

    ref_node = 1
    left_node = 2
    right_node = n_segments + 2
    if resistance:
        right_node += n_segments
    inductor_nodes = range(left_node, right_node, 2 if resistance else 1)
    capacitor_nodes = range(left_node, right_node + 1, 2 if resistance else 1)
    result = f'.SUBCKT {name} {node_str(ref_node)} {node_str(left_node)} {node_str(right_node)}\n'
    for i, n in enumerate(inductor_nodes):
        result += model_str('LS', i+1, Ls, (n, n+1))
    for i, n in enumerate(capacitor_nodes):
        if n == left_node or n == right_node:
            cval = 0.5 * Cp
        else:
            cval = Cp
        result += model_str('CP', i+1, cval, (n, ref_node))
    if resistance:
        Rs = resistance * length / n_segments
        for i, n in enumerate(inductor_nodes):
            result += model_str('RS', i+1, Rs, (n+1, n+2))
    if conductance:
        Gp = conductance * length / n_segments
        for i, n in enumerate(capacitor_nodes):
            if n == left_node or n == right_node:
                rval = 2 / Gp
            else:
                rval = 1 / Gp
            result += model_str('RP', i+1, rval, (n, ref_node))
    result += f'.ENDS {name}\n'
    return result


if __name__ == '__main__':
    print(model_str('LS', 32, 1.6e-6, (1, 2)))
