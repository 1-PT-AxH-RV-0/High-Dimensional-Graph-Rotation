import itertools
import numpy as np

def tuple_to_pairs(input_tuple):
    cycled = itertools.cycle(input_tuple)
    next(cycled)
    pairs = list(zip(input_tuple, cycled))
    return pairs


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_off_file(file_path):
    vertices = []
    edges = set()

    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]
    in_vertices = False
    in_faces = False

    for line in lines:
        if line.startswith('#'):
            in_vertices = False
            in_faces = False
        if '# Vertices' in line:
            in_vertices = True
            in_faces = False
        if '# Faces' in line:
            in_vertices = False
            in_faces = True
        if line and all(map(is_float, line.split())):
            if in_vertices:
                vertices.append(np.array(list(map(float, line.split()))))
            elif in_faces:
                coords = list(map(int, line.split()))[1:]
                pairs = tuple_to_pairs(coords)
                edges.update(map(lambda p: tuple(sorted(p)), pairs))
        
    return vertices, list(edges)
