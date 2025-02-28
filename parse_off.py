import itertools
import numpy as np

def tuple_to_pairs(input_tuple):
    cycled = itertools.cycle(input_tuple)
    next(cycled)
    pairs = list(zip(input_tuple, cycled))
    return pairs


def parse_off_file(file_path):
    edges = set()

    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')][1:]

    vertices_count, faces_count, *_ = map(int, lines[0].split())
    lines = lines[1:]
    
    vertices = list(map(lambda l: np.array(list(map(float, l.split()))), lines[:vertices_count]))
    faces = list(map(lambda l: list(map(int, l.split('\t')[0].strip().split()[1:])), lines[vertices_count:vertices_count + faces_count]))
    for face in faces:
        edges.update(map(lambda p: tuple(sorted(p)), tuple_to_pairs(face)))
        
    return vertices, list(edges)
