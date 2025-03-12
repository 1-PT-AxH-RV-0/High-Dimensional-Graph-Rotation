import numpy as np
import cv2
import math
import itertools
from tqdm import tqdm
from copy import deepcopy
import os
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

from parse_off import parse_off_file
from validate_config import validate_config

CUR_FOLDER = os.path.dirname(__file__)

def sinspace_piece(start, stop, index, num=50, endpoint=True):
    if index > num:
        return 0
    if endpoint:
        angle = np.pi / num * index
        prev_angle = np.pi / num * (index - 1)
    else:
        angle = np.pi / (num + 1) * index
        prev_angle = np.pi / (num + 1) * (index - 1)
    
    sin_values = (np.sin(angle - np.pi / 2) + 1) / 2
    prev_sin_values = (np.sin(prev_angle - np.pi / 2) + 1) / 2
    
    scaled_values = start + (stop - start) * sin_values
    prev_scaled_values = start + (stop - start) * prev_sin_values
    
    return scaled_values - prev_scaled_values

def linspace_piece(start, stop, index, num=50, endpoint=True):
    if index > num:
        return 0
    if endpoint:
        return (start - stop) / num
    return (start - stop) / (num - 1)
    

def pad_arrays(arrays):
    max_length = max(len(arr) for arr in arrays)
    
    padded_arrays = []
    for arr in arrays:
        pad_length = max_length - len(arr)
        if pad_length > 0:
            last_item = arr[-1]
            padding = [last_item] * pad_length
            padded_arr = np.concat((arr, padding))
        else:
            padded_arr = arr
        padded_arrays.append(padded_arr)
    
    return padded_arrays


def get_rotation_matrix(angles, center=None):
    m = len(angles)
    D = 1 + 8 * m
    sqrt_D = math.isqrt(D)
    if sqrt_D ** 2 != D:
        raise ValueError("角度的长度必须为三角形数。")
    k = (1 + sqrt_D) // 2
    if k * (k - 1) // 2 != m:
        raise ValueError("角度的长度无法对应到整数维。")

    planes = []
    for i in range(k):
        for j in range(i + 1, k):
            planes.append((i, j))
    
    total_matrix = np.identity(k)
    for (i, j), angle in zip(planes, angles):
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        rotation_matrix = np.eye(k)
        rotation_matrix[i, i] = cos_theta
        rotation_matrix[i, j] = -sin_theta
        rotation_matrix[j, i] = sin_theta
        rotation_matrix[j, j] = cos_theta
        total_matrix = np.dot(total_matrix, rotation_matrix)
    
    if center is not None:
        center = np.asarray(center)
        if center.shape != (k,):
            raise ValueError(f"旋转中心必须是长度为 {k} 的一维数组")
        t = (np.eye(k) - total_matrix) @ center
        affine_matrix = np.eye(k + 1)
        affine_matrix[:k, :k] = total_matrix
        affine_matrix[:k, k] = t
        return affine_matrix
    else:
        return total_matrix


def rotate_point(point, rotation_matrix):
    k = len(rotation_matrix)

    rotated_part = list(point[:k])
    remain_part = list(point[k:])
    
    vec = np.array(rotated_part)
    rotated_vec = np.dot(rotation_matrix, vec)
    rotated_part = rotated_vec.tolist()
    
    return rotated_part + remain_part


def get_rot_ang(dim, direction, ang):
    if dim < 2:
        raise ValueError("维度必须大于二。")
    
    i, j = direction
    i, j = sorted((i, j))
    if i < 0 or j >= dim or i == j:
        raise ValueError(f"旋转平面 {direction} 在 {dim} 维空间中无效。")
    
    planes = []
    for a in range(dim):
        for b in range(a + 1, dim):
            planes.append((a, b))
    
    try:
        plane_idx = planes.index((i, j))
    except ValueError:
        raise ValueError(f"旋转平面 {direction} 在 {dim} 维空间中找不到。")
    
    ang_rad = math.radians(ang)
    
    rot_angles = [0.0] * (dim * (dim - 1) // 2)
    rot_angles[plane_idx] = ang_rad
    
    return rot_angles


def rotate(angles, vectors, center=None):
    dims_set = set()
    for v in vectors:
        dims_set.add(len(v))
    if len(dims_set) != 1:
        raise ValueError(f"每个向量的维度必须相同。")
    dim = list(dims_set)[0]
    if center is not None and len(center) != dim:
        raise ValueError(f"旋转中心的维度必须与向量的维度相同。")
    
    rotation_matrix = get_rotation_matrix(angles, center)
    res = [rotate_point(np.concat((vector, np.array([1]))), rotation_matrix)[:-1] if center is not None else rotate_point(vector, rotation_matrix) for vector in vectors]

    return res


def move(offset, vectors):
    dims_set = set()
    for v in vectors:
        dims_set.add(len(v))
    if len(dims_set) != 1:
        raise ValueError(f"每个向量的维度必须相同。")
    dim = list(dims_set)[0]

    if len(offset) != dim:
        raise ValueError(f"偏移量的维度必须与向量的维度相同。")
       
    return [vector + np.array(offset) for vector in vectors]


def project_nd_to_2d_perspective(point, focal_length):
    while len(point) > 2:
        depth = point[-1]
        scale = focal_length / max(focal_length + depth, 1e-4)
        point = tuple(coord * scale for coord in point[:-1])
    
    return point    


def clip_line_segment(p0, p1, width, height):
    p0 = np.array(p0, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    
    edges = [
        (-dx, p0[0]),
        (dx, width - p0[0]),
        (-dy, p0[1]),
        (dy, height - p0[1])
    ]
    
    t_enter = 0.0
    t_exit = 1.0
    
    for p, q in edges:
        if p == 0:
            if q < 0:
                return None
            continue
        else:
            t = q / p
            if p < 0:
                if t > t_enter:
                    t_enter = t
            else:
                if t < t_exit:
                    t_exit = t
    
    if t_enter < t_exit and t_enter <= 1 and t_exit >= 0:
        t0 = max(t_enter, 0.0)
        t1 = min(t_exit, 1.0)
        if t0 > t1:
            return None
        
        new_p0 = p0 + t0 * (p1 - p0)
        new_p1 = p0 + t1 * (p1 - p0)
        
        new_p0 = np.clip(new_p0, [0, 0], [width, height])
        new_p1 = np.clip(new_p1, [0, 0], [width, height])
        
        return (np.asarray(new_p0, np.uint32).tolist(), np.asarray(new_p1, np.uint32).tolist())
    else:
        return None


def get_even_perms(n=3):
    all_perms = list(itertools.permutations(range(n)))
    even_perms = []
    for p in all_perms:
        inv = 0
        for i in range(n):
            for j in range(i + 1, n):
                if p[i] > p[j]:
                    inv += 1
        if inv % 2 == 0:
            even_perms.append(p)
    return even_perms


def even_permutations(iterable, r=None):
    iterable = tuple(iterable)
    if r is None:
        r = len(iterable)
    even_perms = get_even_perms(r)
    return ([iterable[i] for i in even_perm] for even_perm in even_perms)


def generate_signed_combinations(coords):
    non_zero_indices = [i for i, coord in enumerate(coords) if coord != 0]
    non_zero_coords = [coord for coord in coords if coord != 0]

    for signs in itertools.product([-1, 1], repeat=len(non_zero_coords)):
        signed_coords = coords.copy()
        for idx, signed_value in zip(non_zero_indices, signs):
            signed_coords[idx] *= signed_value
        yield signed_coords


def tuple_to_pairs(input_tuple):
    cycled = itertools.cycle(input_tuple)
    next(cycled)
    pairs = list(zip(input_tuple, cycled))
    return pairs


def are_cocircular(points):
    points = [np.asarray(p) for p in points]
    n = len(points)
    
    if n <= 2:
        return True
    elif n == 3:
        p1, p2, p3 = points
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)

        return not np.allclose(cross, 0)
    else:
        A = np.array([[p[0], p[1], 1] for p in points])
        b = -np.array([p[0]**2 + p[1]**2 for p in points])
        
        try:
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return False
        
        residuals = A.dot(x) - b
        residual_norm = np.linalg.norm(residuals)
        
        return residual_norm < 1e-10


def is_regular_graph(points, edges_count, side_len):
    if not are_cocircular(points):
        return False
    distances = np.array([np.linalg.norm(p0 - p1) for p0, p1 in itertools.combinations(points, 2)])
    if len(list(filter(lambda d: np.isclose(d, side_len), distances))) == edges_count:
        return True
    return False


def find_regular_graphs(vectors, edges_count, side_len):
    graphs = set()
    
    for indices in itertools.combinations(range(len(vectors)), edges_count):
        points = [vectors[i] for i in indices]
        if is_regular_graph(points, edges_count, side_len):
            graphs.update(map(lambda p: tuple(sorted(p)),tuple_to_pairs(indices)))
    
    return list(graphs)


def generate_edges(vertices, edge_len):
    edges = []
    num_vertices = len(vertices)
    
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            distance = np.linalg.norm(vertices[i] - vertices[j])
            if np.isclose(distance, edge_len):
                edges.append((i, j))
    
    return edges


def generate_regular_polygon(n):
    return generate_regular_star_polygon(n, 1)


def generate_regular_star_polygon(n, k):
    if math.gcd(n, k) != 1:
        raise ValueError("n 和 k 必须互质。")
    
    if k <= 0 or k >= n/2:
        raise ValueError("k 的取值范围为 1 ＜ k ＜ n/2.")
    
    vertices = []
    for angle in np.linspace(np.pi / 2, 2.5 * np.pi, n, endpoint=False):
        x = math.cos(angle)
        y = math.sin(angle)
        vertices.append((x, y))
    
    edges = []
    for i in range(n):
        start = i
        end = (i + k) % n
        edges.append((start, end))
    
    return vertices, edges


def generate_simplex(d):
    n = d + 1
    r_squared = d / (2 * (d + 1))
    c = -1 / (2 * (d + 1))

    G = np.full((n, n), c)
    np.fill_diagonal(G, r_squared)

    eigenvalues, eigenvectors = np.linalg.eigh(G)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    sqrt_eigenvalues = np.sqrt(eigenvalues[:d])
    X = eigenvectors[:, :d] @ np.diag(sqrt_eigenvalues)

    vertices = X
    
    edges = list(itertools.combinations(range(len(vertices)), 2))

    return vertices, edges


def generate_hypercube(dimensions):
    vertices = list(itertools.product([-1, 1], repeat=dimensions))
    vertices = [np.array(v) for v in vertices]
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if np.sum(np.abs(vertices[i] - vertices[j])) == 2:
                edges.append((i, j))
    return vertices, edges


def generate_orthoplex(n):
    scale = 1.0 / math.sqrt(2)
    vertices = []
    for axis in range(n):
        positive = np.zeros(n)
        positive[axis] = scale
        vertices.append(positive)
        negative = np.zeros(n)
        negative[axis] = -scale
        vertices.append(negative)
    
    return vertices, generate_edges(vertices, 1)


def generate_coords(coords, f=itertools.permutations):
    for coords in generate_signed_combinations(coords):
        for perm in f(coords):
            yield np.array(perm)


class RegularPolyhedron:
    @staticmethod
    def tetrahedron():
        return generate_simplex(3)
    
    @staticmethod
    def hexahedron():
        return generate_hypercube(3)
    
    @staticmethod
    def octahedron():
        return generate_orthoplex(3)
    
    @staticmethod
    def dodecahedron():
        phi = (np.sqrt(5) + 1) / 2
        edge_len = np.sqrt(5) - 1
        vertices = []
        
        coords1 = np.array([1, 1, 1]) / edge_len
        coords2 = np.array([phi, 1 / phi, 0]) / edge_len
        
        vertices.extend(generate_coords(coords1))
        vertices.extend(generate_coords(coords2, even_permutations))
        
        return vertices, generate_edges(vertices, 1)

    @staticmethod
    def icosahedron():
        phi = (np.sqrt(5) + 1) / 2
        vertices = []
        even_perms = get_even_perms()
        
        coords = np.array([1, phi, 0]) / 2
        vertices.extend(generate_coords(coords, even_permutations))
        
        return vertices, generate_edges(vertices, 1)


class RegularPolychoron:
    @staticmethod
    def pentachoron():
        return generate_simplex(4)
    
    @staticmethod
    def tesseract():
        return generate_hypercube(4)

    @staticmethod
    def hexadecachoron():
        return generate_orthoplex(4)
    
    @staticmethod
    def icositetrachoron():
        vertices = []
        
        coords1 = [1, 0, 0, 0]
        coords2 = np.ones((4)) / 2
    
        vertices.extend(generate_coords(coords1))
        vertices.extend(generate_coords(coords2))
                
        vertices = np.unique(np.array(vertices), axis=0)
        
        return vertices, generate_edges(vertices, 1)
    
    @staticmethod
    def hecatonicosachoron():
        vertices = []
        edge_len = 3 - np.sqrt(5)
        
        phi = (1 + math.sqrt(5)) / 2
        phi_squared = phi ** 2
        phi_inv = 1 / phi
        phi_inv_squared = phi_inv ** 2
        sqrt5 = math.sqrt(5)
        
        coords1 = np.array([2, 2, 0, 0]) / edge_len
        coords2 = np.array([1, 1, 1, sqrt5]) / edge_len
        coords3 = np.array([phi_inv_squared, phi, phi, phi]) / edge_len
        coords4 = np.array([phi_inv, phi_inv, phi_inv, phi_squared]) / edge_len
        coords5 = np.array([0, phi_inv_squared, 1, phi_squared]) / edge_len
        coords6 = np.array([0, phi_inv, phi, sqrt5]) / edge_len
        coords7 = np.array([phi_inv, 1, phi, 2]) / edge_len
                
        vertices.extend(generate_coords(coords1))
        vertices.extend(generate_coords(coords2))
        vertices.extend(generate_coords(coords3))
        vertices.extend(generate_coords(coords4))
        vertices.extend(generate_coords(coords5, even_permutations))
        vertices.extend(generate_coords(coords6, even_permutations))
        vertices.extend(generate_coords(coords7, even_permutations))
                
        vertices = np.unique(vertices, axis=0)
        
        return vertices, generate_edges(vertices, 1)
    
    @staticmethod
    def hexacosichoron():
        vertices = []
        phi = (1 + math.sqrt(5)) / 2
        phi_inv = 1 / phi
        
        coords1 = np.array([0, 0, 0, 1]) / phi_inv
        coords2 = np.array([1, 1, 1, 1]) / 2 / phi_inv
        coords3 = np.array([phi, 1, phi_inv, 0]) / 2 / phi_inv
        
        vertices.extend(generate_coords(coords1))
        vertices.extend(generate_coords(coords2))
        vertices.extend(generate_coords(coords3, even_permutations))
        
        vertices = np.unique(vertices, axis=0)
        
        return vertices, generate_edges(vertices, 1)


class RegularStarPolyhedron:
    @staticmethod
    def great_dodecahedron():
        vertices, _ = RegularPolyhedron.icosahedron()
        edges = find_regular_graphs(vertices, 5, 1)
        
        return vertices, edges
    
    @staticmethod
    def small_stellated_dodecahedron():
        vertices = []
        
        coords = [0, 0.5, (np.sqrt(5) - 1) / 4]
        vertices.extend(generate_coords(coords))
    
        return vertices, generate_edges(vertices, 1)
    
    @staticmethod
    def great_stellated_dodecahedron():
        vertices = []
        
        coords1 = [(np.sqrt(5) - 1) / 4] * 3
        coords2 = [(3 - np.sqrt(5)) / 4, 0.5, 0]
        
        vertices.extend(generate_coords(coords1))
        vertices.extend(generate_coords(coords2, even_permutations))
        
        vertices = np.unique(vertices, axis=0)
        
        return vertices, generate_edges(vertices, 1)

    @staticmethod
    def great_icosahedron():
        phi = (np.sqrt(5) + 1) / 2
        vertices, _ = RegularPolyhedron.icosahedron()
        edges = find_regular_graphs(vertices, 3, phi)
        
        return list(np.array(vertices) / phi), edges


class RegularStarPolychora:
    @staticmethod
    def great_hecatonicosachoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Gohi.off'))
   
    @staticmethod
    def grand_hecatonicosachoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Gahi.off'))
    
    @staticmethod
    def great_grand_hecatonicosachoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Gaghi.off'))
    
    @staticmethod
    def small_stellated_hecatonicosachoron():
        vertices = []
        
        coords1 = np.array([1, 0, 0, 0])
        coords2 = np.ones((4)) / 2
        coords3 = [(1 + np.sqrt(5)) / 4, (np.sqrt(5) - 1) / 4, 0.5, 0]
        
        vertices.extend(generate_coords(coords1))
        vertices.extend(generate_coords(coords2))
        vertices.extend(generate_coords(coords3, even_permutations))
        
        vertices = np.unique(vertices, axis=0)
        
        return vertices, generate_edges(vertices, 1)
    
    @staticmethod
    def great_stellated_hecatonicosachoron():
        vertices = []
        phi = (1 + math.sqrt(5)) / 2
        
        coords1 = np.array([1, 0, 0, 0]) / phi
        coords2 = np.ones((4)) / 2 / phi
        coords3 = [(3 - np.sqrt(5)) / 4, (np.sqrt(5) - 1) / 4, 0.5, 0]
        
        vertices.extend(generate_coords(coords1))
        vertices.extend(generate_coords(coords2))
        vertices.extend(generate_coords(coords3, even_permutations))
        
        vertices = np.unique(vertices, axis=0)
        
        return vertices, generate_edges(vertices, 1)
    
    @staticmethod
    def grand_stellated_hecatonicosachoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Gashi.off'))
    
    @staticmethod
    def great_grand_stellated_hecatonicosachoron():
        vertices = []
        
        coords1 = np.array([1, 1, 0, 0]) * ((3 - np.sqrt(5)) / 2)
        coords2 = np.concat((np.array([(3 * np.sqrt(5) - 5) / 4]), np.ones(3) * ((3 - np.sqrt(5)) / 4)))
        coords3 = np.concat((np.ones(3) * ((np.sqrt(5) - 2) / 2), np.array([0.5])))
        coords4 = np.concat((np.array([(7 - 3 * np.sqrt(5) ) / 4]), np.ones(3) * ((np.sqrt(5) - 1) / 4)))
        coords5 = [(7 - 3 * np.sqrt(5)) / 4, (3 - np.sqrt(5)) / 4, 0.5, 0]
        coords6 = [(np.sqrt(5) - 2) / 2, (3 * np.sqrt(5) - 5) / 4, 0, (np.sqrt(5) - 1) / 4]
        coords7 = [(np.sqrt(5) - 2) / 2, (3 - np.sqrt(5)) / 4, (3 - np.sqrt(5)) / 2, (np.sqrt(5) - 1) / 4]
        
        vertices.extend(generate_coords(coords1))
        vertices.extend(generate_coords(coords2))
        vertices.extend(generate_coords(coords3))
        vertices.extend(generate_coords(coords4))
        vertices.extend(generate_coords(coords5, even_permutations))
        vertices.extend(generate_coords(coords6, even_permutations))
        vertices.extend(generate_coords(coords7, even_permutations))
        
        vertices = np.unique(vertices, axis=0)
        
        return vertices, generate_edges(vertices, 1)
    
    @staticmethod
    def faceted_hexacosichoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Fix.off'))
    
    @staticmethod
    def great_faceted_hexacosichoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Gofix.off'))
    
    @staticmethod
    def grand_hexacosichoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Gax.off'))


class RegularPolyhedronCompounds:
    @staticmethod
    def stellated_octahedron():
        vertices = []
        
        coords = np.ones((3)) * (np.sqrt(2) / 4)
        vertices.extend(generate_coords(coords))
        
        return vertices, generate_edges(vertices, 1)
    
    @staticmethod
    def chiricosahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Compound_of_five_tetrahedra.off'))
    
    @staticmethod
    def icosicosahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Compound_of_ten_tetrahedra.off'))

    @staticmethod
    def rhombihedron():
        vertices = []
        
        coords1 = np.ones((3)) / 2
        coords2 = [(np.sqrt(5) + 1) / 4, (np.sqrt(5) - 1) / 4, 0]
        
        vertices.extend(generate_coords(coords1))
        vertices.extend(generate_coords(coords2, even_permutations))
        
        vertices = np.unique(vertices, axis=0)
        
        return vertices, generate_edges(vertices, 1)

    @staticmethod
    def small_icosicosahedron():
        vertices = []
        
        coords1 = [np.sqrt(2) / 2, 0, 0]
        coords2 = [(np.sqrt(10) + np.sqrt(2)) / 8, (np.sqrt(10) - np.sqrt(2)) / 8, np.sqrt(2) / 4]
        
        vertices.extend(generate_coords(coords1))
        vertices.extend(generate_coords(coords2, even_permutations))
        
        vertices = np.unique(vertices, axis=0)
        
        return vertices, generate_edges(vertices, 1)


class TruncatedRegularPolyhedron:
    @staticmethod
    def truncated_tetrahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Truncated', 'Truncated_tetrahedron.off'))
    
    @staticmethod
    def truncated_hexahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Truncated', 'Truncated_hexahedron.off'))
    
    @staticmethod
    def truncated_octahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Truncated', 'Truncated_octahedron.off'))
    
    @staticmethod
    def truncated_dodecahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Truncated', 'Truncated_dodecahedron.off'))
    
    @staticmethod
    def truncated_icosahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Truncated', 'Truncated_icosahedron.off'))


class RectifiedRegularPolyhedron:
    @staticmethod
    def rectified_tetrahedron():
        return RegularPolyhedron.octahedron()
    
    @staticmethod
    def rectified_hexahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Rectified', 'Rectified_hexahedron.off'))
    
    @staticmethod
    def rectified_octahedron():
        return RectifiedRegularPolyhedron.rectified_hexahedron()
    
    @staticmethod
    def rectified_dodecahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Rectified', 'Rectified_dodecahedron.off'))
    
    @staticmethod
    def rectified_icosahedron():
        return RectifiedRegularPolyhedron.rectified_dodecahedron()


class TruncatedRegularStarPolyhedron:
    @staticmethod
    def truncated_great_dodecahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Truncated', 'Truncated_great_dodecahedron.off'))
    
    @staticmethod
    def truncated_small_stellated_dodecahedron():
        return RegularPolyhedron.dodecahedron()
    
    @staticmethod
    def truncated_great_stellated_dodecahedron():
        vertices, edges = RegularPolyhedron.icosahedron()
        edges.extend(find_regular_graphs(vertices, 5, 1))
        
        return vertices, edges

    @staticmethod
    def truncated_great_icosahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Truncated', 'Truncated_great_icosahedron.off'))


class RectifiedRegularStarPolyhedron:
    @staticmethod
    def rectified_great_dodecahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Rectified', 'Rectified_great_dodecahedron.off'))
    
    @staticmethod
    def rectified_small_stellated_dodecahedron():
        return RectifiedRegularStarPolyhedron.rectified_great_dodecahedron()
    
    @staticmethod
    def rectified_great_stellated_dodecahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Rectified', 'Rectified_great_stellated_dodecahedron.off'))

    @staticmethod
    def rectified_great_icosahedron():
        return RectifiedRegularStarPolyhedron.rectified_great_stellated_dodecahedron()


class CantellatedRegularPolyhedron:
    @staticmethod
    def cantellated_tetrahedron():
        return RectifiedRegularPolyhedron.rectified_hexahedron()
    
    @staticmethod
    def cantellated_hexahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Rectified', 'Rhombicuboctahedron.off'))
    
    @staticmethod
    def cantellated_octahedron():
        return CantellatedRegularPolyhedron.cantellated_hexahedron()
    
    @staticmethod
    def cantellated_dodecahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Rectified', 'Rhombicosidodecahedron.off'))
    
    @staticmethod
    def cantellated_icosahedron():
        return CantellatedRegularPolyhedron.cantellated_dodecahedron()


class CantitruncatedRegularPolyhedron:
    @staticmethod
    def cantitruncated_tetrahedron():
        return TruncatedRegularPolyhedron.truncated_octahedron()
    
    @staticmethod
    def cantitruncated_hexahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Rectified', 'Great_rhombicuboctahedron.off'))
    
    @staticmethod
    def cantitruncated_octahedron():
        return CantitruncatedRegularPolyhedron.cantitruncated_hexahedron()
    
    @staticmethod
    def cantitruncated_dodecahedron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'Data', 'Rectified', 'Great_rhombicosidodecahedron.off'))
    
    @staticmethod
    def cantitruncated_icosahedron():
        return CantitruncatedRegularPolyhedron.cantitruncated_dodecahedron()


class OutlinePen(BasePen):
    def __init__(self, glyph_set):
        super().__init__(glyph_set)
        self.points = []
        self.edges = []
        self.corner_points = set()
        self._current_point = None
        self._contour_start_index = None

    def moveTo(self, pt):
        self._current_point = pt
        self.points.append(pt)
        self.corner_points.add(len(self.points) - 1)
        self._contour_start_index = len(self.points) - 1

    def lineTo(self, pt):
        start_idx = len(self.points) - 1
        self.points.append(pt)
        self.corner_points.add(len(self.points) - 1)
        self.edges.append((start_idx, start_idx + 1))
        self._current_point = pt

    def qCurveTo(self, *points):
        if not self._current_point:
            return
        start = self._current_point
        control_pts = list(points[:-1])
        end = points[-1]
        
        self.corner_points.add(len(self.points) - 1)

        if not control_pts:
            self.lineTo(end)
            return

        current_start = start
        for i in range(len(control_pts)):
            cp = control_pts[i]
            next_cp = control_pts[i + 1] if i < len(control_pts)-1 else end
            self._decompose_quad_curve(current_start, cp, next_cp)
            current_start = next_cp
        
        self.corner_points.add(len(self.points) - 1)

    def _decompose_quad_curve(self, start, cp, end, steps=7):
        t = np.linspace(0, 1, steps + 1)[1:]
        x = (1 - t)**2 * start[0] + 2 * (1 - t) * t * cp[0] + t**2 * end[0]
        y = (1 - t)**2 * start[1] + 2 * (1 - t) * t * cp[1] + t**2 * end[1]
        new_points = list(zip(x, y))

        start_idx = len(self.points) - 1
        self.points.extend(new_points)

        edge_indices = np.column_stack([
            np.arange(start_idx, start_idx + len(new_points)),
            np.arange(start_idx + 1, start_idx + len(new_points) + 1)
        ])
        self.edges.extend(map(tuple, edge_indices))

        self._current_point = end

    def closePath(self):
        if self._contour_start_index is not None:
            last_idx = len(self.points) - 1
            self.edges.append((last_idx, self._contour_start_index))
        self._current_point = None


def normalize_points(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    norms = np.linalg.norm(centered, axis=1)
    max_norm = np.max(norms)
    
    if max_norm > 1e-12:
        centered /= max_norm
    
    return centered


def text_to_points_edges(text, font_path, elevations=[]):
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    
    all_points = []
    all_edges = []
    x_offset = 0
    
    for i, char in enumerate(text):
        unicode_val = ord(char)
        if unicode_val not in cmap:
            raise ValueError(f"字符 '{char}' (U+{unicode_val:04X}) 不存在于字体中。")
        glyph_name = cmap[unicode_val]
        glyph = glyph_set[glyph_name]
                
        pen = OutlinePen(glyph_set)
        glyph.draw(pen)
        
        if pen.points:
            char_points = np.array(pen.points, dtype=np.float64)
            char_points[:, 0] += x_offset
            
            corner_points = pen.corner_points
            edges = pen.edges
            for elevation in elevations:
                match elevation['type']:
                    case 'cylindrify':
                        char_points, edges, corner_points = cylindrify(char_points, edges, elevation['height'], corner_points)
                    case 'conify':
                        char_points, edges, corner_points = conify(char_points, edges, elevation['height'], corner_points)
            
            point_offset = len(all_points)
            char_edges = (np.array(edges) + point_offset).tolist()
            
            all_points.extend(char_points)
            all_edges.extend(char_edges)
            
        x_advance = glyph.width + glyph.lsb
        x_offset += x_advance
    
    return normalize_points(all_points), all_edges


def cylindrify(vertices, edges, height, corner_points=None):
    l = len(vertices)
    extra_edges = list(map(lambda edge: [edge[0] + l, edge[1] + l], edges))
    edges.extend(extra_edges)
    for i in range(l):
        if corner_points is None or i in corner_points:
            edges.append((i, i + l))
        
    extra_vertices = list(map(lambda v: np.concat([v, [height]]), vertices))
    vertices = list(map(lambda v: np.concat([v, [0]]), vertices))
    vertices.extend(extra_vertices)
    
    if corner_points is None:
        return vertices, edges
    
    extra_corner_points = list(map(lambda corner_point: corner_point + l, corner_points))
    corner_points.update(extra_corner_points)
    
    return vertices, edges, corner_points


def conify(vertices, edges, height, corner_points=None):
    vertices = list(map(lambda v: np.concat([v, [0]]), vertices))
    centroid = np.mean(vertices, axis=0)
    l = len(vertices)
    
    centroid[-1] += height
    for i in range(l):
        if corner_points is None or i in corner_points:
            edges.append((i, l))
    
    vertices.append(centroid)
    
    if corner_points is None:
        return vertices, edges
    
    corner_points.add(l)
    
    return vertices, edges, corner_points


def get_duration(action):
    duration = action.get('duration')
    if duration is None:
        duration = max(action['rotations'], key=lambda rotation: rotation['duration'])['duration']
    
    return duration


def get_current_actions(actions, frame):
    return [(action, frame - action['start']) for action in actions if action['start'] < frame <= action['start'] + get_duration(action)]
        

def get_last_frame(actions):
    def get_last_frame_with_action(action):
        return action['start'] + get_duration(action)
    
    return get_last_frame_with_action(max(actions, key=get_last_frame_with_action))


def generate_frame(graphs, transformation_datas):
    frame = {}
    for graph_id, (vertices, edges, dim) in graphs.items():
        sorted_transformation_data = sorted(transformation_datas[graph_id].items(), key=lambda i: i[0], reverse=True)

        transformed_vertices = vertices
        for _, transformation_data in sorted_transformation_data:
            for center, angles in transformation_data['rotate'].items():
                transformed_vertices = rotate(angles, transformed_vertices, center)
            transformed_vertices = move(transformation_data['offset'], transformed_vertices)
        
        frame[graph_id] = (transformed_vertices, edges)
    
    return frame


def create_rotation_video(config):
    errors = validate_config(config)
    if errors:
        raise ValueError('配置文件中有错误：\n' + '\n'.join(errors))

    graphs_config = config['graphs']
    graphs = {}
    for graph_config in tqdm(graphs_config, desc="生成顶点和边"):
        graph_type = graph_config['type']
        graph_id = graph_config['id']
        
        match graph_type:
            case 'RegularPolyhedron' | 'RegularPolychoron' | 'RegularStarPolyhedron' | 'RegularStarPolychora' | 'RegularPolyhedronCompounds' | 'UniformPolyhedronCompounds' | 'TruncatedRegularPolyhedron' | 'RectifiedRegularPolyhedron' | 'TruncatedRegularStarPolyhedron' | 'RectifiedRegularStarPolyhedron' | 'CantellatedRegularPolyhedron' | 'CantitruncatedRegularPolyhedron':
                graph = getattr(globals()[graph_type], graph_config['name'])()
            case "RegularPolygon":
                graph = generate_regular_polygon(graph_config['edge_count'])
            case "RegularStarPolygon":
                graph = generate_regular_star_polygon(graph_config['edge_count'], graph_config['gap'])
            case 'Simplex' | 'Hypercube' | 'Orthoplex':
                graph = globals()['generate_' + graph_type.lower()](graph_config['dimensions'])
            case 'OffFile':
                graph = parse_off_file(graph_config['path'])
            case 'Text':
                graph = text_to_points_edges(graph_config['text'], graph_config['font_path'], graph_config.get('elevations', []))
            case _:
                raise ValueError('图形类型无效。')
            
        if 'elevations' in graph_config and graph_type != 'Text':
            for elevation in graph_config['elevations']:
                elevation_type = elevation['type']
                elevation_height = elevation['height']
                
                vertices, edges = graph
                vertices = normalize_points(vertices)
                
                match elevation_type:
                    case 'cylindrify':
                        vertices, edges = cylindrify(vertices, edges, elevation_height)
                    case 'conify':
                        vertices, edges = conify(vertices, edges, elevation_height)
                
                vertices = normalize_points(vertices)
                
                graph = vertices, edges
        
        graph_dim = len(graph[0][0])
        graphs[graph_id] = *graph, graph_dim
    
    
    video_config = config.get('video', {})
    output_path = video_config.get('output_path', os.path.join(CUR_FOLDER, 'rotation.mp4'))
    fps = video_config.get('fps', 30)
    width = video_config.get('width', 1920)
    height = video_config.get('height', 1080)
    end_pause_frames = video_config.get('end_pause_frames', fps *  2)
    
    drawing_config = config.get('drawing', {})
    scale = drawing_config.get('scale', 300)
    focal_length = drawing_config.get('focal_length', 12)
    line_width = drawing_config.get('line_width', 5)
    line_color = drawing_config.get('line_color', [0, 0, 0])[::-1]
    background_color = drawing_config.get('background_color', [255, 255, 255])[::-1]
    
    transformation_datas = {}
    for graph_id, _ in graphs.items():
        transformation_datas[graph_id] = {}
    
    initial_configs = config.get('initial', [])
    
    for initial_config in tqdm(initial_configs, desc="初始化帧"):
        target = initial_config['target']
        vertices, edges, dim = graphs[target]
        target_transformation_data = transformation_datas[target]
        
        offset = initial_config.get('offset', [0] * dim)
        move_priority = initial_config.get('move_priority', 0)
        
        rotations = []
        for r in initial_config.get('rotations', []):
            center = tuple(r.get('center', [0] * dim))
            priority = r.get('priority', 0)
            plane = r['plane']
            angle = r['angle']
            
            if priority not in target_transformation_data:
                target_transformation_data[priority] = {
                    'offset': np.zeros((dim)),
                    'rotate': {}
                }
            
            if center not in target_transformation_data[priority]['rotate']:
                target_transformation_data[priority]['rotate'][center] = np.array(get_rot_ang(dim, plane, angle))
            else:
                target_transformation_data[priority]['rotate'][center] += np.array(get_rot_ang(dim, plane, angle))
        
        if move_priority not in target_transformation_data:
            target_transformation_data[move_priority] = {
                'offset': np.zeros((dim)),
                'rotate': {}
            }
        target_transformation_data[move_priority]['offset'] = offset
    
    frames = [deepcopy(transformation_datas)]
    actions = config.get('actions')
    
    if actions is not None:
        for frame in tqdm(range(1, get_last_frame(actions) + 1), desc="解析动作和生成帧"):
            current_actions = get_current_actions(actions, frame)
            for action, past in current_actions:
                target = action['target']
                _, _, dim = graphs[target]
                target_transformation_data = transformation_datas[target]
                match action.get('ease', 'sin'):
                    case 'sin':
                        ease_func = sinspace_piece
                    case 'linear':
                        ease_func = linspace_piece
                    case _:
                        raise ValueError('过渡函数无效。')
                
                if action['type'] != 'rotate_complexly':
                    priority = action.get('priority', 0)
                    if priority not in target_transformation_data:
                        target_transformation_data[priority] = {
                            'offset': np.zeros((dim)),
                            'rotate': {}
                        }
                match action['type']:
                    case 'move':
                        target_transformation_data[priority]['offset'] += np.array(action['offset']) * ease_func(0, 1, past, action['duration'])
                    case 'rotate':
                        center = tuple(action.get('center', [0] * dim))
                        plane = action['plane']
                        angle = action['angle']
                                            
                        if center not in target_transformation_data[priority]['rotate']:
                            target_transformation_data[priority]['rotate'][center] = np.array(get_rot_ang(dim, plane, angle)) * ease_func(0, 1, past, action['duration'])
                        else:
                            target_transformation_data[priority]['rotate'][center] += np.array(get_rot_ang(dim, plane, angle)) * ease_func(0, 1, past, action['duration'])
                    case 'rotate_complexly':
                        total_duration = get_duration(action)
                        for r in action['rotations']:
                            priority = r.get('priority', 0)
                            center = tuple(r.get('center', [0] * dim))
                            plane = r['plane']
                            angle = r['angle']
                            duration = r['duration']
                            rotation_scale = ease_func(0, duration / total_duration, past, duration)
                            
                            if priority not in target_transformation_data:
                                target_transformation_data[priority] = {
                                    'offset': np.zeros((dim)),
                                    'rotate': {}
                                }
                            
                            if center not in target_transformation_data[priority]['rotate']:
                                target_transformation_data[priority]['rotate'][center] = np.array(get_rot_ang(dim, plane, angle)) * rotation_scale
                            else:
                                target_transformation_data[priority]['rotate'][center] += np.array(get_rot_ang(dim, plane, angle)) * rotation_scale
            
            frames.append(deepcopy(transformation_datas))
    
            
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    last_img_arr = None
    for transformation_datas_frame in tqdm(frames, desc="绘制帧"):
        frame = generate_frame(graphs, transformation_datas_frame)
        img_arr = np.full((height, width, 3), background_color, dtype=np.uint8)
        for vertices, edges in frame.values():
            projected = [project_nd_to_2d_perspective(v, focal_length) for v in vertices]
            scaled = [(int(x*scale + width//2), height - (int(y*scale + height//2))) for x, y in projected]
            
            for edge in edges:
                start, end = edge
                start, end = scaled[start], scaled[end]
                if 0 < start[0] <= width and 0 < start[1] <= height and 0 < end[0] <= width and 0 < end[1] <= height:
                    cv2.line(img_arr, start, end, line_color, line_width)
                else:
                    clipped = clip_line_segment(start, end, width, height)
                    if clipped is not None:
                        cv2.line(img_arr, *clipped, line_color, line_width)
            
        video_writer.write(img_arr)
        last_img_arr = img_arr
    
    if end_pause_frames:
        for _ in tqdm(range(end_pause_frames), desc="添加末尾停顿"):
            video_writer.write(last_img_arr)
    
    video_writer.release()


if __name__ == "__main__":
    import toml

    with open(os.path.join(CUR_FOLDER, 'config.toml'), 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    create_rotation_video(config)
    