import numpy as np
import cv2
import math
import itertools
from tqdm import tqdm
import os
import toml


from parseOFF import parse_off_file

CUR_FOLDER = os.path.dirname(__file__)

def sinspace(start, stop, num=50, endpoint=True):
    if endpoint:
        angles = np.linspace(0, np.pi, num)
    else:
        angles = np.linspace(0, np.pi, num + 1)[:-1]
    
    sin_values = (np.sin(angles - np.pi / 2) + 1) / 2
    
    scaled_values = start + (stop - start) * sin_values
    
    return scaled_values


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


def get_rot_angs(dim, direction, ang, duration):
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
    angles = sinspace(0, ang_rad, duration + 1)[1:]
    
    res = []
    for angle in angles:
        rot_angles = [0.0] * (dim * (dim - 1) // 2)
        rot_angles[plane_idx] = angle
        
        res.append(rot_angles)
    
    return res


def rotate(rotation_configs, vectors, center=None):
    dims_set = set()
    for v in vectors:
        dims_set.add(len(v))
    if len(dims_set) != 1:
        raise ValueError(f"每个向量的维度必须相同。")
    dim = list(dims_set)[0]
    if center is not None and len(center) != dim:
        raise ValueError(f"旋转中心的维度必须与向量的维度相同。")
    
    rot_angs = []
    for direction, ang, duration in rotation_configs:
        rot_angs.append(get_rot_angs(dim, direction, ang, duration))
    rot_angs = np.sum(pad_arrays(rot_angs), axis=0)
    
    res = []
    for ang in rot_angs:
        res.append([])
        rotation_matrix = get_rotation_matrix(ang, center)
        for vector in vectors:
            if center is not None:
                res[-1].append(rotate_point(np.concat((vector, np.array([1]))), rotation_matrix)[:-1])
            else:
                res[-1].append(rotate_point(vector, rotation_matrix))

    return res


def move(offset, duration, vectors):
    dims_set = set()
    for v in vectors:
        dims_set.add(len(v))
    if len(dims_set) != 1:
        raise ValueError(f"每个向量的维度必须相同。")
    dim = list(dims_set)[0]

    if len(offset) != dim:
        raise ValueError(f"偏移量的维度必须与向量的维度相同。")
    
    offsets = np.array([sinspace(0, o, duration + 1)[1:] for o in offset])
    offsets = np.rot90(offsets)[::-1]
    res = [list(map(lambda v: v + offsets_frame, vectors)) for offsets_frame in offsets]
   
    return res


def move_and_rotate(move_config, rotation_configs, vectors, rotation_center=None):
    offset, move_duration = move_config
    move_vectors = move(offset, move_duration, vectors)
    
    if rotation_center is not None:
        move_centers = move(offset, move_duration, [rotation_center])
    else:
        move_centers = [[None]] * move_duration
    
    dims_set = {len(v) for v in vectors}
    if len(dims_set) != 1:
        raise ValueError("每个向量的维度必须相同。")
    dim = dims_set.pop()
    if len(offset) != dim:
        raise ValueError(f"偏移量的维度必须与向量的维度相同。")
    if rotation_center is not None and len(rotation_center) != dim:
        raise ValueError(f"旋转中心的维度必须与向量的维度相同。")
    
    rot_angs_list = []
    for config in rotation_configs:
        direction, ang, rot_duration = config
        angs = get_rot_angs(dim, direction, ang, rot_duration)
        rot_angs_list.append(angs)

    padded_angs = pad_arrays(rot_angs_list)
    summed_angs = np.sum(padded_angs, axis=0)
    move_vectors, move_centers, summed_angs = pad_arrays([move_vectors, move_centers, summed_angs])
    
    result = []
    for t in range(len(summed_angs)):
        current_moved = move_vectors[t]
        current_ang = summed_angs[t]
        current_center = move_centers[t][0]
        current_matrix = get_rotation_matrix(current_ang, current_center)
        rotated_vectors = []
        for vec in current_moved:
            if current_center is not None:
                homo_vec = np.concatenate((vec, [1]))
                rotated_homo = rotate_point(homo_vec, current_matrix)
                rotated = rotated_homo[:-1]
            else:
                rotated = rotate_point(vec, current_matrix)
            rotated_vectors.append(rotated)
        result.append(rotated_vectors)
    
    return result


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
    return generate_regular_star_polygon(n + 1, 1)


def generate_regular_star_polygon(n, k):
    if math.gcd(n, k) != 1:
        raise ValueError("n 和 k 必须互质。")
    
    if k <= 0 or k >= n/2:
        raise ValueError("k 的取值范围为 1 ＜ k ＜ n/2.")
    
    vertices = []
    for angle in np.linspace(0, 2 * np.pi, n):
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
        return parse_off_file(os.path.join(CUR_FOLDER, 'offData', 'Gohi.off'))
   
    @staticmethod
    def grand_hecatonicosachoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'offData', 'Gahi.off'))
    
    @staticmethod
    def great_grand_hecatonicosachoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'offData', 'Gaghi.off'))
    
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
        return parse_off_file(os.path.join(CUR_FOLDER, 'offData', 'Gashi.off'))
    
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
        return parse_off_file(os.path.join(CUR_FOLDER, 'offData', 'Fix.off'))
    
    @staticmethod
    def great_faceted_hexacosichoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'offData', 'Gofix.off'))
    
    @staticmethod
    def grand_hexacosichoron():
        return parse_off_file(os.path.join(CUR_FOLDER, 'offData', 'Gax.off'))


def process_action(action, frames):
    current = frames[-1]
    action_type = action['type']
    
    if action_type == 'move':
        new_frames = move(
            tuple(action['offset']), 
            action['duration'], 
            current
        )
        frames.extend(new_frames)
    
    elif action_type == 'rotate':
        rotations = []
        for r in action['rotations']:
            rotations.append((
                tuple(r['plane']), 
                r['angle'], 
                r['duration']
            ))
        center = action.get('center')
        new_frames = rotate(rotations, current, center)
        frames.extend(new_frames)
    
    elif action_type == 'move_and_rotate':
        move_offset = tuple(action['move_offset'])
        move_duration = action['move_duration']
        rotations = []
        for r in action['rotations']:
            rotations.append((
                tuple(r['plane']), 
                r['angle'], 
                r['duration']
            ))
        center = action.get('rotate_center')
        
        new_frames = move_and_rotate((move_offset, move_duration), rotations, current, center)
        frames.extend(new_frames)
    
    else:
        raise ValueError(f"未知动作类型: {action_type}")


def create_rotation_video(config_path="config.toml"):
    CUR_FOLDER = os.path.dirname(__file__)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    vertices_config = config['vertices']
    graph_type = vertices_config['type']
    if graph_type in ['RegularPolyhedron', 'RegularPolychoron', 'RegularStarPolyhedron', 'RegularStarPolychora']:
        vertices, edges = getattr(globals()[graph_type], vertices_config['name'])()
    elif graph_type == "RegularPolygon":
        vertices, edges = generate_regular_polygon(vertices_config['edge_count'])
    elif graph_type == "RegularStarPolygon":
        vertices, edges = generate_regular_star_polygon(vertices_config['edge_count'], vertices_config['gap'])
    elif graph_type in ['Simplex', 'Hypercube', 'Orthoplex']:
        vertices, edges = globals()['generate_' + graph_type.lower()](vertices_config['dimensions'])
    else:
        raise ValueError('图形类型无效。')
    
    video_config = config.get('video', {})
    output_path = video_config.get('output_path', os.path.join(CUR_FOLDER, 'rotation.mp4'))
    fps = video_config.get('fps', 30)
    width = video_config.get('width', 1920)
    height = video_config.get('height', 1080)
    
    drawing_config = config.get('drawing', {})
    scale = drawing_config.get('scale', 300)
    focal_length = drawing_config.get('focal_length', 5)
    line_width = drawing_config.get('line_width', 1)
    line_color = drawing_config.get('line_color', [0, 0, 0])
    background_color = drawing_config.get('background_color', [255, 255, 255])
        
    initial = config.get('initial', None)
    if initial is not None:
        move_offset = initial.get('move_offset')
        
        rotations = []
        for r in initial.get('rotations', []):
            rotations.append((
                tuple(r['plane']), 
                r['angle'],
                1
            ))
        
        center = initial.get('center')
        if move_offset and rotations:
            frames = move(move_offset, 1, vertices)
            frames = rotate(rotations, frames[0], center)
        elif move_offset:
            frames = move(move_offset, 1, vertices)
        elif rotations:
            frames = rotate(rotations, vertices, center)
    else:
        frames = [vertices]
    
    actions = config.get('actions', [])
    for action in actions:
        process_action(action, frames)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for vertices_frame in tqdm(frames):
        img_arr = np.full((height, width, 3), background_color, dtype=np.uint8)
        projected = [project_nd_to_2d_perspective(v, focal_length) for v in vertices_frame]
        scaled = [(int(x*scale + width//2), height - (int(y*scale + height//2))) for x, y in projected]
        
        for edge in edges:
            start, end = edge
            start, end = scaled[start], scaled[end]
            clipped = clip_line_segment(start, end, width, height)
            if clipped is not None:
                cv2.line(img_arr, *clipped, line_color, line_width)
        
        video_writer.write(img_arr)
    
    video_writer.release()


if __name__ == "__main__":
    create_rotation_video(os.path.join(CUR_FOLDER, 'config.toml'))
    