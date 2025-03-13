import itertools
import numpy as np
import os

from parse_off import parse_off_file

CUR_FOLDER = os.path.dirname(__file__)

# 工具函数
def even_permutations(iterable, r=None):
    def get_even_perms(n):
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


def generate_coords(coords, f=itertools.permutations):
    for coords in generate_signed_combinations(coords):
        for perm in f(coords):
            yield np.array(perm)


# 二维
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


# 单纯形
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


# 超方形
def generate_hypercube(dimensions):
    vertices = list(itertools.product([-1, 1], repeat=dimensions))
    vertices = [np.array(v) for v in vertices]
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if np.sum(np.abs(vertices[i] - vertices[j])) == 2:
                edges.append((i, j))
    return vertices, edges


# 正轴形
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


# 正多面体
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


# 正多胞体
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


# 正星形多面体
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


# 正星形多胞体
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


# 正复合多面体
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


# 截角正多面体
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


# 截半正多面体
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


# 截角正星形多面体
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


# 截半正星形多面体
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


# 小斜方截半正多面体
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


# 大斜方截半正多面体
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
