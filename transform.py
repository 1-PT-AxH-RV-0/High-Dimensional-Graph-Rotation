import numpy as np

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
    sqrt_D = int(np.floor(np.sqrt(D)))
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
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
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
    
    ang_rad = np.radians(ang)
    
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
