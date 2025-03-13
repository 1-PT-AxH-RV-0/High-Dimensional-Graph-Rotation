import numpy as np

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