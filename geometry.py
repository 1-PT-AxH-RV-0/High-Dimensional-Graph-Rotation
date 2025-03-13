import numpy as np

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


def center_points(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    return centered