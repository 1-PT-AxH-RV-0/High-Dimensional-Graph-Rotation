import numpy as np
import cv2
from tqdm import tqdm
from copy import deepcopy
import os

from parse_off import parse_off_file
from validate_config import validate_config

from glyphs import *
from transform import *
from elevate import *
from geometry import *
from text_to_points_edges import text_to_points_edges

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
                
                match elevation_type:
                    case 'cylindrify':
                        vertices, edges = cylindrify(vertices, edges, elevation_height)
                    case 'conify':
                        vertices, edges = conify(vertices, edges, elevation_height)
                                
                graph = center_points(vertices), edges
        
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
    