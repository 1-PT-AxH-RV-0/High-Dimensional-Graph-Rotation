import os

def validate_config(config):
    """验证配置文件基本结构正确性"""
    errors = []
    allowed_top_level = {'graphs', 'video', 'drawing', 'initial', 'actions'}
    
    # 预定义有效名称映射
    VALID_NAMES = {
        'RegularPolyhedron': [
            'tetrahedron', 'hexahedron', 'octahedron', 'dodecahedron', 'icosahedron'
        ],
        'RegularPolychoron': [
            'pentachoron', 'tesseract', 'hexadecachoron', 'icositetrachoron',
            'hecatonicosachoron', 'hexacosichoron'
        ],
        'RegularStarPolyhedron': [
            'great_dodecahedron', 'small_stellated_dodecahedron',
            'great_stellated_dodecahedron', 'great_icosahedron'
        ],
        'RegularStarPolychora': [
            'great_hecatonicosachoron', 'grand_hecatonicosachoron',
            'great_grand_hecatonicosachoron', 'small_stellated_hecatonicosachoron',
            'great_stellated_hecatonicosachoron', 'grand_stellated_hecatonicosachoron',
            'great_grand_stellated_hecatonicosachoron', 'faceted_hexacosichoron',
            'great_faceted_hexacosichoron', 'grand_hexacosichoron'
        ],
        'RegularPolyhedronCompounds': [
            'stellated_octahedron', 'chiricosahedron', 'icosicosahedron',
            'rhombihedron', 'small_icosicosahedron'
        ],
        'TruncatedRegularPolyhedron': [
            'truncated_tetrahedron', 'truncated_hexahedron',
            'truncated_octahedron', 'truncated_dodecahedron',
            'truncated_icosahedron'
        ],
        'RectifiedRegularPolyhedron': [
            'rectified_tetrahedron', 'rectified_hexahedron',
            'rectified_octahedron', 'rectified_dodecahedron',
            'rectified_icosahedron'
        ],
        'TruncatedRegularStarPolyhedron': [
            'truncated_great_dodecahedron', 'truncated_small_stellated_dodecahedron',
            'truncated_great_stellated_dodecahedron', 'truncated_great_icosahedron'
        ],
        'RectifiedRegularStarPolyhedron': [
            'rectified_great_dodecahedron', 'rectified_small_stellated_dodecahedron',
            'rectified_great_stellated_dodecahedron', 'rectified_great_icosahedron'
        ],
        'RhombiRectifiedRegularPolyhedron': [
            'rhombi_rectified_tetrahedron', 'rhombi_rectified_hexahedron',
            'rhombi_rectified_octahedron', 'rhombi_rectified_dodecahedron',
            'rhombi_rectified_icosahedron'
        ],
        'GreatRhombiRectifiedRegularPolyhedron': [
            'great_rhombi_rectified_tetrahedron', 'great_rhombi_rectified_hexahedron',
            'great_rhombi_rectified_octahedron', 'great_rhombi_rectified_dodecahedron',
            'great_rhombi_rectified_icosahedron'
        ]
    }

    # 检查顶级字段
    for key in config:
        if key not in allowed_top_level:
            errors.append(f"未声明的顶级字段 '{key}'。")

    # 检查graphs部分
    if 'graphs' not in config:
        errors.append("缺少必填的顶级字段 'graphs'。")
    else:
        if not isinstance(config['graphs'], list):
            errors.append("graphs 必须是一个列表。")
        else:
            seen_ids = set()
            for idx, graph in enumerate(config['graphs']):
                if not isinstance(graph, dict):
                    errors.append(f"graphs[{idx}] 必须是一个字典。")
                    continue
                
                # 检查必填字段
                for field in ['type', 'id']:
                    if field not in graph:
                        errors.append(f"graphs[{idx}] 缺少必填字段 '{field}'。")

                # 检查类型有效性
                graph_type = graph.get('type')
                valid_types = {
                    'RegularPolygon', 'RegularStarPolygon',
                    'Simplex', 'Hypercube', 'Orthoplex', 'OffFile'
                } | set(VALID_NAMES.keys())
                if graph_type and graph_type not in valid_types:
                    errors.append(f"graphs[{idx}].type 无效值 '{graph_type}'。")

                # 检查条件必填字段
                if graph_type in VALID_NAMES.keys():
                    if 'name' not in graph:
                        errors.append(f"graphs[{idx}] 需要字段 'name'（当 type 为 {graph_type} 时）。")
                    else:
                        # 检查名称有效性
                        valid_names = VALID_NAMES.get(graph_type, [])
                        if graph['name'] not in valid_names:
                            errors.append(
                                f"graphs[{idx}].name 无效值 '{graph['name']}'，" 
                                f"有效的 {graph_type} 名称应为：{', '.join(valid_names)}。"
                            )
                elif graph_type in {'RegularPolygon', 'RegularStarPolygon'}:
                    if 'edge_count' not in graph:
                        errors.append(f"graphs[{idx}] 需要字段 'edge_count'（当 type 为 {graph_type} 时）。")
                    if graph_type == 'RegularStarPolygon' and 'gap' not in graph:
                        errors.append(f"graphs[{idx}] 需要字段 'gap'（当 type 为 RegularStarPolygon 时）。")
                elif graph_type in {'Simplex', 'Hypercube', 'Orthoplex'}:
                    if 'dimensions' not in graph:
                        errors.append(f"graphs[{idx}] 需要字段 'dimensions'（当 type 为 {graph_type} 时）。")
                elif graph_type == 'OffFile':
                    if 'path' not in graph:
                        errors.append(f"graphs[{idx}] 需要字段 'path'（当 type 为 OffFile 时）。")
                        continue
                    if not isinstance(graph['path'], str):
                        errors.append(f"graphs[{idx}].path 类型错误，应为字符串。")
                        continue
                    if not os.path.isfile(graph['path']):
                        errors.append(f"graphs[{idx}].path 所指向的路径不是一个文件。")

                # 检查字段类型
                type_checks = [
                    ('name', str, '字符串'),
                    ('edge_count', int, '整数'),
                    ('gap', int, '整数'),
                    ('dimensions', int, '整数'),
                    ('id', str, '字符串')
                ]
                for field, expected_type, expected_type_name in type_checks:
                    if field in graph and not isinstance(graph[field], expected_type):
                        errors.append(f"graphs[{idx}].{field} 类型错误，应为{expected_type_name}。")

                # 检查ID唯一性
                if 'id' in graph:
                    if graph['id'] in seen_ids:
                        errors.append(f"重复的图形 ID: {graph['id']}。")
                    seen_ids.add(graph['id'])

                # 检查未声明字段
                allowed_fields = {'type', 'name', 'edge_count', 'gap', 'dimensions', 'path', 'id'}
                for key in graph:
                    if key not in allowed_fields:
                        errors.append(f"graphs[{idx}] 包含未声明的字段 '{key}'。")

    # 检查video部分
    if 'video' in config:
        video = config['video']
        if not isinstance(video, dict):
            errors.append("video 必须是一个字典。")
        else:
            allowed_fields = {'output_path', 'fps', 'width', 'height', 'end_pause_frames'}
            for key in video:
                if key not in allowed_fields:
                    errors.append(f"video 包含未声明的字段 '{key}'。")

            type_checks = [
                ('output_path', str, '字符串'),
                ('fps', int, '整数'),
                ('width', int, '整数'),
                ('height', int, '整数'),
                ('end_pause_frames', int, '整数')
            ]
            for field, expected_type, expected_type_name in type_checks:
                if field in video and not isinstance(video[field], expected_type):
                    errors.append(f"video.{field} 类型错误，应为{expected_type_name}。")

    # 检查drawing部分
    if 'drawing' in config:
        drawing = config['drawing']
        if not isinstance(drawing, dict):
            errors.append("drawing 必须是一个字典。")
        else:
            allowed_fields = {'scale', 'focal_length', 'line_width', 'line_color', 'background_color', 'ease'}
            for key in drawing:
                if key not in allowed_fields:
                    errors.append(f"drawing 包含未声明的字段 '{key}'。")

            # 特殊检查颜色字段
            for color_field in ['line_color', 'background_color']:
                if color_field in drawing:
                    value = drawing[color_field]
                    if not (isinstance(value, list) and len(value) == 3 and
                            all(isinstance(c, int) for c in value)):
                        errors.append(f"drawing.{color_field} 应为包含 3 个整数的列表。")

            type_checks = [
                ('scale', (int, float), '数字'),
                ('focal_length', (int, float), '数字'),
                ('line_width', (int, float), '数字'),
            ]
            for field, expected_type, expected_type_name in type_checks:
                if field in drawing and not isinstance(drawing[field], expected_type):
                    errors.append(f"drawing.{field} 应为{expected_type_name}。")

    # 检查initial部分
    if 'initial' in config:
        initial = config['initial']
        if not isinstance(initial, list):
            errors.append("initial 必须是一个列表。")
        else:
            graph_ids = {g['id'] for g in config.get('graphs', []) if isinstance(g, dict) and 'id' in g}
            for idx, item in enumerate(initial):
                if not isinstance(item, dict):
                    errors.append(f"initial[{idx}] 必须是一个字典。")
                    continue

                # 检查必填字段
                if 'target' not in item:
                    errors.append(f"initial[{idx}] 缺少必填字段 'target'。")
                elif item['target'] not in graph_ids:
                    errors.append(f"initial[{idx}].target 指向不存在的图形 ID。")

                # 检查字段类型
                if 'offset' in item:
                    if not (isinstance(item['offset'], list) and
                            all(isinstance(x, (int, float)) for x in item['offset'])):
                        errors.append(f"initial[{idx}].offset 应为数字列表。")

                # 检查旋转动作
                if 'rotations' in item:
                    rotations = item['rotations']
                    if not isinstance(rotations, list):
                        errors.append(f"initial[{idx}].rotations 必须是一个列表。")
                    else:
                        for r_idx, rotation in enumerate(rotations):
                            if not isinstance(rotation, dict):
                                errors.append(f"initial[{idx}].rotations[{r_idx}] 必须是一个字典。")
                                continue
                            
                            if 'plane' not in rotation:
                                errors.append(f"initial[{idx}].rotations[{r_idx}] 缺少必填字段 'plane'。")
                            else:
                                plane = rotation['plane']
                                if not (isinstance(plane, list) and len(plane) == 2 and
                                        all(isinstance(p, int) for p in plane)):
                                    errors.append(f"initial[{idx}].rotations[{r_idx}].plane 应为两个整数的列表。")
                            
                            if 'angle' not in rotation:
                                errors.append(f"initial[{idx}].rotations[{r_idx}] 缺少必填字段 'angle'。")
                            elif not isinstance(rotation['angle'], (int, float)):
                                errors.append(f"initial[{idx}].rotations[{r_idx}].angle 应为数字类型。")

                # 检查未声明字段
                allowed_fields = {'offset', 'rotations', 'move_priority', 'target'}
                for key in item:
                    if key not in allowed_fields:
                        errors.append(f"initial[{idx}] 包含未声明的字段 '{key}'。")

    # 检查actions部分
    if 'actions' in config:
        actions = config['actions']
        if not isinstance(actions, list):
            errors.append("actions 必须是一个列表。")
        else:
            valid_action_types = {'move', 'rotate', 'rotate_complexly'}
            for idx, action in enumerate(actions):
                action_type = action.get('type')
                if action_type and action_type not in valid_action_types:
                    errors.append(f"actions[{idx}].type 无效动作类型 '{action_type}'。")
            graph_ids = {g['id'] for g in config.get('graphs', []) if isinstance(g, dict) and 'id' in g}
            for idx, action in enumerate(actions):
                if not isinstance(action, dict):
                    errors.append(f"actions[{idx}] 必须是一个字典。")
                    continue

                # 检查必填字段
                for field in ['type', 'start', 'target']:
                    if field not in action:
                        errors.append(f"actions[{idx}] 缺少必填字段 '{field}'。")
                if 'target' in action and action['target'] not in graph_ids:
                    errors.append(f"actions[{idx}].target 指向不存在的图形 ID。")

                # 根据类型检查字段
                action_type = action.get('type')
                if action_type == 'move':
                    required = ['offset', 'duration']
                    for field in required:
                        if field not in action:
                            errors.append(f"actions[{idx}].move 缺少必填字段 '{field}'。")
                elif action_type == 'rotate':
                    required = ['plane', 'angle', 'duration']
                    for field in required:
                        if field not in action:
                            errors.append(f"actions[{idx}].rotate 缺少必填字段 '{field}'。")
                elif action_type == 'rotate_complexly':
                    if 'rotations' not in action:
                        errors.append(f"actions[{idx}] 缺少必填字段 'rotations'。")
                    else:
                        rotations = action['rotations']
                        if not isinstance(rotations, list):
                            errors.append(f"actions[{idx}].rotations 必须是一个列表。")
                        else:
                            for r_idx, rotation in enumerate(rotations):
                                if not isinstance(rotation, dict):
                                    errors.append(f"actions[{idx}].rotations[{r_idx}] 必须是一个字典。")
                                    continue
                                required = ['plane', 'angle', 'duration']
                                for field in required:
                                    if field not in rotation:
                                        errors.append(f"actions[{idx}].rotations[{r_idx}] 缺少字段 '{field}'。")

                # 检查过渡函数
                allowed_ease = ['sin', 'linear']
                if 'ease' in action and action['ease'] not in allowed_ease:
                    errors.append(f"actions[{idx}].ease 取值只能为：{', '.join(allowed_ease)}。")
                
                # 检查未声明字段
                allowed_base = {'type', 'start', 'target', 'ease'}
                action_allowed = {
                    'move': allowed_base | {'offset', 'duration', 'priority'},
                    'rotate': allowed_base | {'plane', 'angle', 'duration', 'center', 'priority'},
                    'rotate_complexly': allowed_base | {'rotations'}
                }
                allowed = action_allowed.get(action_type, set())
                for key in action:
                    if key not in allowed:
                        errors.append(f"actions[{idx}] 包含未声明的字段 '{key}'。")

    return errors
