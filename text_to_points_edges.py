import numpy

from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

from elevate import *
from geometry import center_points

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


def text_to_points_edges(text, font_path, elevations=[]):
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    
    all_points = []
    all_edges = []
    x_offset = 0
    
    glyph_width = None
    if 0x4E00 in cmap:
        glyph_width = glyph_set[cmap[0x4E00]].width
    elif 0x2001 in cmap:
        glyph_width = glyph_set[cmap[0x2001]].width
    elif 0x2000 in cmap:
        glyph_width = glyph_set[cmap[0x2000]].width * 2
    elif 0x20 in cmap:
        glyph_width = glyph_set[cmap[0x20]].width * 2
    elif 0x61 in cmap:
        glyph_width = glyph_set[cmap[0x61]].width * 2
    
    for i, char in enumerate(text):
        unicode_val = ord(char)
        if unicode_val not in cmap:
            raise ValueError(f"字符 '{char}' (U+{unicode_val:04X}) 不存在于字体中。")
        glyph_name = cmap[unicode_val]
        glyph = glyph_set[glyph_name]
        
        if glyph_width is None:
            glyph_width = glyph.width
        
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
                        char_points, edges, corner_points = cylindrify(char_points, edges, elevation['height'] * glyph_width, corner_points)
                    case 'conify':
                        char_points, edges, corner_points = conify(char_points, edges, elevation['height'] * glyph_width, corner_points)
            
            point_offset = len(all_points)
            char_edges = (np.array(edges) + point_offset).tolist()
            
            all_points.extend(char_points)
            all_edges.extend(char_edges)
            
        x_advance = glyph.width + glyph.lsb
        x_offset += x_advance
    
    return center_points(all_points) / glyph_width, all_edges