# 高维图形旋转

## 简介

这是一个生成高维图形（当然，只要是二维以上都可以）旋转视频的 Python 脚本。

## 依赖项

本脚本依赖以下第三方库，请在使用前安装它们：

- `numpy`
- `opencv-python`
- `tqdm`
- `toml`

## 用法

运行 rotation.py 即可。可以通过 config.toml 来修改配置。（关于配置文件结构，参见[配置文件结构](#配置文件结构)）

## 配置文件结构

配置文件由以下几个部分组成：

1. **顶点配置（`vertices`）**：定义图形的类型和参数。
2. **视频配置（`video`）**：定义视频输出的参数。
3. **绘图配置（`drawing`）**：定义图形的绘制参数。
4. **初始动作（`initial`）**：定义初始的平移和旋转动作。
5. **其他动作（`actions`）**：定义后续的动画动作。

### 字段说明

#### 1. 顶点配置（`vertices`）

| 字段名       | 类型     | 必填 | 描述                                                                                                                                                                                                   |
|--------------|----------|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `type`       | 字符串   | 是   | 图形类型，支持以下值：`RegularPolyhedron`, `RegularPolychoron`, `RegularStarPolyhedron`, `RegularStarPolychora`, `RegularPolygon`, `RegularStarPolygon`, `Simplex`, `Hypercube`, `Orthoplex`。         |
| `name`       | 图形名称 | 条件 | 当 `type` 为 `RegularPolyhedron`, `RegularPolychoron`, `RegularStarPolyhedron`, `RegularStarPolychora` 时必填，表示图形的名称。关于目前支持的图形名称，参见[目前支持的图形名称](#目前支持的图形名称)。 |
| `edge_count` | 整数     | 条件 | 当 `type` 为 `RegularPolygon` 或 `RegularStarPolygon` 时必填，表示边数。                                                                                                                               |
| `gap`        | 整数     | 条件 | 当 `type` 为 `RegularStarPolygon` 时必填，表示星形多边形的间隔。                                                                                                                                       |
| `dimensions` | 整数     | 条件 | 当 `type` 为 `Simplex`, `Hypercube`, `Orthoplex` 时必填，表示图形的维度。                                                                                                                              |

#### 2. 视频配置（`video`）

| 字段名         | 类型   | 必填 | 描述                                              |
|----------------|--------|------|---------------------------------------------------|
| `output_path`  | 字符串 | 否   | 视频输出路径，默认为当前目录下的 `rotation.mp4`。 |
| `fps`          | 整数   | 否   | 视频帧率，默认为 `30`。                           |
| `width`        | 整数   | 否   | 视频宽度，默认为 `1920`。                         |
| `height`       | 整数   | 否   | 视频高度，默认为 `1080`。                         |

#### 3. 绘图配置（`drawing`）

| 字段名            | 类型       | 必填 | 描述                                                |
|-------------------|------------|------|-----------------------------------------------------|
| `scale`           | 浮点       | 否   | 图形的缩放比例，默认为 `300`。                      |
| `focal_length`    | 浮点       | 否   | 焦距，默认为 `5`。                                  |
| `line_width`      | 浮点       | 否   | 线条宽度，默认为 `1`。                              |
| `line_color`      | 整数列表   | 否   | 线条颜色（RGB），默认为 `[0, 0, 0]`（黑色）。       |
| `background_color`| 整数列表   | 否   | 背景颜色（RGB），默认为 `[255, 255, 255]`（白色）。 |

#### 4. 初始动作（`initial`）

| 字段名         | 类型       | 必填 | 描述                                       |
|----------------|------------|------|--------------------------------------------|
| `move_offset`  | 浮点列表   | 否   | 初始平移的偏移量，例如 `[1.0, 2.0, 3.0]`。 |
| `rotations`    | 字典列表   | 否   | 初始旋转动作列表，每个字典包含以下字段：   |
| - `plane`      | 整数列表   | 是   | 旋转平面，例如 `[0, 1]`。                  |
| - `angle`      | 浮点       | 是   | 旋转角度（单位：度）。                     |
| `center`       | 浮点列表   | 否   | 旋转中心点，例如 `[0.0, 0.0, 0.0]`。       |

#### 5. 其他动作（`actions`）

`actions` 是一个列表，每个动作是一个字典，支持以下动作类型：

##### 5.1 平移动作（`move`）

| 字段名     | 类型       | 必填 | 描述                                   |
|------------|------------|------|----------------------------------------|
| `type`     | 字符串     | 是   | 动作类型，必须为 `"move"`。            |
| `offset`   | 浮点列表   | 是   | 平移的偏移量，例如 `[1.0, 2.0, 3.0]`。 |
| `duration` | 整数       | 是   | 动作持续时间（单位：帧）。             |

##### 5.2 旋转动作（`rotate`）

| 字段名       | 类型       | 必填 | 描述                                 |
|--------------|------------|------|--------------------------------------|
| `type`       | 字符串     | 是   | 动作类型，必须为 `"rotate"`。        |
| `rotations`  | 字典列表   | 是   | 旋转动作列表，每个字典包含以下字段： |
| - `plane`    | 整数列表   | 是   | 旋转平面，例如 `[1, 2]`。            |
| - `angle`    | 浮点       | 是   | 旋转角度（单位：度）。               |
| - `duration` | 整数       | 是   | 旋转持续时间（单位：帧）。           |
| `center`     | 浮点列表   | 否   | 旋转中心点，例如 `[0.0, 0.0, 0.0]`。 |

##### 5.3 平移并旋转动作（`move_and_rotate`）

| 字段名           | 类型       | 必填 | 描述                                  |
|------------------|------------|------|---------------------------------------|
| `type`           | 字符串     | 是   | 动作类型，必须为 `"move_and_rotate"`。|
| `move_offset`    | 浮点列表   | 是   | 平移的偏移量，例如 `[1.0, 2.0, 3.0]`。|
| `move_duration`  | 整数       | 是   | 平移动作持续时间（单位：帧）。        |
| `rotations`      | 字典列表   | 是   | 旋转动作列表，每个字典包含以下字段：  |
| - `plane`        | 整数列表   | 是   | 旋转平面，例如 `[0, 1]`。             |
| - `angle`        | 浮点       | 是   | 旋转角度（单位：度）。                |
| - `duration`     | 整数       | 是   | 旋转持续时间（单位：帧）。            |
| `rotate_center`  | 浮点列表   | 否   | 旋转中心点，例如 `[0.0, 0.0, 0.0]`。  |

### 示例配置文件

```toml
[vertices]
type = "Hypercube"
dimensions = 4

[video]
output_path = "rotation.mp4"
fps = 30
width = 1920
height = 1080

[drawing]
scale = 300
line_width = 1
line_color = [230, 230, 230]
background_color = [20, 20, 20]
focal_length = 5

[initial]
move_offset = [1, 1, 1, 1]
rotations = [
    { plane = [0, 2], angle = 45 }
]
center = [1, 1, 1, 1]

[[actions]]
type = "rotate"
rotations = [
    { plane = [0, 2], angle = 360, duration = 100 }
]
center = [1, 1, 1, 1]

[[actions]]
type = "move_and_rotate"
move_offset = [-1, -1, -1, -1]
move_duration = 100
rotations = [
    { plane = [0, 2], angle = -45, duration = 100 }
    { plane = [2, 3], angle = 360, duration = 200 }
]
rotate_center = [1, 1, 1, 1]

[[actions]]
type = "move"
offset = [0.5, 0.5, 0.0, 0.0]
duration = 100

[[actions]]
type = "move_and_rotate"
move_offset = [-1, -1, 0, 0]
move_duration = 100
rotations = [
    { plane = [0, 2], angle = -45, duration = 100 }
    { plane = [2, 3], angle = 360, duration = 200 }
]
rotate_center = [0.5, 0.5, 0.0, 0.0]
```

### 注意事项

1. 如果字段未填写，将使用默认值。
2. 平移并旋转动作为先平移后旋转。
3. 图形名称需填写英文名的小写下划线命名，其他命名法、中文名或其他语言无效。
4. 图形类型需填写英文名的大驼峰命名，其他命名法、中文名或其他语言无效。

## 目前支持的图形名称

- `RegularPolyhedron`（正多面体）：
  
  | 图形名称       | 中文名     |
  |----------------|------------|
  | `tetrahedron`  | 正四面体   |
  | `hexahedron`   | 正六面体   |
  | `octahedron`   | 正八面体   |
  | `dodecahedron` | 正十二面体 |
  | `icosahedron`  | 正二十面体 |

- `RegularPolychoron`（正多胞体）：
  
  | 图形名称             | 中文名         |
  |----------------------|----------------|
  | `pentachoron`        | 正五胞体       |
  | `tesseract`          | 正八胞体       |
  | `hexadecachoron`     | 正十六胞体     |
  | `icositetrachoron`   | 正二十四胞体   |
  | `hecatonicosachoron` | 正一百二十胞体 |
  | `hexacosichoron`     | 正六百胞体     |

- `RegularStarPolyhedron`（星状正多面体）：
  
  | 图形名称                        | 中文名           |
  |---------------------------------|------------------|
  | `great_dodecahedron`            | 大正十二面体     |
  | `small_stellated_dodecahedron`  | 小星状正十二面体 |
  | `great_stellated_dodecahedron`  | 大星状正十二面体 |
  | `great_icosahedron`             | 大正二十面体     |

- `RegularStarPolychora`（星状正多胞体）：
  
  | 图形名称                                   | 中文名                 |
  |--------------------------------------------|------------------------|
  | `great_hecatonicosachoron`                 | 大正一百二十胞体       |
  | `grand_hecatonicosachoron`                 | 巨正一百二十胞体       |
  | `great_grand_hecatonicosachoron`           | 巨大正一百二十胞体     |
  | `small_stellated_hecatonicosachoron`       | 小星状正一百二十胞体   |
  | `great_stellated_hecatonicosachoron`       | 大星状正一百二十胞体   |
  | `grand_stellated_hecatonicosachoron`       | 巨星状正一百二十胞体   |
  | `great_grand_stellated_hecatonicosachoron` | 巨大星状正一百二十胞体 |
  | `faceted_hexacosichoron`                   | 刻面正六百胞体         |
  | `great_faceted_hexacosichoron`             | 大刻面正六百胞体       |
  | `grand_hexacosichoron`                     | 巨正六百胞体           |
