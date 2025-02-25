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

1. **图形配置（`graphs`）**：定义图形的类型和参数。
2. **视频配置（`video`）**：定义视频输出的参数。
3. **绘图配置（`drawing`）**：定义图形的绘制参数。
4. **初始变换（`initial`）**：定义初始的平移和旋转。
5. **其他动作（`actions`）**：定义后续的动画动作。

### 字段说明

#### 1. 顶点配置（`graphs`）

`graphs` 是一个列表，每个图形是一个字典。

| 字段名       | 类型     | 必填 | 描述                                                                                                                                                                                                      |
|--------------|----------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `type`       | 字符串   | 是   | 图形类型，支持以下值：`RegularPolyhedron`, `RegularPolychoron`, `RegularStarPolyhedron`, `RegularStarPolychora`, `RegularPolygon`, `RegularStarPolygon`, `Simplex`, `Hypercube`, `Orthoplex`, `OffFile`。 |
| `name`       | 图形名称 | 条件 | 当 `type` 为 `RegularPolyhedron`, `RegularPolychoron`, `RegularStarPolyhedron`, `RegularStarPolychora` 时必填，表示图形的名称。关于目前支持的图形名称，参见[目前支持的图形名称](#目前支持的图形名称)。    |
| `edge_count` | 整数     | 条件 | 当 `type` 为 `RegularPolygon` 或 `RegularStarPolygon` 时必填，表示边数。                                                                                                                                  |
| `gap`        | 整数     | 条件 | 当 `type` 为 `RegularStarPolygon` 时必填，表示星形多边形的间隔。                                                                                                                                          |
| `dimensions` | 整数     | 条件 | 当 `type` 为 `Simplex`, `Hypercube`, `Orthoplex` 时必填，表示图形的维度。                                                                                                                                 |
| `path`       | 文件路径 | 条件 | 当 `type` 为 `OffFile` 时必填，表示 OFF 文件的路径。                                                                                                                                                      |
| `id`         | 字符串   | 是   | 图形的 `id`，不可重复，否则后声明的图形会覆盖前面的。                                                                                                                                                     |

#### 2. 视频配置（`video`）

| 字段名             | 类型   | 必填 | 描述                                              |
|--------------------|--------|------|---------------------------------------------------|
| `output_path`      | 字符串 | 否   | 视频输出路径，默认为当前目录下的 `rotation.mp4`。 |
| `fps`              | 整数   | 否   | 视频帧率，默认为 `30`。                           |
| `width`            | 整数   | 否   | 视频宽度，默认为 `1920`。                         |
| `height`           | 整数   | 否   | 视频高度，默认为 `1080`。                         |
| `end_pause_frames` | 整数   | 否   | 未尾的停顿帧数，默认为 `fps` 参数的两倍，即两秒。 |

#### 3. 绘图配置（`drawing`）

| 字段名            | 类型     | 必填 | 描述                                                |
|-------------------|----------|------|-----------------------------------------------------|
| `scale`           | 浮点     | 否   | 图形的缩放比例，默认为 `300`。                      |
| `focal_length`    | 浮点     | 否   | 焦距，默认为 `5`。                                  |
| `line_width`      | 浮点     | 否   | 线条宽度，默认为 `1`。                              |
| `line_color`      | 整数列表 | 否   | 线条颜色（RGB），默认为 `[0, 0, 0]`（黑色）。       |
| `background_color`| 整数列表 | 否   | 背景颜色（RGB），默认为 `[255, 255, 255]`（白色）。 |

#### 4. 初始变换（`initial`）

`initial` 是一个列表，每个初始变换是一个字典。

| 字段名          | 类型             | 必填 | 描述                                       |
|-----------------|------------------|------|--------------------------------------------|
| `offset`        | 浮点列表         | 否   | 初始平移的偏移量，例如 `[1.0, 2.0, 3.0]`。 |
| `rotations`     | 字典列表         | 否   | 初始旋转动作列表，每个字典包含以下字段：   |
| - `plane`       | 整数列表（两项） | 是   | 旋转平面，例如 `[0, 1]`。                  |
| - `angle`       | 浮点             | 是   | 旋转角度（单位：度）。                     |
| - `center`      | 浮点列表         | 否   | 旋转中心点，例如 `[2.0, 0.5, 0.0]`。       |
| - `priority`    | 整数             | 否   | 该旋转的优先级。                           |
| `move_priority` | 整数             | 否   | 平移的优先级。                             |
| `target`        | 图形 `id`        | 是   | 该初始变换所作用的图形的 `id`。            |

#### 5. 其他动作（`actions`）

`actions` 是一个列表，每个动作是一个字典。

每种动作类型都有以下**必填**字段：

| 字段名   | 类型      | 描述                                   |
|----------|-----------|----------------------------------------|
| `type`   | 字符串    | 动作类型。                             |
| `start`  | 整数      | 动作的开始时间（单位：帧）。           |
| `target` | 图形 `id` | 该动作所作用的图形的 `id`。            |

支持以下动作类型：

##### 5.1 平移动作（`move`）

| 字段名     | 类型     | 必填 | 描述                                   |
|------------|----------|------|----------------------------------------|
| `offset`   | 浮点列表 | 是   | 平移的偏移量，例如 `[1.0, 2.0, 3.0]`。 |
| `duration` | 整数     | 是   | 动作持续时间（单位：帧）。             |
| `priority` | 整数     | 否   | 该平移的优先级。                       |

##### 5.2 旋转动作（`rotate`）

| 字段名     | 类型             | 必填 | 描述                                 |
|------------|------------------|------|--------------------------------------|
| `plane`    | 整数列表（两项） | 是   | 旋转平面，例如 `[1, 2]`。            |
| `angle`    | 浮点             | 是   | 旋转角度（单位：度）。               |
| `duration` | 整数             | 是   | 旋转持续时间（单位：帧）。           |
| `center`   | 浮点列表         | 否   | 旋转中心点，例如 `[2.0, 0.5, 0.0]`。 |
| `priority` | 整数             | 否   | 该旋转的优先级。                     |

##### 5.3 复合旋转动作（`rotate_complexly`）

这实际上等价于多个旋转动作的简写。

| 字段名       | 类型             | 必填 | 描述                                 |
|--------------|------------------|------|--------------------------------------|
| `rotations`  | 字典列表         | 是   | 旋转动作列表，每个字典包含以下字段： |
| - `plane`    | 整数列表（两项） | 是   | 旋转平面，例如 `[1, 2]`。            |
| - `angle`    | 浮点             | 是   | 旋转角度（单位：度）。               |
| - `duration` | 整数             | 是   | 旋转持续时间（单位：帧）。           |
| - `center`   | 浮点列表         | 否   | 旋转中心点，例如 `[2.0, 0.5, 0.0]`。 |
| - `priority` | 整数             | 否   | 该旋转的优先级。                     |

> [!NOTE]
>
> 优先级默认为 `0`。
>
> 当同一优先级上同时有平移和旋转动作时，为先旋转后平移。
>
> 平移和旋转是累加的，因此一般把旋转中心设为 `(0, 0, 0, ...)`（**也就是默认值**）即可表示图形的中心。
>
> 图形类型需填写英文名的大驼峰命名，图形名称需填写英文名的小写下划线命名，其他命名法、中文名或其他语言无效。

### 示例配置文件

```toml
[[graphs]]
type = "Hypercube"
dimensions = 4
id = "g1"

[[graphs]]
type = "RegularPolychoron"
name = "tesseract"
id = "g2"

[video]
fps = 30
width = 1920
height = 1080
end_pause_frames = 60

[drawing]
scale = 120
line_width = 5
line_color = [230, 230, 230]
background_color = [20, 20, 20]
focal_length = 7

[[initial]]
rotations = [
    { plane = [0, 2], angle = 45, priority = 1 }
]
offset = [1, 1, 1, 1]
target = "g1"

[[initial]]
rotations = [
    { plane = [0, 2], angle = 45 }
]
offset = [-2, 0, 0, -2]
target = "g2"

[[actions]]
type = "move"
offset = [-3.0, -1.0, -1.0, -3.0]
duration = 150
start = 0
target = "g1"

[[actions]]
type = "move"
offset = [8.0, 1.0, 1.0, 3.0]
duration = 150
start = 0
target = "g2"

[[actions]]
type = "rotate"
plane = [0, 2]
angle = 360
center = [2, 0, 0, 0]
duration = 100
start = 150
target = "g1"

[[actions]]
type = "rotate"
plane = [0, 2]
angle = 360
center = [6, 1, 1, 1]
duration = 100
start = 200
priority = -1
target = "g2"

[[actions]]
type = "rotate_complexly"
rotations = [
    { plane = [0, 2], angle = -45, duration = 200 },
    { plane = [2, 3], angle = 360, duration = 200 }
]
start = 250
target = "g1"

[[actions]]
type = "rotate_complexly"
rotations = [
    { plane = [0, 2], angle = -45, duration = 200 },
    { plane = [2, 3], angle = 270, duration = 200 },
    { plane = [1, 2], angle = 270, duration = 200, center = [0, -3, 0, 0], priority = 1 }
]
start = 300
target = "g2"

[[actions]]
type = "move"
offset = [2.0, 0.0, 0.0, 2.0]
duration = 150
start = 500
target = "g1"

[[actions]]
type = "move"
offset = [-6, -1, -1, -1]
duration = 150
start = 500
target = "g2"

[[actions]]
type = "rotate"
plane = [1, 2]
angle = 90
center = [0, -3, 0, 0]
duration = 150
start = 500
target = "g2"
priority = 1
```

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
