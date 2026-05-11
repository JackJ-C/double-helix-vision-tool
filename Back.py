"""
Back.py — Double Helix Reflection Engine (反射引擎)

核心理念 (What.png 所示):
  正向 (Forward):   2D 视界 → 双螺旋采样 → 1D 奇异点 (Singularity)
  反向 (Reflection): 1D 奇异点 → 逆双螺旋膨胀 → 3D 空间重建

仿生原理:
  Alpha 和 Beta 两条螺旋线相当于两只"眼睛"。
  它们在同一深度层采样了不同的空间位置。
  利用这种天然的视差 (Disparity)，从单目摄像头中提取立体深度信息。
  从奇异点(焦点)反向沿双螺旋路径向外扩散，铺开扫描整个周围空间。

API 接口:
  DHVisionAPI 类提供标准化的 JSON 接口，供 LLM / 机器人控制系统调用。
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any
import json


# ============================================================
# 第一层：正向扫描引擎 (复制自 DH-vision.py)
# ============================================================

class DoubleHelixVisionTool:
    def __init__(self, width=1920, height=1080):
        """
        初始化工具：确立视界尺寸
        """
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2

        # 四角定位 (Four-Corner Positioning)
        self.corners = np.array([
            [0, 0],           # 左上 (TL)
            [width, 0],       # 右上 (TR)
            [width, height],  # 右下 (BR)
            [0, height]       # 左下 (BL)
        ])

        # 预计算双螺旋路径 (Double Helix Rifling)
        self._init_spiral_path()

    def _init_spiral_path(self):
        """
        生成黄金螺旋膛线路径
        """
        num_points = 3000
        rotations = 8
        growth_factor = 0.55

        max_radius = min(self.width, self.height) / 1.8

        thetas = np.linspace(0, rotations * 2 * np.pi, num_points)
        radii = max_radius * (thetas / (thetas[-1] + 1e-9)) ** growth_factor

        # --- 螺旋 Alpha (顺时针) ---
        self.x_a = (self.center_x + radii * np.cos(thetas)).astype(int)
        self.y_a = (self.center_y + radii * np.sin(thetas)).astype(int)

        # --- 螺旋 Beta (逆时针/相位差 180度) ---
        self.x_b = (self.center_x + radii * np.cos(thetas + np.pi)).astype(int)
        self.y_b = (self.center_y + radii * np.sin(thetas + np.pi)).astype(int)

        self.depth_map = radii

        self.valid_mask = (
            (self.x_a >= 0) & (self.x_a < self.width) &
            (self.y_a >= 0) & (self.y_a < self.height) &
            (self.x_b >= 0) & (self.x_b < self.width) &
            (self.y_b >= 0) & (self.y_b < self.height)
        )

        self.idx_xa = self.x_a[self.valid_mask]
        self.idx_ya = self.y_a[self.valid_mask]
        self.idx_xb = self.x_b[self.valid_mask]
        self.idx_yb = self.y_b[self.valid_mask]
        self.valid_depth = self.depth_map[self.valid_mask]

    def scan(self, frame_path_or_array):
        """
        核心功能：执行一次视觉扫描
        """
        if isinstance(frame_path_or_array, str):
            frame = cv2.imread(frame_path_or_array)
            if frame is None:
                raise ValueError("无法读取图片路径")
        else:
            frame = frame_path_or_array

        if (frame.shape[1] != self.width) or (frame.shape[0] != self.height):
            frame = cv2.resize(frame, (self.width, self.height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        signal_alpha = gray[self.idx_ya, self.idx_xa]
        signal_beta  = gray[self.idx_yb, self.idx_xb]

        return {
            "alpha_stream": signal_alpha,
            "beta_stream":  signal_beta,
            "depth_stream": self.valid_depth,
            "original_img": frame
        }

    def visualize(self, scan_result):
        """
        辅助功能：可视化双螺旋采样效果
        """
        frame = scan_result['original_img']
        disp_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(disp_img)
        plt.scatter(self.idx_xa, self.idx_ya, c='lime', s=0.5, alpha=0.5, label='Helix A')
        plt.scatter(self.idx_xb, self.idx_yb, c='cyan', s=0.5, alpha=0.5, label='Helix B')
        pts = self.corners.reshape((-1, 1, 2))
        cv2.polylines(disp_img, [pts], True, (255, 0, 0), 5)
        plt.title("Double Helix Project: Cone-Slice Geometry")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.plot(scan_result['alpha_stream'][::-1], color='lime', alpha=0.8, linewidth=1, label='Alpha')
        plt.plot(scan_result['beta_stream'][::-1], color='cyan', alpha=0.6, linewidth=1, label='Beta')
        plt.title("Collapsed Depth Signal (Center -> Edge)")
        plt.xlabel("Sampling Steps (Time/Depth)")
        plt.ylabel("Intensity")
        plt.legend()

        plt.tight_layout()
        plt.show()


# ============================================================
# 第二层：反射引擎 (Reflection Engine)
# 从 1D 奇异点反向回归，沿双螺旋路径膨胀回 2D/3D
# ============================================================

class ReflectionEngine:
    """
    反射引擎：实现从 Singularity 到 3D 空间的逆向重建。

    工作流程:
      1. inverse_scan:     1D信号 → 2D画布 (逆螺旋喷射 + 扩散填充)
      2. temporal_collision: 连续帧1D碰撞 → 运动/深度边缘
      3. stereo_depth:      Alpha vs Beta 视差 → 立体深度层
    """

    # 深度分层数 (将连续深度离散化为多少个层级)
    NUM_DEPTH_LAYERS = 32

    def __init__(self, dh_tool: DoubleHelixVisionTool):
        self.dh = dh_tool
        self.prev_scan: Optional[Dict] = None

        # 预计算深度分层索引
        max_d = self.dh.valid_depth.max() + 1e-9
        bins = np.linspace(0, max_d, self.NUM_DEPTH_LAYERS + 1)
        self.depth_bins = bins
        self.layer_indices = np.clip(
            np.digitize(self.dh.valid_depth, bins) - 1,
            0, self.NUM_DEPTH_LAYERS - 1
        )

    # ----------------------------------------------------------
    # 1. 逆向扫描 (1D → 2D)
    # ----------------------------------------------------------
    def inverse_scan(self, scan_result: Dict, diffuse_ksize: int = 15) -> Dict:
        """
        从 1D 奇异点信号，沿双螺旋路径反向膨胀回 2D 画布。

        过程:
          - 在全黑画布上，按螺旋坐标"喷射"1D亮度值
          - 对未采样区域做高斯扩散填充 (模拟光从焦点向外扩散)
          - 合并 Alpha/Beta 双眼视界

        Args:
            scan_result: scan() 的返回值
            diffuse_ksize: 高斯扩散核大小 (奇数)

        Returns:
            alpha_2d, beta_2d: 各自的2D重建
            merged_2d: 双眼合并视界
        """
        alpha = scan_result['alpha_stream'].astype(np.float32)
        beta  = scan_result['beta_stream'].astype(np.float32)
        h, w  = self.dh.height, self.dh.width

        # 创建全黑画布，沿螺旋坐标喷射
        canvas_a = np.zeros((h, w), dtype=np.float32)
        canvas_b = np.zeros((h, w), dtype=np.float32)
        canvas_a[self.dh.idx_ya, self.dh.idx_xa] = alpha
        canvas_b[self.dh.idx_yb, self.dh.idx_xb] = beta

        # 高斯扩散：模拟光从奇异点向外的自然扩散
        k = diffuse_ksize if diffuse_ksize % 2 == 1 else diffuse_ksize + 1
        filled_a = cv2.GaussianBlur(canvas_a, (k, k), 0)
        filled_b = cv2.GaussianBlur(canvas_b, (k, k), 0)
        merged   = (filled_a + filled_b) / 2.0

        return {
            'alpha_2d':  filled_a,
            'beta_2d':   filled_b,
            'merged_2d': merged,
            'raw_alpha': canvas_a,
            'raw_beta':  canvas_b,
        }

    # ----------------------------------------------------------
    # 2. 时序碰撞 (帧差 → 运动深度)
    # ----------------------------------------------------------
    def temporal_collision(self, current_scan: Dict) -> Dict:
        """
        连续帧的 1D 信号碰撞，提取运动边缘和深度轮廓。

        数学: collision = |signal_T - signal_{T-1}|
        物理: 峰值处 = 有物体运动 / 有深度跳变的空间边缘
        """
        prev = self.prev_scan
        self.prev_scan = current_scan

        if prev is None:
            n = len(current_scan['alpha_stream'])
            return {
                'collision_alpha': np.zeros(n, dtype=np.float32),
                'collision_beta':  np.zeros(n, dtype=np.float32),
                'depth_edges':     np.zeros(n, dtype=np.float32),
                'has_data': False,
            }

        ca = np.abs(current_scan['alpha_stream'].astype(np.float32)
                     - prev['alpha_stream'].astype(np.float32))
        cb = np.abs(current_scan['beta_stream'].astype(np.float32)
                     - prev['beta_stream'].astype(np.float32))

        # 深度加权: 近处权重大 (边缘), 远处权重小 (中心)
        max_d = self.dh.valid_depth.max() + 1e-9
        weight = 1.0 - (self.dh.valid_depth / max_d)
        depth_edges = (ca + cb) * weight / 2.0

        return {
            'collision_alpha': ca,
            'collision_beta':  cb,
            'depth_edges':     depth_edges,
            'has_data': True,
        }

    # ----------------------------------------------------------
    # 3. 立体深度 (Alpha-Beta 视差)
    # ----------------------------------------------------------
    def stereo_depth(self, scan_result: Dict) -> Dict:
        """
        利用 Alpha/Beta 双螺旋的天然视差估算立体深度。

        原理:
          两条螺旋在同一深度层采样了 180° 对称的空间位置。
          同一层上亮度差异大 → 该深度存在显著的空间结构(物体边界)。
          差异小 → 该深度层为均匀区域(空旷/天空)。
        """
        alpha = scan_result['alpha_stream'].astype(np.float32)
        beta  = scan_result['beta_stream'].astype(np.float32)
        n_layers = self.NUM_DEPTH_LAYERS

        disparity = np.zeros(n_layers, dtype=np.float32)
        energy    = np.zeros(n_layers, dtype=np.float32)

        for i in range(n_layers):
            mask = self.layer_indices == i
            if mask.sum() == 0:
                continue
            a_l, b_l = alpha[mask], beta[mask]
            disparity[i] = np.mean(np.abs(a_l - b_l))
            energy[i]    = np.mean(a_l + b_l) / 2.0

        # 深度置信度: 视差高 + 能量高 = 近处有物体
        confidence = disparity * energy / (energy.max() + 1e-9)

        return {
            'layer_disparity':  disparity,
            'layer_energy':     energy,
            'depth_confidence': confidence,
            'num_layers':       n_layers,
        }


# ============================================================
# 第三层：3D 空间映射器 (Spatial Mapper)
# 将 2D 螺旋坐标 + 深度 → 3D 视锥体点云
# ============================================================

class SpatialMapper:
    """
    空间映射器：模拟人眼的 3D 视锥体。

    参数:
      fov_horizontal:  水平视场角 (度)
      max_depth_meters: 最大探测深度 (米)

    输出:
      每个双螺旋采样点对应的 (X, Y, Z) 三维坐标，
      其中 Z 轴由螺旋半径映射而来 (中心=远, 边缘=近)。
    """

    def __init__(self, dh_tool: DoubleHelixVisionTool,
                 fov_horizontal: float = 60.0,
                 max_depth_meters: float = 10.0):
        self.dh = dh_tool
        self.fov_h = np.radians(fov_horizontal)
        self.fov_v = self.fov_h * (dh_tool.height / dh_tool.width)
        self.max_depth = max_depth_meters
        self._build_projection()

    def _build_projection(self):
        """
        预计算: 2D 螺旋像素坐标 → 3D 空间坐标 (针孔相机模型)
        """
        # 归一化到 [-1, 1]
        nx_a = (self.dh.idx_xa - self.dh.center_x) / (self.dh.width / 2.0)
        ny_a = (self.dh.idx_ya - self.dh.center_y) / (self.dh.height / 2.0)
        nx_b = (self.dh.idx_xb - self.dh.center_x) / (self.dh.width / 2.0)
        ny_b = (self.dh.idx_yb - self.dh.center_y) / (self.dh.height / 2.0)

        # 半径 → 物理深度  (中心=远=max_depth, 边缘=近=0)
        max_r = self.dh.valid_depth.max() + 1e-9
        z = self.max_depth * (1.0 - self.dh.valid_depth / max_r)

        tan_h = np.tan(self.fov_h / 2)
        tan_v = np.tan(self.fov_v / 2)

        self.pts_a = np.column_stack([nx_a * z * tan_h,
                                      ny_a * z * tan_v,
                                      z])
        self.pts_b = np.column_stack([nx_b * z * tan_h,
                                      ny_b * z * tan_v,
                                      z])

    def map_to_3d(self, scan_result: Dict) -> Dict:
        """
        将一帧扫描结果映射为带亮度的 3D 点云。
        """
        ia = scan_result['alpha_stream'].astype(np.float32) / 255.0
        ib = scan_result['beta_stream'].astype(np.float32) / 255.0

        merged_pts = np.vstack([self.pts_a, self.pts_b])
        merged_int = np.concatenate([ia, ib])

        return {
            'cloud_alpha':  {'points': self.pts_a, 'intensity': ia},
            'cloud_beta':   {'points': self.pts_b, 'intensity': ib},
            'merged_cloud': {'points': merged_pts, 'intensity': merged_int},
            'bounds': {
                'x': [float(merged_pts[:, 0].min()), float(merged_pts[:, 0].max())],
                'y': [float(merged_pts[:, 1].min()), float(merged_pts[:, 1].max())],
                'z': [float(merged_pts[:, 2].min()), float(merged_pts[:, 2].max())],
            }
        }


# ============================================================
# 第四层：API 接口 (供 LLM / 机器人调用)
# ============================================================

class DHVisionAPI:
    """
    Double Helix 视觉系统 — 标准化 API

    所有 process_frame() 返回值均为 JSON 可序列化格式，
    可直接供 LLM Agent 解析和决策。

    使用示例:
        api = DHVisionAPI(width=1920, height=1080)
        report = api.process_frame(cv2.imread('test.jpg'))
        print(api.to_json(report))
    """

    def __init__(self, width: int = 1920, height: int = 1080,
                 fov: float = 60.0, max_depth: float = 10.0):
        self.dh = DoubleHelixVisionTool(width, height)
        self.reflection = ReflectionEngine(self.dh)
        self.spatial = SpatialMapper(self.dh, fov, max_depth)
        self._frame_id = 0

    def process_frame(self, frame) -> Dict[str, Any]:
        """
        处理单帧，返回完整空间感知报告 (JSON-ready)。

        Args:
            frame: BGR numpy array 或 图片文件路径

        Returns:
            dict — 包含 signal / depth / motion / spatial 四大板块
        """
        scan      = self.dh.scan(frame)
        collision = self.reflection.temporal_collision(scan)
        stereo    = self.reflection.stereo_depth(scan)
        spatial   = self.spatial.map_to_3d(scan)
        self._frame_id += 1

        return {
            'frame_id': self._frame_id,
            'signal': {
                'alpha_mean': float(np.mean(scan['alpha_stream'])),
                'beta_mean':  float(np.mean(scan['beta_stream'])),
                'alpha_std':  float(np.std(scan['alpha_stream'])),
                'beta_std':   float(np.std(scan['beta_stream'])),
            },
            'depth': {
                'layer_disparity':  stereo['layer_disparity'].tolist(),
                'layer_energy':     stereo['layer_energy'].tolist(),
                'depth_confidence': stereo['depth_confidence'].tolist(),
                'num_layers':       stereo['num_layers'],
            },
            'motion': {
                'detected':          collision['has_data'],
                'intensity':         float(np.mean(collision['depth_edges'])),
                'peak_depth_index':  int(np.argmax(collision['depth_edges'])),
            },
            'spatial': {
                'bounds':         spatial['bounds'],
                'total_points':   len(spatial['merged_cloud']['points']),
                'mean_intensity': float(np.mean(spatial['merged_cloud']['intensity'])),
            },
        }

    def get_raw(self, frame) -> Dict:
        """
        获取全部原始 numpy 数据 (供内部模块 / 可视化使用，不可JSON序列化)。
        """
        scan       = self.dh.scan(frame)
        collision  = self.reflection.temporal_collision(scan)
        stereo     = self.reflection.stereo_depth(scan)
        spatial    = self.spatial.map_to_3d(scan)
        reflection = self.reflection.inverse_scan(scan)
        return {
            'scan': scan, 'collision': collision, 'stereo': stereo,
            'spatial': spatial, 'reflection': reflection,
        }

    def to_json(self, report: Dict) -> str:
        """将 process_frame 的返回值序列化为 JSON 字符串。"""
        return json.dumps(report, indent=2, ensure_ascii=False)


# ============================================================
# 可视化辅助
# ============================================================

def visualize_reflection(api: DHVisionAPI, frame):
    """
    四面板可视化:
      左上: 原图 + 双螺旋轨迹
      右上: 反射重建 (merged_2d)
      左下: Alpha-Beta 立体深度层视差
      右下: 时序碰撞深度边缘 (需要连续帧)
    """
    raw = api.get_raw(frame)
    scan = raw['scan']
    ref  = raw['reflection']
    st   = raw['stereo']
    col  = raw['collision']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 左上: 原图 + 螺旋
    ax = axes[0, 0]
    disp = cv2.cvtColor(scan['original_img'], cv2.COLOR_BGR2RGB)
    ax.imshow(disp)
    ax.scatter(api.dh.idx_xa, api.dh.idx_ya, c='lime', s=0.3, alpha=0.4)
    ax.scatter(api.dh.idx_xb, api.dh.idx_yb, c='cyan', s=0.3, alpha=0.4)
    ax.set_title("Forward: 2D → 1D (正向坍缩)")
    ax.axis('off')

    # 右上: 反射重建
    ax = axes[0, 1]
    ax.imshow(ref['merged_2d'], cmap='inferno')
    ax.set_title("Reflection: 1D → 2D (逆向膨胀)")
    ax.axis('off')

    # 左下: 立体深度层
    ax = axes[1, 0]
    x = np.arange(st['num_layers'])
    ax.bar(x, st['depth_confidence'], color='cyan', alpha=0.7, label='Confidence')
    ax.plot(x, st['layer_disparity'], color='lime', linewidth=2, label='Disparity')
    ax.set_title("Stereo Depth (立体深度层)")
    ax.set_xlabel("Depth Layer (远 → 近)")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)

    # 右下: 碰撞波形
    ax = axes[1, 1]
    if col['has_data']:
        ax.plot(col['depth_edges'][::-1], color='orange', linewidth=1)
        ax.set_title("Temporal Collision (时序碰撞)")
    else:
        ax.text(0.5, 0.5, "需要连续帧\n(第二帧起生效)",
                ha='center', va='center', fontsize=14, color='gray',
                transform=ax.transAxes)
        ax.set_title("Temporal Collision (等待中...)")
    ax.set_xlabel("Depth Index")
    ax.set_ylabel("Edge Intensity")

    plt.suptitle("Double Helix Reflection Engine", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Double Helix Reflection Engine (Back.py)")
    print("  反射引擎已加载。")
    print("=" * 55)
    print()
    print("  可用模块:")
    print("    DoubleHelixVisionTool  — 正向扫描 (2D→1D)")
    print("    ReflectionEngine       — 逆向反射 (1D→2D/3D)")
    print("    SpatialMapper          — 3D 视锥体映射")
    print("    DHVisionAPI            — LLM/机器人 API 接口")
    print()

    # --- 快速演示 ---
    api = DHVisionAPI(width=1920, height=1080, fov=60.0, max_depth=10.0)
    frame = cv2.imread('IMG_6033.PNG')
    report = api.process_frame(frame)
    print(api.to_json(report))
    visualize_reflection(api, frame)

    print("  使用示例:")
    print("    api = DHVisionAPI()")
    print("    report = api.process_frame(cv2.imread('your_image.jpg'))")
    print("    print(api.to_json(report))")
