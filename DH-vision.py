import numpy as np
import cv2
import matplotlib.pyplot as plt

class DoubleHelixVisionTool:
    def __init__(self, width=1920, height=1080):
        """
        初始化工具：确立视界尺寸
        """
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # 1. 四角定位 (Four-Corner Positioning)
        # 定义视锥体的最外层切片 (近景平面 Near Plane)
        self.corners = np.array([
            [0, 0],           # 左上 (TL)
            [width, 0],       # 右上 (TR)
            [width, height],  # 右下 (BR)
            [0, height]       # 左下 (BL)
        ])
        
        # 2. 预计算双螺旋路径 (Double Helix Rifling)
        # 只需要计算一次，极大节省后续处理算力
        self._init_spiral_path()

    def _init_spiral_path(self):
        """
        生成黄金螺旋膛线路径
        """
        # 参数设定：决定采样密度和视锥角度
        num_points = 3000   # 采样点数
        rotations = 8       # 旋转圈数
        growth_factor = 0.55 # 决定螺旋张开的速度 (模拟透视)
        
        # 最大的采样半径 (也就是从中心到四角的距离)
        max_radius = min(self.width, self.height) / 1.8
        
        # 生成角度 (Theta)
        thetas = np.linspace(0, rotations * 2 * np.pi, num_points)
        
        # 生成半径 (Radius) - 基于指数增长，模拟中心密集(远)、边缘稀疏(近)
        # 加上 1e-9 防止除零
        radii = max_radius * (thetas / (thetas[-1] + 1e-9)) ** growth_factor
        
        # --- 螺旋 Alpha (顺时针) ---
        self.x_a = (self.center_x + radii * np.cos(thetas)).astype(int)
        self.y_a = (self.center_y + radii * np.sin(thetas)).astype(int)
        
        # --- 螺旋 Beta (逆时针/相位差 180度) ---
        # 通过增加 Pi 实现双螺旋纠缠
        self.x_b = (self.center_x + radii * np.cos(thetas + np.pi)).astype(int)
        self.y_b = (self.center_y + radii * np.sin(thetas + np.pi)).astype(int)
        
        # 保存半径数据用于深感计算 (Radius 越小 = 离中心越近 = 距离越远)
        self.depth_map = radii 
        
        # 创建有效点掩码 (过滤掉超出四角范围的点)
        self.valid_mask = (
            (self.x_a >= 0) & (self.x_a < self.width) & 
            (self.y_a >= 0) & (self.y_a < self.height) &
            (self.x_b >= 0) & (self.x_b < self.width) & 
            (self.y_b >= 0) & (self.y_b < self.height)
        )
        
        # 提取有效坐标，供后续快速索引
        self.idx_xa = self.x_a[self.valid_mask]
        self.idx_ya = self.y_a[self.valid_mask]
        self.idx_xb = self.x_b[self.valid_mask]
        self.idx_yb = self.y_b[self.valid_mask]
        self.valid_depth = self.depth_map[self.valid_mask]

    def scan(self, frame_path_or_array):
        """
        核心功能：执行一次视觉扫描
        """
        # 输入处理：支持文件路径或直接传入数组
        if isinstance(frame_path_or_array, str):
            frame = cv2.imread(frame_path_or_array)
            if frame is None:
                raise ValueError("无法读取图片路径")
        else:
            frame = frame_path_or_array

        # 强制缩放到初始化尺寸 (保持四角定位准确)
        if (frame.shape[1] != self.width) or (frame.shape[0] != self.height):
            frame = cv2.resize(frame, (self.width, self.height))

        # 转灰度 (只取亮度信息，极大提升速度)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- 极速采样 (Rifling Sampling) ---
        # 利用 Numpy 高级索引，瞬间提取几千个点
        signal_alpha = gray[self.idx_ya, self.idx_xa]
        signal_beta  = gray[self.idx_yb, self.idx_xb]
        
        # 返回结构化数据流
        return {
            "alpha_stream": signal_alpha, # 螺旋A的亮度流
            "beta_stream":  signal_beta,  # 螺旋B的亮度流
            "depth_stream": self.valid_depth, # 对应的视锥半径 (可换算为Z轴深度)
            "original_img": frame # 返回原图用于可视化
        }

    def visualize(self, scan_result):
        """
        辅助功能：可视化双螺旋采样效果
        """
        frame = scan_result['original_img']
        # 转换为RGB用于显示
        disp_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        
        # 左图：几何视界
        plt.subplot(1, 2, 1)
        plt.imshow(disp_img)
        plt.scatter(self.idx_xa, self.idx_ya, c='lime', s=0.5, alpha=0.5, label='Helix A')
        plt.scatter(self.idx_xb, self.idx_yb, c='cyan', s=0.5, alpha=0.5, label='Helix B')
        # 画出四角定位框
        pts = self.corners.reshape((-1, 1, 2))
        cv2.polylines(disp_img, [pts], True, (255, 0, 0), 5) # 这一步在plt里看不见，示意逻辑
        plt.title("Double Helix Project: Cone-Slice Geometry")
        plt.axis('off')
        
        # 右图：塌陷数据流 (1D Signal)
        plt.subplot(1, 2, 2)
        # X轴反转：从中心(远) -> 边缘(近)
        plt.plot(scan_result['alpha_stream'][::-1], color='lime', alpha=0.8, linewidth=1, label='Alpha')
        plt.plot(scan_result['beta_stream'][::-1], color='cyan', alpha=0.6, linewidth=1, label='Beta')
        plt.title("Collapsed Depth Signal (Center -> Edge)")
        plt.xlabel("Sampling Steps (Time/Depth)")
        plt.ylabel("Intensity")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 实例化工具 (定义你的视界大小)
    tool = DoubleHelixVisionTool(width=1920, height=1080)
    
    # 2. 扫描图片 (替换为你的图片路径)
    # img_path = 'your_dragon_image.jpg' 
    # result = tool.scan(img_path)
    
    # 3. 可视化结果
    # tool.visualize(result)
    
    print("Double Helix 视觉几何工具已加载。随时待命。")
