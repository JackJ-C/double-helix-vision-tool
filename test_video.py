"""
test_video.py — 双螺旋反射引擎 · 视频测试脚本

测试流程:
  1. 逐帧读取视频
  2. 对每帧执行正向扫描 + 时序碰撞 + 立体深度
  3. 将碰撞强度和深度置信度随时间的变化记录下来
  4. 输出:
     - 终端实时打印每帧的运动强度
     - 最终生成一张汇总分析图 (保存为 result_<视频名>.png)
     - 抽取关键帧生成4面板可视化
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 无头模式，直接保存图片
import matplotlib.pyplot as plt
import time
import os

from Back import DHVisionAPI, ReflectionEngine, visualize_reflection


def test_video(video_path: str, sample_interval: int = 10):
    """
    对一个视频文件执行完整的反射引擎测试。

    Args:
        video_path:      视频文件路径
        sample_interval: 每隔多少帧抽取一次关键帧做详细可视化
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*60}")
    print(f"  测试视频: {video_name}")
    print(f"{'='*60}")

    # --- 打开视频 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ 无法打开视频: {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  分辨率: {w}x{h}  |  帧率: {fps:.0f}fps  |  总帧数: {total}")

    # --- 初始化反射引擎 ---
    api = DHVisionAPI(width=1920, height=1080, fov=60.0, max_depth=10.0)
    print(f"  引擎已初始化 (视界: 1920x1080)")
    print(f"  开始逐帧处理...\n")

    # --- 记录时序数据 ---
    frame_ids = []
    motion_intensities = []       # 每帧的碰撞运动强度
    depth_confidence_history = []  # 每帧的32层深度置信度
    stereo_disparity_history = []  # 每帧的32层视差
    keyframe_indices = []          # 关键帧的编号 (用于后续可视化)
    keyframe_collisions = []       # 关键帧的碰撞波形

    t_start = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 正向扫描 ---
        scan = api.dh.scan(frame)

        # --- 时序碰撞 ---
        collision = api.reflection.temporal_collision(scan)

        # --- 立体深度 ---
        stereo = api.reflection.stereo_depth(scan)

        # --- 记录数据 ---
        frame_ids.append(frame_idx)
        mi = float(np.mean(collision['depth_edges'])) if collision['has_data'] else 0.0
        motion_intensities.append(mi)
        depth_confidence_history.append(stereo['depth_confidence'].copy())
        stereo_disparity_history.append(stereo['layer_disparity'].copy())

        # 记录关键帧的碰撞波形
        if frame_idx % sample_interval == 0 and collision['has_data']:
            keyframe_indices.append(frame_idx)
            keyframe_collisions.append(collision['depth_edges'].copy())

        # 实时打印
        if frame_idx % 30 == 0:
            elapsed = time.time() - t_start
            speed = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  帧 {frame_idx:>4d}/{total}  |  运动强度: {mi:>8.3f}  |  速度: {speed:.1f} fps")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start
    print(f"\n  ✅ 处理完成! 共 {frame_idx} 帧, 耗时 {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")

    # ================================================================
    # 生成汇总分析图
    # ================================================================
    motion_arr = np.array(motion_intensities)
    confidence_arr = np.array(depth_confidence_history)  # (T, 32)
    disparity_arr = np.array(stereo_disparity_history)   # (T, 32)
    time_axis = np.array(frame_ids) / fps  # 转为秒

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1.2, 1.2]})
    fig.suptitle(f'Double Helix Reflection Analysis: {video_name}', fontsize=16, fontweight='bold')

    # --- 面板1: 运动强度时间线 ---
    ax = axes[0]
    ax.fill_between(time_axis, motion_arr, alpha=0.3, color='orange')
    ax.plot(time_axis, motion_arr, color='orange', linewidth=1, label='Motion Intensity')
    # 标记峰值
    if len(motion_arr) > 5:
        threshold = np.mean(motion_arr) + 2 * np.std(motion_arr)
        peaks = motion_arr > threshold
        if peaks.any():
            ax.scatter(time_axis[peaks], motion_arr[peaks], c='red', s=20, zorder=5, label=f'Peaks (>{threshold:.2f})')
    ax.set_ylabel('Collision Intensity')
    ax.set_title('Temporal Collision: Motion Intensity Over Time')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 面板2: 深度置信度热力图 (时间 x 深度层) ---
    ax = axes[1]
    im = ax.imshow(confidence_arr.T, aspect='auto', cmap='inferno',
                   extent=[time_axis[0], time_axis[-1], 31, 0],
                   interpolation='bilinear')
    ax.set_ylabel('Depth Layer\n(0=Far, 31=Near)')
    ax.set_title('Depth Confidence Heatmap (Stereo Disparity x Energy)')
    plt.colorbar(im, ax=ax, label='Confidence', shrink=0.8)

    # --- 面板3: 关键帧碰撞波形叠加 ---
    ax = axes[2]
    if keyframe_collisions:
        n_kf = len(keyframe_collisions)
        colors = plt.cm.viridis(np.linspace(0, 1, n_kf))
        for i, (ki, kc) in enumerate(zip(keyframe_indices, keyframe_collisions)):
            t_sec = ki / fps
            ax.plot(kc[::-1], color=colors[i], alpha=0.6, linewidth=0.8,
                    label=f't={t_sec:.1f}s' if i % max(1, n_kf // 6) == 0 else None)
        ax.set_title(f'Collision Waveforms at Keyframes (every {sample_interval} frames)')
        ax.legend(loc='upper right', fontsize=8, ncol=3)
    else:
        ax.text(0.5, 0.5, 'No keyframe collision data', ha='center', va='center',
                fontsize=14, color='gray', transform=ax.transAxes)
        ax.set_title('Collision Waveforms')
    ax.set_xlabel('Sampling Index (Center/Far -> Edge/Near)')
    ax.set_ylabel('Edge Intensity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f'result_{video_name}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 汇总分析图已保存: {out_path}")

    # --- 统计摘要 ---
    print(f"\n  📋 统计摘要:")
    print(f"     平均运动强度: {motion_arr.mean():.4f}")
    print(f"     最大运动强度: {motion_arr.max():.4f} (帧 {motion_arr.argmax()}, t={motion_arr.argmax()/fps:.2f}s)")
    print(f"     运动标准差:   {motion_arr.std():.4f}")
    print(f"     最活跃深度层: {confidence_arr.mean(axis=0).argmax()} / 31")

    return {
        'video': video_name,
        'motion': motion_arr,
        'confidence': confidence_arr,
        'disparity': disparity_arr,
    }


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Double Helix Reflection Engine — Video Test Suite")
    print("=" * 60)

    # 先测平滑视频，再测晃动视频
    videos = ['IMG_7011.MOV', 'IMG_7010.MOV']

    results = {}
    for v in videos:
        if os.path.exists(v):
            results[v] = test_video(v, sample_interval=15)
        else:
            print(f"\n  ⚠️  跳过 {v} (文件不存在)")

    # --- 如果两个视频都跑了，生成对比图 ---
    if len(results) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle('Motion Intensity Comparison: Smooth vs Shaky', fontsize=14, fontweight='bold')

        for i, (name, data) in enumerate(results.items()):
            ax = axes[i]
            vname = os.path.splitext(name)[0]
            m = data['motion']
            t = np.arange(len(m)) / 60.0
            ax.fill_between(t, m, alpha=0.3, color=['cyan', 'orange'][i])
            ax.plot(t, m, color=['cyan', 'orange'][i], linewidth=1)
            ax.set_title(f'{vname}\nmean={m.mean():.3f}, max={m.max():.3f}, std={m.std():.3f}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Motion Intensity')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('result_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  📊 对比图已保存: result_comparison.png")

    print(f"\n{'='*60}")
    print("  全部测试完成!")
    print(f"{'='*60}")
