"""
test.py — Double Helix Vision: Comprehensive Benchmark Suite
=============================================================
Benchmarks:
  1. Throughput — FPS at multiple resolutions vs Grid/Random sampling
  2. Information Density — Entropy & edge preservation per sample point
  3. Depth Collision Accuracy — Synthetic known-depth scenes, verify correlation
  4. Extreme Lighting Robustness — Pure black / white / gradient
  5. Robotics API Latency — End-to-end pipeline, JSON response time
  6. Compression Ratio — Full frame vs 1D signal data volume

Outputs:
  - Terminal summary table
  - benchmark_results.json  (machine-readable)
  - benchmark_report.png    (visual report)
"""

import numpy as np
import cv2
import time
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Back import DoubleHelixVisionTool, ReflectionEngine, SpatialMapper, DHVisionAPI


# ============================================================
# Helpers
# ============================================================

def make_synthetic_scene(w, h, num_objects=5, seed=42):
    """Generate a synthetic BGR scene with objects at known 'depths' (brightness layers)."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Background gradient (simulates depth: dark=far, bright=near)
    for y in range(h):
        v = int(40 + 60 * y / h)
        img[y, :] = (v, v, v)
    depths = []
    for i in range(num_objects):
        cx = rng.randint(w // 4, 3 * w // 4)
        cy = rng.randint(h // 4, 3 * h // 4)
        r = rng.randint(30, min(w, h) // 6)
        brightness = int(100 + 155 * (i / max(num_objects - 1, 1)))
        cv2.circle(img, (cx, cy), r, (brightness, brightness, brightness), -1)
        dist_from_center = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
        depths.append({'cx': cx, 'cy': cy, 'r': r, 'brightness': brightness,
                       'dist_center': float(dist_from_center)})
    return img, depths


def make_moving_frame(w, h, t, speed=5.0):
    """Generate frame with a bright rectangle moving horizontally."""
    img = np.full((h, w, 3), 50, dtype=np.uint8)
    x = int((w * 0.1) + (w * 0.6) * (0.5 + 0.5 * np.sin(t * speed)))
    y = h // 2
    cv2.rectangle(img, (x - 60, y - 40), (x + 60, y + 40), (220, 220, 220), -1)
    return img, x


def grid_sample(gray, n_points):
    """Uniform grid sampling."""
    h, w = gray.shape
    side = max(1, int(np.sqrt(n_points)))
    xs = np.linspace(0, w - 1, side).astype(int)
    ys = np.linspace(0, h - 1, side).astype(int)
    xx, yy = np.meshgrid(xs, ys)
    return gray[yy.ravel(), xx.ravel()]


def random_sample(gray, n_points, seed=0):
    """Random sampling."""
    h, w = gray.shape
    rng = np.random.RandomState(seed)
    xs = rng.randint(0, w, n_points)
    ys = rng.randint(0, h, n_points)
    return gray[ys, xs]


def signal_entropy(signal):
    """Shannon entropy of a uint8 signal."""
    hist, _ = np.histogram(signal.astype(np.uint8), bins=256, range=(0, 255))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def edge_score(signal):
    """Mean absolute gradient — measures how many edges are captured."""
    return float(np.mean(np.abs(np.diff(signal.astype(np.float32)))))


def print_table(headers, rows, col_width=16):
    """Pretty-print a table."""
    fmt = " | ".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("-" * (col_width * len(headers) + 3 * (len(headers) - 1)))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print()


# ============================================================
# Benchmark 1: Throughput
# ============================================================

def bench_throughput():
    print("\n" + "=" * 70)
    print("  BENCHMARK 1: Throughput (FPS) — DH vs Grid vs Random")
    print("=" * 70)

    resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
    n_warmup, n_iter = 50, 500
    results = []

    for (w, h) in resolutions:
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tool = DoubleHelixVisionTool(w, h)
        n_pts = len(tool.idx_xa)

        # DH
        for _ in range(n_warmup):
            tool.scan(img)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            tool.scan(img)
        dh_fps = n_iter / (time.perf_counter() - t0)

        # Grid
        for _ in range(n_warmup):
            grid_sample(gray, n_pts)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            grid_sample(gray, n_pts)
        grid_fps = n_iter / (time.perf_counter() - t0)

        # Random
        for _ in range(n_warmup):
            random_sample(gray, n_pts)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            random_sample(gray, n_pts)
        rand_fps = n_iter / (time.perf_counter() - t0)

        results.append({
            'resolution': f"{w}x{h}", 'dh_fps': round(dh_fps, 1),
            'grid_fps': round(grid_fps, 1), 'random_fps': round(rand_fps, 1),
            'sample_points': n_pts,
        })

    headers = ["Resolution", "Points", "DH FPS", "Grid FPS", "Random FPS"]
    rows = [[r['resolution'], r['sample_points'], r['dh_fps'], r['grid_fps'], r['random_fps']] for r in results]
    print_table(headers, rows)
    return results


# ============================================================
# Benchmark 2: Information Density
# ============================================================

def bench_info_density():
    print("=" * 70)
    print("  BENCHMARK 2: Information Density (Entropy & Edge Score)")
    print("=" * 70)

    w, h = 1920, 1080
    img, _ = make_synthetic_scene(w, h, num_objects=8, seed=42)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tool = DoubleHelixVisionTool(w, h)
    n_pts = len(tool.idx_xa)

    scan = tool.scan(img)
    dh_signal = np.concatenate([scan['alpha_stream'], scan['beta_stream']])
    grid_signal = grid_sample(gray, n_pts * 2)
    rand_signal = random_sample(gray, n_pts * 2)

    results = []
    for name, sig in [("DH", dh_signal), ("Grid", grid_signal), ("Random", rand_signal)]:
        e = signal_entropy(sig)
        es = edge_score(sig)
        results.append({'method': name, 'entropy': round(e, 4),
                        'edge_score': round(es, 4), 'points': len(sig)})

    headers = ["Method", "Points", "Entropy (bits)", "Edge Score"]
    rows = [[r['method'], r['points'], r['entropy'], r['edge_score']] for r in results]
    print_table(headers, rows)
    return results


# ============================================================
# Benchmark 3: Depth Collision Accuracy (Synthetic)
# ============================================================

def bench_depth_collision():
    print("=" * 70)
    print("  BENCHMARK 3: Temporal Collision — Motion Detection Accuracy")
    print("=" * 70)

    w, h = 1920, 1080
    api = DHVisionAPI(w, h)
    n_frames = 60
    speeds = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = []

    for spd in speeds:
        api.reflection.prev_scan = None
        intensities = []
        positions = []
        for i in range(n_frames):
            t = i / 30.0
            frame, x_pos = make_moving_frame(w, h, t, speed=spd)
            scan = api.dh.scan(frame)
            col = api.reflection.temporal_collision(scan)
            mi = float(np.mean(col['depth_edges'])) if col['has_data'] else 0.0
            intensities.append(mi)
            positions.append(x_pos)

        pos_diff = np.abs(np.diff(positions)).astype(float)
        col_arr = np.array(intensities[1:])
        # Correlation between position change and collision intensity
        corr = float(np.corrcoef(pos_diff, col_arr)[0, 1]) if pos_diff.std() > 0 else 0.0
        results.append({
            'speed': spd, 'mean_collision': round(float(col_arr.mean()), 3),
            'max_collision': round(float(col_arr.max()), 3),
            'motion_correlation': round(corr, 4),
        })

    headers = ["Speed", "Mean Collision", "Max Collision", "Correlation (r)"]
    rows = [[r['speed'], r['mean_collision'], r['max_collision'], r['motion_correlation']] for r in results]
    print_table(headers, rows)
    return results


# ============================================================
# Benchmark 4: Extreme Lighting Robustness
# ============================================================

def bench_extreme_lighting():
    print("=" * 70)
    print("  BENCHMARK 4: Extreme Lighting Robustness")
    print("=" * 70)

    w, h = 1920, 1080
    tool = DoubleHelixVisionTool(w, h)
    ref = ReflectionEngine(tool)

    scenes = {
        'Pure Black': np.zeros((h, w, 3), dtype=np.uint8),
        'Pure White': np.full((h, w, 3), 255, dtype=np.uint8),
        'Low Light (10)': np.full((h, w, 3), 10, dtype=np.uint8),
        'Overexposed (245)': np.full((h, w, 3), 245, dtype=np.uint8),
        'H-Gradient': None,
        'Normal Scene': None,
    }
    # Horizontal gradient
    grad = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        v = int(255 * x / w)
        grad[:, x] = (v, v, v)
    scenes['H-Gradient'] = grad
    scenes['Normal Scene'], _ = make_synthetic_scene(w, h, seed=99)

    # Add a small moving square to each to test if collision still works
    results = []
    for name, base in scenes.items():
        ref.prev_scan = None
        signals = []
        for i in range(5):
            frame = base.copy()
            x = 400 + i * 80
            cv2.rectangle(frame, (x, 400), (x + 100, 500),
                          (min(255, base.mean() + 80),) * 3, -1)
            scan = tool.scan(frame)
            col = ref.temporal_collision(scan)
            st = ref.stereo_depth(scan)
            if col['has_data']:
                signals.append(float(np.mean(col['depth_edges'])))

        detected = len(signals) > 0 and max(signals) > 0.5
        results.append({
            'scene': name,
            'motion_detected': '✅ Yes' if detected else '❌ No',
            'mean_signal': round(np.mean(signals), 3) if signals else 0.0,
            'max_signal': round(max(signals), 3) if signals else 0.0,
        })

    headers = ["Scene", "Motion Detected", "Mean Signal", "Max Signal"]
    rows = [[r['scene'], r['motion_detected'], r['mean_signal'], r['max_signal']] for r in results]
    print_table(headers, rows)
    return results


# ============================================================
# Benchmark 5: Robotics API End-to-End Latency
# ============================================================

def bench_robotics_api():
    print("=" * 70)
    print("  BENCHMARK 5: Robotics API — End-to-End Latency")
    print("=" * 70)

    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    n_warmup, n_iter = 20, 200
    results = []

    for (w, h) in resolutions:
        api = DHVisionAPI(w, h, fov=60.0, max_depth=10.0)
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        for _ in range(n_warmup):
            api.process_frame(img)

        latencies = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            report = api.process_frame(img)
            latencies.append((time.perf_counter() - t0) * 1000)

        lat = np.array(latencies)
        # Also measure JSON serialization
        t0 = time.perf_counter()
        for _ in range(n_iter):
            api.to_json(report)
        json_ms = (time.perf_counter() - t0) / n_iter * 1000

        results.append({
            'resolution': f"{w}x{h}",
            'mean_ms': round(float(lat.mean()), 3),
            'p50_ms': round(float(np.percentile(lat, 50)), 3),
            'p99_ms': round(float(np.percentile(lat, 99)), 3),
            'json_ms': round(json_ms, 3),
            'total_ms': round(float(lat.mean()) + json_ms, 3),
            'effective_fps': round(1000.0 / (float(lat.mean()) + json_ms), 1),
        })

    headers = ["Resolution", "Pipeline(ms)", "P50(ms)", "P99(ms)", "JSON(ms)", "Total(ms)", "Eff. FPS"]
    rows = [[r['resolution'], r['mean_ms'], r['p50_ms'], r['p99_ms'],
             r['json_ms'], r['total_ms'], r['effective_fps']] for r in results]
    print_table(headers, rows)

    # Robotics decision simulation
    print("  --- Robotics Decision Simulation ---")
    api = DHVisionAPI(1920, 1080)
    scene, _ = make_synthetic_scene(1920, 1080, seed=7)
    report = api.process_frame(scene)
    print(f"  API Response Sample (1920x1080):")
    print(f"    Frame ID:          {report['frame_id']}")
    print(f"    Signal Alpha Mean: {report['signal']['alpha_mean']:.2f}")
    print(f"    Motion Detected:   {report['motion']['detected']}")
    print(f"    Spatial Bounds X:  {report['spatial']['bounds']['x']}")
    print(f"    Spatial Bounds Z:  {report['spatial']['bounds']['z']}")
    print(f"    Total 3D Points:   {report['spatial']['total_points']}")
    dc = report['depth']['depth_confidence']
    peak_layer = int(np.argmax(dc))
    print(f"    Peak Depth Layer:  {peak_layer}/31 (confidence={dc[peak_layer]:.2f})")
    print(f"    JSON Size:         {len(api.to_json(report))} bytes")
    print()
    return results


# ============================================================
# Benchmark 6: Compression Ratio
# ============================================================

def bench_compression():
    print("=" * 70)
    print("  BENCHMARK 6: Data Compression Ratio")
    print("=" * 70)

    resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
    results = []

    for (w, h) in resolutions:
        tool = DoubleHelixVisionTool(w, h)
        full_size = w * h  # grayscale bytes
        dh_size = len(tool.idx_xa) * 2  # alpha + beta (uint8 each)
        ratio = full_size / dh_size

        results.append({
            'resolution': f"{w}x{h}",
            'full_frame_bytes': full_size,
            'dh_signal_bytes': dh_size,
            'compression_ratio': round(ratio, 1),
            'reduction_pct': round((1 - 1 / ratio) * 100, 2),
        })

    headers = ["Resolution", "Full Frame", "DH Signal", "Ratio", "Reduction %"]
    rows = [[r['resolution'], r['full_frame_bytes'], r['dh_signal_bytes'],
             f"{r['compression_ratio']}x", f"{r['reduction_pct']}%"] for r in results]
    print_table(headers, rows)
    return results


# ============================================================
# Report Generation
# ============================================================

def generate_report(all_results):
    """Generate visual benchmark report."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Double Helix Vision — Benchmark Report', fontsize=18, fontweight='bold', y=0.98)

    # 1. Throughput bar chart
    ax = axes[0, 0]
    tp = all_results['throughput']
    x = np.arange(len(tp))
    bw = 0.25
    ax.bar(x - bw, [r['dh_fps'] for r in tp], bw, label='DH', color='#00e5ff')
    ax.bar(x, [r['grid_fps'] for r in tp], bw, label='Grid', color='#ff9100')
    ax.bar(x + bw, [r['random_fps'] for r in tp], bw, label='Random', color='#b0bec5')
    ax.set_xticks(x)
    ax.set_xticklabels([r['resolution'] for r in tp], fontsize=8)
    ax.set_ylabel('FPS')
    ax.set_title('1. Throughput (FPS)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 2. Info density
    ax = axes[0, 1]
    info = all_results['info_density']
    methods = [r['method'] for r in info]
    colors = ['#00e5ff', '#ff9100', '#b0bec5']
    ax.bar(methods, [r['entropy'] for r in info], color=colors, alpha=0.8)
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title('2. Information Density')
    ax2 = ax.twinx()
    ax2.plot(methods, [r['edge_score'] for r in info], 'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('Edge Score', color='red')
    ax.grid(axis='y', alpha=0.3)

    # 3. Depth collision correlation
    ax = axes[0, 2]
    dc = all_results['depth_collision']
    speeds = [r['speed'] for r in dc]
    corrs = [r['motion_correlation'] for r in dc]
    ax.plot(speeds, corrs, 'o-', color='#00e5ff', linewidth=2, markersize=10)
    ax.fill_between(speeds, corrs, alpha=0.2, color='#00e5ff')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Strong (r>0.8)')
    ax.set_xlabel('Object Speed')
    ax.set_ylabel('Correlation (r)')
    ax.set_title('3. Motion-Collision Correlation')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Extreme lighting
    ax = axes[1, 0]
    el = all_results['extreme_lighting']
    scene_names = [r['scene'] for r in el]
    max_sigs = [r['max_signal'] for r in el]
    bar_colors = ['#00e5ff' if r['motion_detected'] == '✅ Yes' else '#ff1744' for r in el]
    bars = ax.barh(scene_names, max_sigs, color=bar_colors, alpha=0.8)
    ax.set_xlabel('Max Collision Signal')
    ax.set_title('4. Extreme Lighting Robustness')
    ax.grid(axis='x', alpha=0.3)

    # 5. Robotics API latency
    ax = axes[1, 1]
    rl = all_results['robotics_api']
    res_labels = [r['resolution'] for r in rl]
    total_ms = [r['total_ms'] for r in rl]
    eff_fps = [r['effective_fps'] for r in rl]
    ax.bar(res_labels, total_ms, color='#00e5ff', alpha=0.8, label='Latency (ms)')
    ax.set_ylabel('Total Latency (ms)')
    ax.set_title('5. Robotics API Latency')
    ax2 = ax.twinx()
    ax2.plot(res_labels, eff_fps, 'ro-', linewidth=2, markersize=10, label='Eff. FPS')
    ax2.set_ylabel('Effective FPS', color='red')
    ax.grid(axis='y', alpha=0.3)

    # 6. Compression ratio
    ax = axes[1, 2]
    cr = all_results['compression']
    res_labels = [r['resolution'] for r in cr]
    ratios = [r['compression_ratio'] for r in cr]
    ax.bar(res_labels, ratios, color='#00e5ff', alpha=0.8)
    for i, v in enumerate(ratios):
        ax.text(i, v + 0.5, f"{v}x", ha='center', fontweight='bold', fontsize=10)
    ax.set_ylabel('Compression Ratio')
    ax.set_title('6. Data Compression Ratio')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = 'benchmark_report.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  📊 Visual report saved: {out}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   Double Helix Vision — Comprehensive Benchmark Suite            ║")
    print("║   Testing: Throughput · Info Density · Depth · Lighting · API    ║")
    print("╚" + "═" * 68 + "╝")

    all_results = {}
    all_results['throughput'] = bench_throughput()
    all_results['info_density'] = bench_info_density()
    all_results['depth_collision'] = bench_depth_collision()
    all_results['extreme_lighting'] = bench_extreme_lighting()
    all_results['robotics_api'] = bench_robotics_api()
    all_results['compression'] = bench_compression()

    # Save JSON
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  💾 Results JSON saved: benchmark_results.json")

    # Generate visual report
    generate_report(all_results)

    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    tp = all_results['throughput']
    best = max(tp, key=lambda x: x['dh_fps'])
    print(f"  Peak Throughput:       {best['dh_fps']} FPS @ {best['resolution']}")
    dc = all_results['depth_collision']
    avg_corr = np.mean([r['motion_correlation'] for r in dc])
    print(f"  Motion Correlation:    r = {avg_corr:.4f} (avg across speeds)")
    el = all_results['extreme_lighting']
    robust = sum(1 for r in el if r['motion_detected'] == '✅ Yes')
    print(f"  Lighting Robustness:   {robust}/{len(el)} scenes detected motion")
    rl = all_results['robotics_api']
    best_api = min(rl, key=lambda x: x['total_ms'])
    print(f"  Best API Latency:      {best_api['total_ms']}ms @ {best_api['resolution']} ({best_api['effective_fps']} FPS)")
    cr = all_results['compression']
    best_cr = max(cr, key=lambda x: x['compression_ratio'])
    print(f"  Max Compression:       {best_cr['compression_ratio']}x @ {best_cr['resolution']}")
    print()
    print("  All results saved to: benchmark_results.json + benchmark_report.png")
    print("=" * 70)
