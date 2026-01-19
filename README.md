# Double-Helix Vision (DH) 

> **A tiny, geometry-based visual sampler extracted from my Swarm Intelligence project.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JackJ-C/double-helix-vision-tool/blob/main/DH.ipynb)

 💡 The Concept

I built DH to solve a specific problem: **How can AI agents "see" effectively when bandwidth is extremely limited?**

Instead of processing a full 1080p image, DH mimics biological eyes. It uses a **Golden Spiral** to sample the image, focusing high density on the center (Fovea) and low density on the edges (Peripheral).

![DH Architecture](assets/architecture.jpg)
*(Figure: Compressing 2D space into a 1D signal using spiral geometry)*

🧪 Minimal Demo (Reproducible)

This repo contains a clean, fair comparison between **DH** and **Random Sampling**.
I kept the code minimal so you can review it in < 5 minutes.

**The Experiment:**
* **Budget:** Only 256 pixels allowed (approx. 0.01% of original image).
* **Task:** Train a classifier using only these pixels.

**The Result (Epoch 15):**
* **Random Sampling:** ~20.4% Accuracy (Stuck)
* **DH (My Method):** **~27.6% Accuracy** (Learning steadily)

> **Conclusion:** Geometry matters. Even with just 256 points, structured sampling preserves vital context that random sampling misses.

## 🧠 Design Philosophy: Focus vs. Coverage

**Why did I define K=128 per helix (Total=256)?**

This parameter wasn't chosen for file size—it was chosen for **Robotic Focus**.

* **The Baseline Problem:** Random or Grid sampling attempts to "cover the screen" evenly. It assumes all pixels are equal.
* **The DH Solution:** In robotics, an agent needs to **"lock on"** to a target, not just view the scenery.
    * I set **K=128** as the minimal structural threshold for one helix to define a trajectory.
    * The two helices (Alpha & Beta) work together to create **Dynamic Foveation**—prioritizing the center (action) over the background (noise).

**In short:** We aren't trying to *compress* an image; we are simulating a biological eye searching for a target.
## 🛠️ Quick Start (Local)

If you prefer running this locally instead of Colab:

```bash
# 1. Clone the repo
git clone [https://github.com/JackJ-C/double-helix-vision-tool.git](https://github.com/JackJ-C/double-helix-vision-tool.git)
cd double-helix-vision-tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the script (Optional)
# You can use the notebook `DH.ipynb` or run the core script if you have one.
