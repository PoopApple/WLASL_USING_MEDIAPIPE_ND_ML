---

## 3. Data Extraction & Pre-processing Pipeline

Based on `ExtractLandmarks/1-pipeline.py` and `ExtractLandmarks/normalise_data.py`:

**Extraction Engine:**
*   **Models:** MediaPipe `pose_landmarker_heavy.task` and `hand_landmarker.task`.
*   **Raw Output:** 75 landmarks (33 pose, 21 left hand, 21 right hand).
*   **Reduced Set:** 64 landmarks selected for signing (21 pose landmarks + 42 hand landmarks + 1 custom feature).

**Normalization (The Signer-Agnostic Process):**
1.  **Spatial:** 
    *   Centering: Subtract the mid-point of the shoulders from all landmarks.
    *   Scaling: Divide all (X, Y) coordinates by the shoulder width. This ensures that whether a person is far or near, the movement "units" are the same.
    *   Wrist Delta: The 64th landmark is explicitly calculated as `Left_Wrist - Right_Wrist`.
2.  **Temporal:** 
    *   Fixed to **128 frames**. 
    *   Short videos are zero-padded (and a `mask` is generated).
    *   Long videos are downsampled using `linspace` indices to retain the full motion path.

**Training Augmentations (`augmentation.py`):**
*   **Gaussian Noise:** Adds jitter ($\sigma=0.015$) to landmarks to prevent the model from memorizing exact coordinates.
*   **Spatial Scaling:** Randomly scales the signer by 0.9x to 1.1x.
*   **Note:** Aggressive temporal drops and landmark dropouts are currently *disabled* in the code because they "teleport" joints to the shoulder-center (0,0,0) in our normalized space, which can confuse the GRU's trajectory logic.

---

## 4. Comprehensive Performance Benchmarks (500-Word Set)

From the `Results/dataset3_500words_analysis/dataset3_500w_summary.md`, we see how different architectures handle the same kinematic features:

| Model | Top-1 Acc | Top-5 Acc | F1-Score | Status |
| :--- | :---: | :---: | :---: | :--- |
| **bigru_bigger_v1** | **88.0%** | **97.4%** | **0.88** | Previous Champion |
| **bigru_bigger_angular_v1** | 80.0% | 95.3% | 0.80 | Solid |
| **bigru_v2** | 79.0% | 94.8% | 0.78 | Legacy |
| **original** | 78.0% | 94.2% | 0.78 | Baseline |
| **bigru_flash** | 74.0% | 93.3% | 0.74 | Lighter model |
| **bigru_angular_v1** | 61.0% | 87.8% | 0.61 | Under-capacity |
| **dualpath** | 32.0% | 62.4% | 0.30 | Struggling |
| **conv_bigru / tcn** | ~0.0% | ~0.0% | 0.00 | Failed to converge |

### Observations:
*   **The "Conv" Failure:** CNN-based models (`conv_bigru`, `tcn`) failed to converge entirely. This suggests that for kinematic vectors, the local spatial correlation found by 1D-Convs is less important than the global temporal state tracked by BiGRUs.
*   **The "Flash" Potential:** Even a "lighter" flash model reached 74%. Our current `bigru_biggest_angular_flash_v1` combines the best of the champion (`bigru_bigger_v1`'s capacity) with the explicit kinematics of `flash`.

---

## 5. Mathematical Feasibility & Conclusion

