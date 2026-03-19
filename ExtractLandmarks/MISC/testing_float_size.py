import numpy as np


if __name__ == "__main__":
    np.save("./zeros_only.npy", np.zeros((100, 75, 4), dtype=np.float32))
    rng = np.random.default_rng()
    float32_array = rng.random(size=(100, 75, 4), dtype=np.float32)
    np.save("./filled_garbage.npy", float32_array)
