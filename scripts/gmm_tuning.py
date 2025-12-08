import jax
import jax.numpy as jnp
from tpu_inference.kernels.megablox.gmm import gmm
import tune_jax
from tune_jax import tune, tune_logger

# Select the first TPU device to ensure execution on 1 TPU only
try:
    device = jax.devices()[0]
    print(f"Running tests on device: {device}")
except IndexError:
    print("No devices found. Using CPU fallback.")
    device = jax.local_devices()[0]


hyperparams = {
    "tiling_m": [32, 64, 128, 256, 512, 1024],
    "tiling_k": [32, 64, 128, 256, 512, 1024],
    "tiling_n": [32, 64, 128, 256, 512, 1024],
}

def test_gmm_configurations():
    g = 128
    m = 64 
    k = 3072
    n = 3072


    def gmm_wrapper(*args, tiling_m=None, tiling_n=None, tiling_k=None, **kwargs):
        return gmm(*args, **dict(kwargs, tiling=(tiling_m, tiling_n, tiling_k), preferred_element_type=jnp.float32, transpose_rhs=True))
    
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    lhs = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16)
    
    group_sizes = jnp.array([m // g] * g, dtype=jnp.int32)
    
    rhs = jax.random.normal(k2, (g, n, k), dtype=jnp.bfloat16)
    
    lhs = jax.device_put(lhs, device)
    rhs = jax.device_put(rhs, device)
    group_sizes = jax.device_put(group_sizes, device)

    gmm_tuned_jit = jax.jit(tune(gmm_wrapper, hyperparams=hyperparams))

    gmm_tuned_jit(lhs, rhs, group_sizes)

    print(tune_jax.tabulate(gmm_tuned_jit.timing_results))

    print(gmm_tuned_jit.optimal_hyperparams)

    

# Run the tests
if __name__ == "__main__":
    test_gmm_configurations()
