import tensorflow as tf
import os
from tensorflow.keras.mixed_precision import experimental as mixed_precision

def optm():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        # ============================================
        # Optimisation Flags - Do not remove
        # ============================================

        # os.environ['CUDA_CACHE_DISABLE'] = '0'
        #
        # os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

        # os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        #
        # os.environ['TF_ADJUST_HUE_FUSED'] = '1'
        # os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
        # os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        #
        # os.environ['TF_SYNC_ON_FINISH'] = '0'
        # os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
        # os.environ['TF_DISABLE_NVTX_RANGES'] = '1'

        # =================================================
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)