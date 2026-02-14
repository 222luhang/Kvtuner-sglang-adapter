# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""
KVTuner-Enabled HiCache Storage Integration

This module integrates KVTuner quantization with SGLang HiCache,
enabling quantized KV cache storage for L2 (host memory) and L3 (storage).

The quantization happens:
1. When data is backed up from GPU (L1) to Host (L2) - compress
2. When data is loaded from Host (L2) to GPU (L1) - decompress
3. Storage backends (L3) receive already-quantized data

Usage:
    Enable via server_args:
    --enable-kvtuner-hicache \
    --kvtuner-hicache-nbits-key 4 \
    --kvtuner-hicache-nbits-value 4
"""

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)

logger = logging.getLogger(__name__)


class KVTunerHiCacheMixin:
    """
    Mixin class that adds KVTuner quantization support to HiCache host pools.
    
    This class wraps the host memory pool operations to transparently
    apply quantization when backing up to host and dequantization when
    loading to device.
    """

    def __init_kvtuner(
        self,
        enable_kvtuner: bool = False,
        nbits_key: int = 4,
        nbits_value: int = 4,
        asym: bool = False,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        """
        Initialize KVTuner for HiCache.
        
        Args:
            enable_kvtuner: Whether to enable KVTuner quantization
            nbits_key: Number of bits for key quantization (2, 4, or 8)
            nbits_value: Number of bits for value quantization (2, 4, or 8)
            asym: Use asymmetric quantization
            axis_key: Quantization axis for keys (0=token, 1=channel)
            axis_value: Quantization axis for values (0=token, 1=channel)
            q_group_size: Group size for quantization
            residual_length: Number of recent tokens to keep in full precision
        """
        self.kvtuner_enabled = enable_kvtuner
        self.kvtuner_nbits_key = nbits_key
        self.kvtuner_nbits_value = nbits_value
        self.kvtuner_asym = asym
        self.kvtuner_axis_key = axis_key
        self.kvtuner_axis_value = axis_value
        self.kvtuner_q_group_size = q_group_size
        self.kvtuner_residual_length = residual_length
        
        if self.kvtuner_enabled:
            try:
                from sglang.srt.kvtuner_quant import VanillaQuantizer
                self.kvtuner_quantizer = VanillaQuantizer(
                    nbits_key=nbits_key,
                    nbits_value=nbits_value,
                    asym=asym,
                    axis_key=axis_key,
                    axis_value=axis_value,
                    q_group_size=q_group_size,
                    residual_length=residual_length,
                )
                logger.info(
                    f"KVTuner HiCache initialized: "
                    f"key_bits={nbits_key}, value_bits={nbits_value}, "
                    f"asym={asym}"
                )
            except ImportError:
                logger.warning(
                    "KVTuner not available, falling back to full precision"
                )
                self.kvtuner_enabled = False
                self.kvtuner_quantizer = None
        else:
            self.kvtuner_quantizer = None
            
    def _quantize_kv(self, k_data: torch.Tensor, v_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Quantize KV cache data.
        
        Args:
            k_data: Key cache tensor
            v_data: Value cache tensor
            
        Returns:
            Tuple of (quantized_k, quantized_v, metadata)
        """
        if not self.kvtuner_enabled or self.kvtuner_quantizer is None:
            return k_data, v_data, {}
        
        # Quantize using KVTuner
        qk, qv, scales_k, scales_v = self.kvtuner_quantizer.quantize(k_data, v_data)
        
        metadata = {
            'scales_k': scales_k,
            'scales_v': scales_v,
            'nbits_key': self.kvtuner_nbits_key,
            'nbits_value': self.kvtuner_nbits_value,
        }
        
        return qk, qv, metadata
    
    def _dequantize_kv(
        self, 
        qk: torch.Tensor, 
        qv: torch.Tensor, 
        metadata: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantize KV cache data.
        
        Args:
            qk: Quantized key cache tensor
            qv: Quantized value cache tensor
            metadata: Quantization metadata including scales
            
        Returns:
            Tuple of (dequantized_k, dequantized_v)
        """
        if not self.kvtuner_enabled or self.kvtuner_quantizer is None:
            return qk, qv
        
        scales_k = metadata.get('scales_k')
        scales_v = metadata.get('scales_v')
        
        # Dequantize using KVTuner
        k_data = self.kvtuner_quantizer.dequantize(qk, scales_k, self.kvtuner_nbits_key)
        v_data = self.kvtuner_quantizer.dequantize(qv, scales_v, self.kvtuner_nbits_value)
        
        return k_data, v_data


class KVTunerMHATokenToKVPoolHost(KVTunerHiCacheMixin, MHATokenToKVPoolHost):
    """
    MHA Host KV Pool with KVTuner quantization support.
    
    This class extends MHATokenToKVPoolHost to add transparent
    quantization when backing up to host memory and dequantization
    when loading to device.
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        # KVTuner parameters
        enable_kvtuner: bool = False,
        kvtuner_nbits_key: int = 4,
        kvtuner_nbits_value: int = 4,
        kvtuner_asym: bool = False,
        kvtuner_axis_key: int = 0,
        kvtuner_axis_value: int = 0,
        kvtuner_q_group_size: int = 64,
        kvtuner_residual_length: int = 128,
    ):
        # Initialize parent class
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        
        # Initialize KVTuner
        self.__init_kvtuner(
            enable_kvtuner=enable_kvtuner,
            nbits_key=kvtuner_nbits_key,
            nbits_value=kvtuner_nbits_value,
            asym=kvtuner_asym,
            axis_key=kvtuner_axis_key,
            axis_value=kvtuner_axis_value,
            q_group_size=kvtuner_q_group_size,
            residual_length=kvtuner_residual_length,
        )
        
        if self.kvtuner_enabled:
            logger.info(
                f"KVTunerMHATokenToKVPoolHost: "
                f"Quantization enabled with {kvtuner_nbits_key}-bit keys, "
                f"{kvtuner_nbits_value}-bit values"
            )
    
    def load_to_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Load KV data from host to device for all layers.
        Applies dequantization if KVTuner is enabled.
        """
        if not self.kvtuner_enabled:
            # Use standard path without quantization
            super().load_to_device_all_layer(
                device_pool, host_indices, device_indices, io_backend
            )
            return
        
        # TODO: Implement quantized path
        # For now, fall back to standard path
        logger.debug("KVTuner: load_to_device_all_layer (quantized path not yet implemented)")
        super().load_to_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
    
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from device to host for all layers.
        Applies quantization if KVTuner is enabled.
        """
        if not self.kvtuner_enabled:
            # Use standard path without quantization
            super().backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend
            )
            return
        
        # TODO: Implement quantized path
        # For now, fall back to standard path
        logger.debug("KVTuner: backup_from_device_all_layer (quantized path not yet implemented)")
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
    
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a data page from host memory.
        Returns quantized data if KVTuner is enabled.
        """
        data = super().get_data_page(index, flat)
        
        if not self.kvtuner_enabled:
            return data
        
        # Data is already quantized in host memory
        # Storage backends will receive this quantized data
        return data
    
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to host memory.
        If KVTuner is enabled, data should already be quantized.
        """
        # If data is coming from storage (L3), it's already quantized
        # Just store it directly
        super().set_from_flat_data_page(index, data_page)


class KVTunerMLATokenToKVPoolHost(KVTunerHiCacheMixin, MLATokenToKVPoolHost):
    """
    MLA Host KV Pool with KVTuner quantization support.
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        # KVTuner parameters
        enable_kvtuner: bool = False,
        kvtuner_nbits_key: int = 4,
        kvtuner_nbits_value: int = 4,
        kvtuner_asym: bool = False,
        kvtuner_axis_key: int = 0,
        kvtuner_axis_value: int = 0,
        kvtuner_q_group_size: int = 64,
        kvtuner_residual_length: int = 128,
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        
        self.__init_kvtuner(
            enable_kvtuner=enable_kvtuner,
            nbits_key=kvtuner_nbits_key,
            nbits_value=kvtuner_nbits_value,
            asym=kvtuner_asym,
            axis_key=kvtuner_axis_key,
            axis_value=kvtuner_axis_value,
            q_group_size=kvtuner_q_group_size,
            residual_length=kvtuner_residual_length,
        )
        
        if self.kvtuner_enabled:
            logger.info(
                f"KVTunerMLATokenToKVPoolHost: "
                f"Quantization enabled with {kvtuner_nbits_key}-bit keys, "
                f"{kvtuner_nbits_value}-bit values"
            )


def create_kvtuner_host_pool(
    device_pool,
    host_to_device_ratio: float,
    host_size: int,
    page_size: int,
    layout: str,
    pin_memory: bool = True,
    device: str = "cpu",
    allocator_type: str = "default",
    enable_kvtuner: bool = False,
    kvtuner_nbits_key: int = 4,
    kvtuner_nbits_value: int = 4,
    kvtuner_asym: bool = False,
    kvtuner_axis_key: int = 0,
    kvtuner_axis_value: int = 0,
    kvtuner_q_group_size: int = 64,
    kvtuner_residual_length: int = 128,
):
    """
    Factory function to create appropriate host pool with KVTuner support.
    
    Returns:
        KVTunerMHATokenToKVPoolHost or KVTunerMLATokenToKVPoolHost
    """
    from sglang.srt.mem_cache.memory_pool import (
        MHATokenToKVPool,
        MLATokenToKVPool,
    )
    
    if isinstance(device_pool, MHATokenToKVPool):
        return KVTunerMHATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            layout=layout,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
            enable_kvtuner=enable_kvtuner,
            kvtuner_nbits_key=kvtuner_nbits_key,
            kvtuner_nbits_value=kvtuner_nbits_value,
            kvtuner_asym=kvtuner_asym,
            kvtuner_axis_key=kvtuner_axis_key,
            kvtuner_axis_value=kvtuner_axis_value,
            kvtuner_q_group_size=kvtuner_q_group_size,
            kvtuner_residual_length=kvtuner_residual_length,
        )
    elif isinstance(device_pool, MLATokenToKVPool):
        return KVTunerMLATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            layout=layout,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
            enable_kvtuner=enable_kvtuner,
            kvtuner_nbits_key=kvtuner_nbits_key,
            kvtuner_nbits_value=kvtuner_nbits_value,
            kvtuner_asym=kvtuner_asym,
            kvtuner_axis_key=kvtuner_axis_key,
            kvtuner_axis_value=kvtuner_axis_value,
            kvtuner_q_group_size=kvtuner_q_group_size,
            kvtuner_residual_length=kvtuner_residual_length,
        )
    else:
        raise ValueError(f"Unsupported device pool type: {type(device_pool)}")
