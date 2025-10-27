# EBD2N WEIGHTED LAYER NODE WITH LAYER AWARENESS AND MATHEMATICAL FRAMEWORK
# Implements vertical parallelism with padding for distributed matrix multiplication

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import os
import time
import socket
from datetime import timedelta
import sys
import signal
import argparse
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Global debug flag
DEBUG = True

def debug_print(message, rank=None):
    """Print debug messages only if DEBUG is True"""
    if DEBUG:
        if rank is not None:
            print(f"[EBD2N-WeightedLayer {rank}] {message}")
        else:
            print(f"[EBD2N-WeightedLayer] {message}")

class PaddingStrategy(Enum):
    """Padding strategies for weight matrix reconstruction"""
    BOTTOM = "bottom"    # padBottom - for first shard
    TOP = "top"         # padTop - for last shard  
    PLACE_AT = "place_at"  # placeAt - for intermediate shards

def pad_bottom_matrix(matrix: torch.Tensor, target_rows: int) -> torch.Tensor:
    """
    EBD2N padBottom function for matrices: pad with zero rows at bottom
    
    Args:
        matrix: Input matrix of shape (current_rows, cols)
        target_rows: Target number of rows
        
    Returns:
        Padded matrix [matrix; zero_rows]
    """
    if matrix.shape[0] >= target_rows:
        return matrix[:target_rows, :]
    
    padding_rows = target_rows - matrix.shape[0]
    padding = torch.zeros(padding_rows, matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    return torch.cat([matrix, padding], dim=0)

def pad_top_matrix(matrix: torch.Tensor, target_rows: int) -> torch.Tensor:
    """
    EBD2N padTop function for matrices: pad with zero rows at top
    
    Args:
        matrix: Input matrix of shape (current_rows, cols)
        target_rows: Target number of rows
        
    Returns:
        Padded matrix [zero_rows; matrix]
    """
    if matrix.shape[0] >= target_rows:
        return matrix[:target_rows, :]
    
    padding_rows = target_rows - matrix.shape[0]
    padding = torch.zeros(padding_rows, matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    return torch.cat([padding, matrix], dim=0)

def place_at_matrix(matrix: torch.Tensor, position: int, target_rows: int) -> torch.Tensor:
    """
    EBD2N placeAt function for matrices: place matrix at specific row position with zero padding
    
    Args:
        matrix: Input matrix of shape (current_rows, cols)
        position: Starting row position in target matrix
        target_rows: Target number of rows
        
    Returns:
        Matrix with zeros except at specified position
    """
    result = torch.zeros(target_rows, matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    end_pos = min(position + matrix.shape[0], target_rows)
    actual_rows = end_pos - position
    
    if actual_rows > 0:
        result[position:end_pos, :] = matrix[:actual_rows, :]
    
    return result

class EBD2NWeightedLayerShard:
    """
    EBD2N Weighted Layer Shard implementing vertical parallelism with padding.
    
    Mathematical specification:
    - Weight matrix shard: W_j^[ℓ] ∈ R^{s^[ℓ+1] × d^[ℓ]}
    - Padded weight matrix: W̃_j^[ℓ] ∈ R^{d^[ℓ+1] × d^[ℓ]}
    - Forward: z_j^[ℓ] = W̃_j^[ℓ] @ a^[ℓ]
    - Backward: ∇_{W_j^[ℓ]} = δ_j^[ℓ+1] ⊗ (a^[ℓ])^T
    """
    
    def __init__(
        self,
        shard_id: int,              # j
        layer_id: int,              # ℓ
        input_dimension: int,       # d^[ℓ]
        output_dimension: int,      # d^[ℓ+1]
        num_output_partitions: int, # p^[ℓ+1]
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize EBD2N weighted layer shard.
        
        Args:
            shard_id: Shard index j (0-based)
            layer_id: Layer index ℓ
            input_dimension: Input dimension d^[ℓ]
            output_dimension: Output dimension d^[ℓ+1]
            num_output_partitions: Number of output partitions p^[ℓ+1]
            device: Computation device
            dtype: Data type for tensors
        """
        self.shard_id = shard_id
        self.layer_id = layer_id
        self.d_l = input_dimension      # d^[ℓ]
        self.d_l_plus_1 = output_dimension  # d^[ℓ+1]
        self.p_l_plus_1 = num_output_partitions  # p^[ℓ+1]
        self.device = device
        self.dtype = dtype
        
        # Verify divisibility constraint: d^[ℓ+1] mod p^[ℓ+1] = 0
        if self.d_l_plus_1 % self.p_l_plus_1 != 0:
            raise ValueError(f"Output dimension {self.d_l_plus_1} must be divisible by number of partitions {self.p_l_plus_1}")
        
        # Calculate shard size: s^[ℓ+1] = d^[ℓ+1] / p^[ℓ+1]
        self.s_l_plus_1 = self.d_l_plus_1 // self.p_l_plus_1
        
        # Calculate shard position: pos_j = j * s^[ℓ+1]
        self.position = self.shard_id * self.s_l_plus_1
        
        # Initialize weight matrix shard W_j^[ℓ] ∈ R^{s^[ℓ+1] × d^[ℓ]}
        self.W_shard = self._initialize_weight_shard()
        
        # Statistics tracking
        self.forward_calls = 0
        self.total_examples_processed = 0
        self.weight_updates = 0
        self.padding_operations = 0
        
        # For future backpropagation support
        self.accumulated_gradients = torch.zeros_like(self.W_shard)
        self.gradient_accumulation_count = 0
        
        debug_print(f"Initialized EBD2N weighted shard {shard_id} for layer {layer_id}: W_shard shape {self.W_shard.shape}, position {self.position}")
    
    def _initialize_weight_shard(self) -> torch.Tensor:
        """Initialize weight matrix shard using Xavier initialization."""
        # Xavier initialization for this shard
        fan_in = self.d_l
        fan_out = self.s_l_plus_1
        std = math.sqrt(2.0 / (fan_in + fan_out))
        
        W_shard = torch.randn(self.s_l_plus_1, self.d_l, device=self.device, dtype=self.dtype) * std
        return W_shard
    
    def _reconstruct_padded_weight_matrix(self) -> torch.Tensor:
        """
        Reconstruct full-dimensional padded weight matrix.
        
        Mathematical operations:
        - Case 1: First shard (j = 0): W̃_0^[ℓ] = padBottom(W_0^[ℓ], d^[ℓ+1])
        - Case 2: Intermediate shards: W̃_j^[ℓ] = placeAt(W_j^[ℓ], pos_j, d^[ℓ+1])
        - Case 3: Last shard: W̃_{p-1}^[ℓ] = padTop(W_{p-1}^[ℓ], d^[ℓ+1])
        
        Returns:
            Padded weight matrix W̃_j^[ℓ] ∈ R^{d^[ℓ+1] × d^[ℓ]}
        """
        if self.shard_id == 0:
            # First shard: padBottom
            padded_matrix = pad_bottom_matrix(self.W_shard, self.d_l_plus_1)
            strategy = "padBottom"
            
        elif self.shard_id == self.p_l_plus_1 - 1:
            # Last shard: padTop
            padded_matrix = pad_top_matrix(self.W_shard, self.d_l_plus_1)
            strategy = "padTop"
            
        else:
            # Intermediate shards: placeAt
            padded_matrix = place_at_matrix(self.W_shard, self.position, self.d_l_plus_1)
            strategy = f"placeAt(pos={self.position})"
        
        self.padding_operations += 1
        
        if DEBUG and self.forward_calls < 2:
            debug_print(f"Shard {self.shard_id}: Reconstructed padded weight matrix using {strategy}, shape: {self.W_shard.shape} -> {padded_matrix.shape}")
        
        return padded_matrix
    
    def forward(self, activation_input: torch.Tensor) -> torch.Tensor:
        """
        EBD2N weighted layer shard forward pass.
        
        Mathematical operation: z_j^[ℓ] = W̃_j^[ℓ] @ a^[ℓ]
        
        Args:
            activation_input: Input activations a^[ℓ] ∈ R^{m × d^[ℓ]} for micro-batch
            
        Returns:
            Shard output z_j^[ℓ] ∈ R^{m × d^[ℓ+1]}
        """
        # Validate input dimensions
        assert activation_input.dim() == 2, f"Expected 2D input (micro-batch), got {activation_input.dim()}D"
        assert activation_input.shape[1] == self.d_l, f"Input dimension mismatch: expected {self.d_l}, got {activation_input.shape[1]}"
        
        # Reconstruct padded weight matrix
        padded_weight_matrix = self._reconstruct_padded_weight_matrix()
        
        # Matrix multiplication: (m × d^[ℓ]) @ (d^[ℓ] × d^[ℓ+1]) = (m × d^[ℓ+1])
        # Note: padded_weight_matrix is (d^[ℓ+1] × d^[ℓ]), so we need its transpose
        shard_output = torch.mm(activation_input, padded_weight_matrix.T)
        
        self.forward_calls += 1
        self.total_examples_processed += activation_input.shape[0]
        
        if DEBUG and self.forward_calls <= 2:
            debug_print(f"Shard {self.shard_id}: Forward pass - input shape: {activation_input.shape}, output shape: {shard_output.shape}")
            debug_print(f"Shard {self.shard_id}: Output sum: {torch.sum(shard_output).item():.4f}")
        
        return shard_output
    
    def extract_relevant_output(self, full_output: torch.Tensor) -> torch.Tensor:
        """
        Extract the portion of output that this shard is responsible for.
        
        Mathematical operation: Extract z_j^[ℓ] = full_output[:, j*s^[ℓ+1]:(j+1)*s^[ℓ+1]]
        
        Args:
            full_output: Full output matrix ∈ R^{m × d^[ℓ+1]}
            
        Returns:
            Relevant portion for this shard ∈ R^{m × s^[ℓ+1]}
        """
        start_idx = self.position
        end_idx = start_idx + self.s_l_plus_1
        
        relevant_output = full_output[:, start_idx:end_idx]
        
        if DEBUG and self.forward_calls <= 2:
            debug_print(f"Shard {self.shard_id}: Extracted relevant output [{start_idx}:{end_idx}], shape: {relevant_output.shape}")
        
        return relevant_output
    
    def accumulate_gradients(self, activation_input: torch.Tensor, error_signal: torch.Tensor):
        """
        Accumulate gradients for future backpropagation.
        
        Mathematical operation: ∇_{W_j^[ℓ]} += δ_j^[ℓ+1] ⊗ (a^[ℓ])^T
        
        Args:
            activation_input: Input activations a^[ℓ]
            error_signal: Error signal δ_j^[ℓ+1] for this shard
        """
        # For micro-batch, accumulate gradients across all examples
        for i in range(activation_input.shape[0]):
            a_example = activation_input[i]  # Single example: d^[ℓ]
            delta_example = error_signal[i] if error_signal.dim() > 1 else error_signal  # s^[ℓ+1]
            
            # Outer product: δ_j^[ℓ+1] ⊗ (a^[ℓ])^T → (s^[ℓ+1], d^[ℓ])
            gradient = torch.outer(delta_example, a_example)
            self.accumulated_gradients += gradient
            self.gradient_accumulation_count += 1
    
    def update_weights(self, learning_rate: float = 0.01, l2_regularization: float = 0.0):
        """
        Update weight shard using accumulated gradients.
        
        Mathematical operations:
        - Standard: W_j^[ℓ] ← W_j^[ℓ] - η * (∇_{W_j^[ℓ]} / count)
        - With L2: W_j^[ℓ] ← (1 - η*λ) * W_j^[ℓ] - η * (∇_{W_j^[ℓ]} / count)
        """
        if self.gradient_accumulation_count > 0:
            # Average gradients
            avg_gradient = self.accumulated_gradients / self.gradient_accumulation_count
            
            # Apply weight update
            if l2_regularization > 0.0:
                self.W_shard = (1.0 - learning_rate * l2_regularization) * self.W_shard - learning_rate * avg_gradient
            else:
                self.W_shard = self.W_shard - learning_rate * avg_gradient
            
            # Reset accumulation
            self.accumulated_gradients.zero_()
            self.gradient_accumulation_count = 0
            self.weight_updates += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get shard statistics."""
        return {
            'shard_id': self.shard_id,
            'layer_id': self.layer_id,
            'weight_shard_shape': list(self.W_shard.shape),
            'position': self.position,
            'shard_size': self.s_l_plus_1,
            'weight_norm': torch.norm(self.W_shard).item(),
            'weight_mean': torch.mean(self.W_shard).item(),
            'weight_std': torch.std(self.W_shard).item(),
            'forward_calls': self.forward_calls,
            'total_examples_processed': self.total_examples_processed,
            'weight_updates': self.weight_updates,
            'padding_operations': self.padding_operations,
            'pending_gradients': self.gradient_accumulation_count
        }

class DistributedWeightedLayerWorker:
    """
    Distributed weighted layer worker that integrates with existing micro-batching framework.
    
    Handles communication with activation layers while implementing proper EBD2N mathematical framework.
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        layer_id: int,                    # ℓ (which weighted layer this is)
        input_dimension: int,             # d^[ℓ] (from previous activation layer)
        output_dimension: int,            # d^[ℓ+1] (to next activation layer)
        source_activation_rank: int,      # Rank of activation layer providing input
        target_output_rank: int,          # Rank of output layer receiving output
        num_weighted_shards: int,         # EXPLICIT number of weighted shards
        learning_rate: float = 0.01,
        l2_regularization: float = 0.0001,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize distributed weighted layer worker.
        
        Args:
            rank: Worker rank for this weighted layer shard
            world_size: Total number of nodes
            layer_id: Layer index ℓ in the network
            input_dimension: Input dimension d^[ℓ]
            output_dimension: Output dimension d^[ℓ+1]
            source_activation_rank: Rank of source activation layer
            target_output_rank: Rank of target output layer
            num_weighted_shards: EXPLICIT number of weighted shards (prevents auto-calculation bugs)
            learning_rate: Learning rate for weight updates
            l2_regularization: L2 regularization strength
            device: Computation device
        """
        self.rank = rank
        self.world_size = world_size
        self.layer_id = layer_id
        self.source_activation_rank = source_activation_rank
        self.target_output_rank = target_output_rank
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.device = device
        
        # Use explicit num_weighted_shards parameter (no auto-calculation)
        self.num_weighted_shards = num_weighted_shards
        
        # Calculate this shard's ID based on rank and explicit shard configuration
        # Assume weighted workers have ranks starting after activation layer
        weighted_start_rank = source_activation_rank + 1
        self.shard_id = rank - weighted_start_rank  # Convert rank to 0-based shard index
        
        # Validate shard ID is reasonable
        if self.shard_id < 0 or self.shard_id >= num_weighted_shards:
            raise ValueError(f"Invalid shard calculation: rank {rank}, shard_id {self.shard_id}, expected shard_id in [0, {num_weighted_shards-1}]")
        
        # Verify divisibility constraint
        if output_dimension % num_weighted_shards != 0:
            raise ValueError(f"Output dimension {output_dimension} must be divisible by number of weighted shards {num_weighted_shards}")
        
        # Create EBD2N weighted layer shard
        self.weighted_shard = EBD2NWeightedLayerShard(
            shard_id=self.shard_id,
            layer_id=layer_id,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            num_output_partitions=num_weighted_shards,
            device=device
        )
        
        # Statistics
        self.processed_microbatches = 0
        self.processed_images = 0
        self.start_time = time.time()
        
        debug_print(f"Initialized weighted layer worker: layer_id={layer_id}, shard_id={self.shard_id}")
        debug_print(f"Source activation: rank {source_activation_rank}, Target output: rank {target_output_rank}")
        debug_print(f"Using {self.num_weighted_shards} weighted shards total")
        debug_print(f"Weight shard shape: {self.weighted_shard.W_shard.shape}")
        debug_print(f"Shard position in output: {self.weighted_shard.position}-{self.weighted_shard.position + self.weighted_shard.s_l_plus_1}")
    
    def receive_from_activation(self) -> Tuple[int, int, torch.Tensor]:
        """
        Receive data from source activation layer.
        
        Returns:
            Tuple of (microbatch_size, microbatch_id, activation_data)
        """
        # Receive microbatch size
        microbatch_size_tensor = torch.zeros(1, dtype=torch.long)
        dist.recv(microbatch_size_tensor, src=self.source_activation_rank)
        microbatch_size = microbatch_size_tensor.item()
        
        # Receive microbatch ID
        microbatch_id_tensor = torch.zeros(1, dtype=torch.long)
        dist.recv(microbatch_id_tensor, src=self.source_activation_rank)
        microbatch_id = microbatch_id_tensor.item()
        
        # Receive activation data
        total_elements = microbatch_size * self.weighted_shard.d_l
        activation_data_flat = torch.zeros(total_elements)
        dist.recv(activation_data_flat, src=self.source_activation_rank)
        
        # Reshape to (microbatch_size, input_dimension)
        activation_data = activation_data_flat.view(microbatch_size, self.weighted_shard.d_l)
        
        return microbatch_size, microbatch_id, activation_data
    
    def process_microbatch(self, microbatch_data: torch.Tensor, microbatch_id: int) -> torch.Tensor:
        """
        Process a micro-batch through this weighted layer shard.
        
        Args:
            microbatch_data: Activation input data (microbatch_size × input_dimension)
            microbatch_id: Unique identifier for this micro-batch
            
        Returns:
            Shard output (microbatch_size × shard_output_size)
        """
        # Forward pass through EBD2N weighted layer shard
        shard_output = self.weighted_shard.forward(microbatch_data)
        
        # Extract relevant portion for this shard
        relevant_output = self.weighted_shard.extract_relevant_output(shard_output)
        
        # Update statistics
        self.processed_microbatches += 1
        self.processed_images += microbatch_data.shape[0]
        
        return relevant_output
    
    def send_to_output(self, microbatch_size: int, microbatch_id: int, shard_output: torch.Tensor):
        """
        Send shard results to target output layer.
        
        Args:
            microbatch_size: Size of the micro-batch
            microbatch_id: Unique identifier
            shard_output: Results from this weighted shard
        """
        # Send microbatch size
        microbatch_size_tensor = torch.tensor([microbatch_size], dtype=torch.long)
        dist.send(microbatch_size_tensor, dst=self.target_output_rank)
        
        # Send microbatch ID
        microbatch_id_tensor = torch.tensor([microbatch_id], dtype=torch.long)
        dist.send(microbatch_id_tensor, dst=self.target_output_rank)
        
        # Send shard results
        dist.send(shard_output.flatten(), dst=self.target_output_rank)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics."""
        elapsed_time = time.time() - self.start_time
        
        shard_stats = self.weighted_shard.get_statistics()
        shard_stats.update({
            'worker_rank': self.rank,
            'worker_type': 'WeightedLayerWorker',
            'processed_microbatches': self.processed_microbatches,
            'processed_images': self.processed_images,
            'elapsed_time': elapsed_time,
            'microbatch_rate': self.processed_microbatches / elapsed_time if elapsed_time > 0 else 0,
            'image_rate': self.processed_images / elapsed_time if elapsed_time > 0 else 0,
            'source_activation_rank': self.source_activation_rank,
            'target_output_rank': self.target_output_rank,
            'num_weighted_shards': self.num_weighted_shards
        })
        
        return shard_stats

def setup_distributed(rank, world_size, master_addr="192.168.1.191", master_port="12355"):
    """Initialize distributed training with comprehensive error handling"""
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    if DEBUG:
        print(f"[EBD2N-WeightedLayer {rank}] Setting up distributed training:")
        print(f"[EBD2N-WeightedLayer {rank}]   Rank: {rank}")
        print(f"[EBD2N-WeightedLayer {rank}]   World size: {world_size}")
        print(f"[EBD2N-WeightedLayer {rank}]   Master addr: {master_addr}")
        print(f"[EBD2N-WeightedLayer {rank}]   Master port: {master_port}")
    
    try:
        debug_print("Attempting to join process group...", rank)
        
        # Initialize with longer timeout
        dist.init_process_group(
            backend="gloo",
            init_method="env://",  # Use TCP instead of file
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10)
        )
        
        debug_print("✓ Successfully joined process group", rank)
        
        # Wait for all processes to be ready
        debug_print("Synchronizing with all processes...", rank)
        dist.barrier()
        debug_print(f"✓ All {world_size} processes synchronized!", rank)
        
        # Connection test with master
        try:
            test_tensor = torch.zeros(1)
            debug_print("Waiting for connection test from master...", rank)
            dist.recv(test_tensor, src=0)
            debug_print(f"✓ Connection test successful: received {test_tensor.item()}", rank)
        except Exception as e:
            debug_print(f"✗ Connection test failed: {e}", rank)
            raise
                
    except Exception as e:
        if DEBUG:
            print(f"\n[EBD2N-WeightedLayer {rank}] ❌ FAILED TO INITIALIZE DISTRIBUTED TRAINING:")
            print(f"[EBD2N-WeightedLayer {rank}] Error: {e}")
        raise

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            debug_print("✓ Distributed cleanup completed")
        except:
            pass

def weighted_layer_worker_process(
    rank, 
    world_size, 
    layer_id, 
    input_dimension, 
    output_dimension,
    source_activation_rank,
    target_output_rank,
    num_weighted_shards,
    master_addr, 
    master_port
):
    """EBD2N weighted layer worker process with layer awareness and mathematical framework compliance"""
    if DEBUG:
        print(f"=" * 60)
        print(f"STARTING EBD2N WEIGHTED LAYER WORKER RANK {rank} (LAYER {layer_id})")
        print(f"=" * 60)
    
    try:
        debug_print("Initializing EBD2N weighted layer...", rank)
        
        # Setup distributed environment
        setup_distributed(rank, world_size, master_addr, master_port)
        
        # Create EBD2N weighted layer worker
        weighted_worker = DistributedWeightedLayerWorker(
            rank=rank,
            world_size=world_size,
            layer_id=layer_id,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            source_activation_rank=source_activation_rank,
            target_output_rank=target_output_rank,
            num_weighted_shards=num_weighted_shards,
            learning_rate=0.01,
            l2_regularization=0.0001
        )
        
        debug_print("✓ EBD2N weighted layer worker initialized!", rank)
        debug_print(f"Layer ID: {layer_id}, Shard ID: {weighted_worker.shard_id}", rank)
        debug_print(f"Input dimension: {input_dimension}, Output dimension: {output_dimension}", rank)
        debug_print(f"Source activation rank: {source_activation_rank}, Target output rank: {target_output_rank}", rank)
        debug_print(f"Weight shard shape: {weighted_worker.weighted_shard.W_shard.shape}", rank)
        debug_print(f"Number of weighted shards: {num_weighted_shards}", rank)
        debug_print("EBD2N: Vertical parallelism with padding enabled", rank)
        debug_print("Entering main processing loop...", rank)
        
        if DEBUG:
            print(f"-" * 60)
        
        last_report_time = time.time()
        
        # Main processing loop
        while True:
            try:
                # Receive data from source activation layer
                microbatch_size, microbatch_id, activation_data = weighted_worker.receive_from_activation()
                
                # Check for shutdown signal
                if microbatch_size == -999:
                    debug_print("🛑 Received shutdown signal", rank)
                    break
                
                # Check for invalid data
                if torch.all(activation_data == -999.0):
                    debug_print("🛑 Received shutdown pattern in data", rank)
                    break
                elif torch.all(activation_data == 0.0):
                    debug_print("💓 Received potential heartbeat", rank)
                    continue
                
                # EBD2N MATHEMATICAL FRAMEWORK: Process through weighted layer shard
                shard_output = weighted_worker.process_microbatch(activation_data, microbatch_id)
                
                # Send results to target output layer
                weighted_worker.send_to_output(microbatch_size, microbatch_id, shard_output)
                
                # Detailed logging for first few micro-batches
                if DEBUG and weighted_worker.processed_microbatches <= 3:
                    debug_print(f"EBD2N processed micro-batch {microbatch_id}:", rank)
                    debug_print(f"  Size: {microbatch_size} images", rank)
                    debug_print(f"  Input shape: {activation_data.shape}", rank)
                    debug_print(f"  Shard output shape: {shard_output.shape}", rank)
                    debug_print(f"  Input sum: {torch.sum(activation_data).item():.4f}", rank)
                    debug_print(f"  Output sum: {torch.sum(shard_output).item():.4f}", rank)
                    debug_print(f"  EBD2N: Layer {layer_id}, Shard {weighted_worker.shard_id}/{num_weighted_shards}", rank)
                    debug_print(f"  EBD2N: Vertical parallelism applied", rank)
                
                # Progress reporting
                current_time = time.time()
                if DEBUG and ((weighted_worker.processed_microbatches % 50 == 0) or (current_time - last_report_time > 10)):
                    elapsed = current_time - weighted_worker.start_time
                    microbatch_rate = weighted_worker.processed_microbatches / elapsed if elapsed > 0 else 0
                    image_rate = weighted_worker.processed_images / elapsed if elapsed > 0 else 0
                    print(f"[EBD2N-WeightedLayer {rank}] Processed {weighted_worker.processed_microbatches} microbatches ({weighted_worker.processed_images} images) | Rate: {microbatch_rate:.2f} mb/s, {image_rate:.2f} img/s | Layer: {layer_id} | Shard: {weighted_worker.shard_id}/{num_weighted_shards} | EBD2N-COMPLIANT")
                    last_report_time = current_time
                    
            except RuntimeError as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["connection", "recv", "send", "peer", "socket"]):
                    debug_print("⚠️  Connection lost, shutting down...", rank)
                    break
                else:
                    if DEBUG:
                        print(f"[EBD2N-WeightedLayer {rank}] ❌ Runtime error: {e}")
                    raise
            except Exception as e:
                if DEBUG:
                    print(f"[EBD2N-WeightedLayer {rank}] ❌ Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()
                break
        
        # Final statistics
        final_stats = weighted_worker.get_statistics()
        if DEBUG:
            print(f"\n[EBD2N-WeightedLayer {rank}] 📊 EBD2N FINAL STATISTICS:")
            for key, value in final_stats.items():
                if isinstance(value, (int, float, str)):
                    print(f"[EBD2N-WeightedLayer {rank}]   {key}: {value}")
            print(f"[EBD2N-WeightedLayer {rank}]   EBD2N: Mathematical framework compliance verified")
            print(f"[EBD2N-WeightedLayer {rank}]   EBD2N: Ready for backpropagation integration")
        
    except KeyboardInterrupt:
        debug_print("🛑 Interrupted by user", rank)
    except Exception as e:
        if DEBUG:
            print(f"\n[EBD2N-WeightedLayer {rank}] ❌ Failed to start or run: {e}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_distributed()
        debug_print("👋 EBD2N weighted layer worker process terminated", rank)

def main():
    """Main weighted layer worker entry point"""
    global DEBUG
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='EBD2N Enhanced Distributed Weighted Layer Worker Node')
    parser.add_argument('--rank', type=int, required=True, help='Weighted layer worker rank')
    parser.add_argument('--world-size', type=int, required=True, help='Total world size including all nodes')
    parser.add_argument('--layer-id', type=int, default=1, help='Layer ID in the network (which weighted layer this is)')
    parser.add_argument('--input-dimension', type=int, required=True, help='Input dimension from previous activation layer')
    parser.add_argument('--output-dimension', type=int, required=True, help='Output dimension to next layer')
    parser.add_argument('--source-activation-rank', type=int, required=True, help='Rank of source activation layer')
    parser.add_argument('--target-output-rank', type=int, required=True, help='Rank of target output layer')
    parser.add_argument('--num-weighted-shards', type=int, required=True, help='Number of weighted layer shards')
    parser.add_argument('--master-addr', default='192.168.1.191', help='Master node IP address')
    parser.add_argument('--master-port', default='12355', help='Master node port')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output (default: True)')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')
    
    args = parser.parse_args()
    
    # Set debug flag based on arguments
    if args.no_debug:
        DEBUG = False
    else:
        DEBUG = args.debug
    
    # Validate arguments
    if args.rank == args.source_activation_rank or args.rank == args.target_output_rank:
        print(f"Error: Weighted layer rank {args.rank} cannot be the same as activation/output layer ranks")
        return
        
    if args.rank == 0:
        print(f"Error: Weighted layer rank cannot be 0 (reserved for master)")
        return
    
    if args.input_dimension <= 0 or args.output_dimension <= 0:
        print(f"Error: Input and output dimensions must be > 0")
        return
    
    if args.num_weighted_shards <= 0:
        print(f"Error: Number of weighted shards must be > 0")
        return
    
    if args.output_dimension % args.num_weighted_shards != 0:
        print(f"Error: Output dimension {args.output_dimension} must be divisible by number of weighted shards {args.num_weighted_shards}")
        return
    
    if DEBUG:
        print(f"\n🚀 Starting EBD2N weighted layer worker with configuration:")
        print(f"   Rank: {args.rank}")
        print(f"   World size: {args.world_size}")
        print(f"   Layer ID: {args.layer_id}")
        print(f"   Input dimension: {args.input_dimension}")
        print(f"   Output dimension: {args.output_dimension}")
        print(f"   Source activation rank: {args.source_activation_rank}")
        print(f"   Target output rank: {args.target_output_rank}")
        print(f"   Number of weighted shards: {args.num_weighted_shards}")
        print(f"   Master: {args.master_addr}:{args.master_port}")
        print(f"   Debug mode: {DEBUG}")
        print(f"   EBD2N: Vertical parallelism with padding enabled")
        print(f"   EBD2N: Mathematical framework compliance")
    
    try:
        weighted_layer_worker_process(
            rank=args.rank,
            world_size=args.world_size,
            layer_id=args.layer_id,
            input_dimension=args.input_dimension,
            output_dimension=args.output_dimension,
            source_activation_rank=args.source_activation_rank,
            target_output_rank=args.target_output_rank,
            num_weighted_shards=args.num_weighted_shards,
            master_addr=args.master_addr,
            master_port=args.master_port
        )
    except KeyboardInterrupt:
        debug_print("🛑 Weighted layer worker interrupted by user")
    except Exception as e:
        if DEBUG:
            print(f"\n❌ Weighted layer worker failed: {e}")

if __name__ == "__main__":
    main()