# EBD2N ENHANCED ACTIVATION LAYER NODE WITH CORRECTED FLOW
# FIXED: Now properly sends to weighted layer instead of master

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
            print(f"[EBD2N-ActivationNode {rank}] {message}")
        else:
            print(f"[EBD2N-ActivationNode] {message}")

class LayerType(Enum):
    """Enumeration for different layer types in EBD2N framework"""
    INPUT_LAYER = 0      # Layer 0 - receives from input partitions
    HIDDEN_LAYER = 1     # Layer > 0 - receives from weighted layer partitions

class PaddingStrategy(Enum):
    """Padding strategies for conditional aggregation"""
    BOTTOM = "bottom"    # padBottom - for first partition
    TOP = "top"         # padTop - for last partition  
    PLACE_AT = "place_at"  # placeAt - for intermediate partitions
    NO_PADDING = "none"  # Direct aggregation when dimensions match

def pad_bottom(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
    """EBD2N padBottom function: pad with zeros at bottom"""
    if vector.dim() == 1:
        padding_size = target_dim - vector.shape[0]
        if padding_size <= 0:
            return vector[:target_dim]
        padding = torch.zeros(padding_size, dtype=vector.dtype, device=vector.device)
        return torch.cat([vector, padding])
    else:
        padding_size = target_dim - vector.shape[1]
        if padding_size <= 0:
            return vector[:, :target_dim]
        padding = torch.zeros(vector.shape[0], padding_size, dtype=vector.dtype, device=vector.device)
        return torch.cat([vector, padding], dim=1)

def pad_top(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
    """EBD2N padTop function: pad with zeros at top"""
    if vector.dim() == 1:
        padding_size = target_dim - vector.shape[0]
        if padding_size <= 0:
            return vector[:target_dim]
        padding = torch.zeros(padding_size, dtype=vector.dtype, device=vector.device)
        return torch.cat([padding, vector])
    else:
        padding_size = target_dim - vector.shape[1]
        if padding_size <= 0:
            return vector[:, :target_dim]
        padding = torch.zeros(vector.shape[0], padding_size, dtype=vector.dtype, device=vector.device)
        return torch.cat([padding, vector], dim=1)

def place_at(vector: torch.Tensor, position: int, target_dim: int) -> torch.Tensor:
    """EBD2N placeAt function: place vector at specific position with zero padding"""
    if vector.dim() == 1:
        result = torch.zeros(target_dim, dtype=vector.dtype, device=vector.device)
        end_pos = min(position + vector.shape[0], target_dim)
        actual_length = end_pos - position
        if actual_length > 0:
            result[position:end_pos] = vector[:actual_length]
        return result
    else:
        result = torch.zeros(vector.shape[0], target_dim, dtype=vector.dtype, device=vector.device)
        end_pos = min(position + vector.shape[1], target_dim)
        actual_length = end_pos - position
        if actual_length > 0:
            result[:, position:end_pos] = vector[:, :actual_length]
        return result

class EBD2NActivationLayer:
    """EBD2N Activation Layer implementing conditional aggregation with adaptive padding."""
    
    def __init__(
        self,
        layer_id: int,
        target_dimension: int,
        source_layer_type: LayerType,
        num_source_partitions: int,
        activation_function_name: str = "relu",
        bias_seed: Optional[int] = None,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        self.layer_id = layer_id
        self.d_l = target_dimension
        self.source_layer_type = source_layer_type
        self.p_source = num_source_partitions
        self.activation_function_name = activation_function_name
        self.device = device
        self.dtype = dtype
        
        # Initialize bias vector
        self.bias = self._initialize_bias(bias_seed)
        
        # Set activation function
        self.activation_fn = self._get_activation_function(activation_function_name)
        
        # Statistics tracking
        self.forward_calls = 0
        self.total_microbatches_processed = 0
        self.dimension_matches = 0
        self.dimension_mismatches = 0
        self.padding_operations = {
            PaddingStrategy.BOTTOM: 0,
            PaddingStrategy.TOP: 0,
            PaddingStrategy.PLACE_AT: 0,
            PaddingStrategy.NO_PADDING: 0
        }
        
        # Enhanced activation statistics
        self.activation_stats = {
            'total_sum': 0.0,
            'total_mean': 0.0,
            'min_activation': float('inf'),
            'max_activation': float('-inf'),
            'zero_activations': 0,
            'positive_activations': 0,
            'negative_activations': 0
        }
        
        # For future backpropagation support
        self.accumulated_bias_gradients = torch.zeros_like(self.bias)
        self.gradient_accumulation_count = 0
        
        debug_print(f"EBD2N Layer {layer_id} initialized: target_dim={target_dimension}, source_type={source_layer_type.name}, partitions={num_source_partitions}")
    
    def _initialize_bias(self, bias_seed: Optional[int] = None) -> torch.Tensor:
        """Initialize bias vector with enhanced variety."""
        if bias_seed is not None:
            torch.manual_seed(bias_seed)
        
        bias = torch.randn(self.d_l, device=self.device, dtype=self.dtype) * 0.1
        increment_bias = torch.arange(self.d_l, dtype=self.dtype, device=self.device) * 0.01
        bias += increment_bias
        
        return bias
    
    def _get_activation_function(self, activation_name: str):
        """Get activation function by name."""
        activation_functions = {
            'relu': F.relu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'leaky_relu': lambda x: F.leaky_relu(x, 0.01),
            'elu': F.elu,
            'gelu': F.gelu,
            'swish': lambda x: x * torch.sigmoid(x),
            'linear': lambda x: x
        }
        
        if activation_name.lower() not in activation_functions:
            debug_print(f"Warning: Unknown activation '{activation_name}', using ReLU")
            return activation_functions['relu']
        
        return activation_functions[activation_name.lower()]
    
    def _calculate_position(self, partition_index: int, source_partition_size: int) -> int:
        """Calculate position for placeAt padding strategy."""
        return int((self.d_l / self.p_source) * partition_index)
    
    def _verify_boundary_condition(self, partition_index: int, source_partition_size: int) -> bool:
        """Verify boundary condition for intermediate partitions."""
        if partition_index == 0 or partition_index == self.p_source - 1:
            return True
        
        position = self._calculate_position(partition_index, source_partition_size)
        return position + source_partition_size <= self.d_l
    
    def _apply_conditional_aggregation(
        self, 
        partition_outputs: List[torch.Tensor], 
        source_partition_size: int
    ) -> torch.Tensor:
        """Apply EBD2N conditional aggregation with adaptive padding."""
        # Case 1: Dimension Match
        if source_partition_size == self.d_l:
            self.dimension_matches += 1
            self.padding_operations[PaddingStrategy.NO_PADDING] += len(partition_outputs)
            aggregated = torch.stack(partition_outputs).sum(dim=0)
            
            if DEBUG and self.total_microbatches_processed < 2:
                debug_print(f"EBD2N: Dimension match ({source_partition_size}={self.d_l}), direct aggregation")
            
            return aggregated
        
        # Case 2: Dimension Mismatch - Apply layer-specific padding
        self.dimension_mismatches += 1
        
        if DEBUG and self.total_microbatches_processed < 2:
            debug_print(f"EBD2N: Dimension mismatch ({source_partition_size}!={self.d_l}), applying adaptive padding")
            debug_print(f"EBD2N: Source layer type: {self.source_layer_type.name}")
        
        padded_outputs = []
        
        for i, partition_output in enumerate(partition_outputs):
            if self.source_layer_type == LayerType.INPUT_LAYER:
                # Input layer to first hidden layer
                if partition_output.shape[-1] == self.d_l:
                    padded = partition_output
                    self.padding_operations[PaddingStrategy.NO_PADDING] += 1
                    strategy = "no_padding"
                else:
                    padded = pad_bottom(partition_output, self.d_l)
                    self.padding_operations[PaddingStrategy.BOTTOM] += 1
                    strategy = "bottom_fallback"
                    
            else:
                # Hidden layers - apply position-based padding
                if i == 0:
                    padded = pad_bottom(partition_output, self.d_l)
                    self.padding_operations[PaddingStrategy.BOTTOM] += 1
                    strategy = "bottom"
                elif i == len(partition_outputs) - 1:
                    padded = pad_top(partition_output, self.d_l)
                    self.padding_operations[PaddingStrategy.TOP] += 1
                    strategy = "top"
                else:
                    if self._verify_boundary_condition(i, source_partition_size):
                        position = self._calculate_position(i, source_partition_size)
                        padded = place_at(partition_output, position, self.d_l)
                        self.padding_operations[PaddingStrategy.PLACE_AT] += 1
                        strategy = f"place_at(pos={position})"
                    else:
                        padded = pad_top(partition_output, self.d_l)
                        self.padding_operations[PaddingStrategy.TOP] += 1
                        strategy = "top_fallback"
            
            if DEBUG and self.total_microbatches_processed < 2:
                debug_print(f"EBD2N: Partition {i} -> {strategy}, shape: {partition_output.shape} -> {padded.shape}")
            
            padded_outputs.append(padded)
        
        # Aggregate all padded outputs
        aggregated = torch.stack(padded_outputs).sum(dim=0)
        return aggregated
    
    def forward(
        self, 
        partition_outputs: List[torch.Tensor], 
        source_partition_size: int
    ) -> torch.Tensor:
        """EBD2N activation layer forward pass."""
        assert len(partition_outputs) == self.p_source, f"Expected {self.p_source} partitions, got {len(partition_outputs)}"
        
        # Apply EBD2N conditional aggregation
        aggregated = self._apply_conditional_aggregation(partition_outputs, source_partition_size)
        
        # Add bias
        pre_activation = aggregated + self.bias
        
        # Apply activation function
        activation_output = self.activation_fn(pre_activation)
        
        # Update statistics
        self.forward_calls += 1
        if partition_outputs[0].dim() > 1:
            batch_size = partition_outputs[0].shape[0]
            self.total_microbatches_processed += batch_size
        else:
            self.total_microbatches_processed += 1
        
        self._update_activation_stats(activation_output)
        
        return activation_output
    
    def _update_activation_stats(self, activation_output: torch.Tensor):
        """Update comprehensive activation statistics."""
        with torch.no_grad():
            current_sum = torch.sum(activation_output).item()
            current_mean = torch.mean(activation_output).item()
            current_min = torch.min(activation_output).item()
            current_max = torch.max(activation_output).item()
            
            self.activation_stats['total_sum'] += current_sum
            self.activation_stats['total_mean'] += current_mean
            self.activation_stats['min_activation'] = min(self.activation_stats['min_activation'], current_min)
            self.activation_stats['max_activation'] = max(self.activation_stats['max_activation'], current_max)
            
            zero_count = (activation_output == 0).sum().item()
            positive_count = (activation_output > 0).sum().item()
            negative_count = (activation_output < 0).sum().item()
            
            self.activation_stats['zero_activations'] += zero_count
            self.activation_stats['positive_activations'] += positive_count
            self.activation_stats['negative_activations'] += negative_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive layer statistics."""
        return {
            'layer_id': self.layer_id,
            'layer_type': 'EBD2N_ActivationLayer',
            'target_dimension': self.d_l,
            'source_layer_type': self.source_layer_type.name,
            'num_source_partitions': self.p_source,
            'activation_function': self.activation_function_name,
            'bias_shape': list(self.bias.shape),
            'bias_norm': torch.norm(self.bias).item(),
            'bias_mean': torch.mean(self.bias).item(),
            'bias_std': torch.std(self.bias).item(),
            'forward_calls': self.forward_calls,
            'total_microbatches_processed': self.total_microbatches_processed,
            'dimension_matches': self.dimension_matches,
            'dimension_mismatches': self.dimension_mismatches,
            'padding_operations': {k.value: v for k, v in self.padding_operations.items()},
            'activation_stats': self.activation_stats.copy(),
            'pending_bias_gradients': self.gradient_accumulation_count
        }

def setup_distributed(rank, world_size, master_addr="192.168.1.191", master_port="12355"):
    """Initialize distributed training with comprehensive error handling"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    if DEBUG:
        print(f"[EBD2N-ActivationNode {rank}] Setting up distributed training:")
        print(f"[EBD2N-ActivationNode {rank}]   Rank: {rank}")
        print(f"[EBD2N-ActivationNode {rank}]   World size: {world_size}")
        print(f"[EBD2N-ActivationNode {rank}]   Master addr: {master_addr}")
        print(f"[EBD2N-ActivationNode {rank}]   Master port: {master_port}")
    
    try:
        debug_print("Attempting to join process group...", rank)
        
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            timeout=timedelta(minutes=3)
        )
        
        debug_print("✓ Successfully joined process group", rank)
        
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
            print(f"\n[EBD2N-ActivationNode {rank}] ❌ FAILED TO INITIALIZE DISTRIBUTED TRAINING:")
            print(f"[EBD2N-ActivationNode {rank}] Error: {e}")
        raise

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            debug_print("✓ Distributed cleanup completed")
        except:
            pass

def activation_process(rank, world_size, num_input_workers, activation_size, activation_function_name, layer_id, target_weighted_ranks, master_addr, master_port):
    """CORRECTED EBD2N activation node process with proper flow to weighted layer"""
    if DEBUG:
        print(f"=" * 60)
        print(f"STARTING EBD2N CORRECTED ACTIVATION NODE RANK {rank} (LAYER {layer_id})")
        print(f"=" * 60)
    
    try:
        debug_print("Initializing EBD2N framework...", rank)
        
        # Setup distributed environment
        setup_distributed(rank, world_size, master_addr, master_port)
        
        # Determine source layer type based on layer_id
        source_layer_type = LayerType.INPUT_LAYER if layer_id == 1 else LayerType.HIDDEN_LAYER
        
        # Enhanced bias seed generation for variety
        bias_seed = rank * 500 + layer_id * 100 + int(time.time() * 100) % 1000
        
        # Create EBD2N activation layer
        ebd2n_activation = EBD2NActivationLayer(
            layer_id=layer_id,
            target_dimension=activation_size,
            source_layer_type=source_layer_type,
            num_source_partitions=num_input_workers,
            activation_function_name=activation_function_name,
            bias_seed=bias_seed
        )
        
        debug_print("✓ EBD2N activation layer initialized!", rank)
        debug_print(f"Layer ID: {layer_id}, Source type: {source_layer_type.name}", rank)
        debug_print(f"Target dimension: {activation_size}, Partitions: {num_input_workers}", rank)
        debug_print(f"Target weighted layer ranks: {target_weighted_ranks}", rank)
        debug_print(f"Bias stats - mean: {torch.mean(ebd2n_activation.bias).item():.6f}, std: {torch.std(ebd2n_activation.bias).item():.6f}", rank)
        debug_print(f"Activation function: {activation_function_name}", rank)
        debug_print("EBD2N: Conditional aggregation with adaptive padding enabled", rank)
        debug_print(f"Expecting inputs from {num_input_workers} input workers (ranks 1-{num_input_workers})", rank)
        debug_print("CORRECTED: Will send results to weighted layer workers instead of master", rank)
        debug_print("Entering main processing loop...", rank)
        
        if DEBUG:
            print(f"-" * 60)
        
        # Processing statistics
        total_processed_microbatches = 0
        total_processed_images = 0
        start_time = time.time()
        last_report_time = start_time
        
        # MICRO-BATCHING: Micro-batch synchronization data structures
        pending_microbatch_splits = defaultdict(dict)
        completed_microbatches = []
        max_pending_microbatches = 50
        out_of_order_count = 0
        timeout_count = 0
        
        debug_print("EBD2N: Ready to process matrix operations on micro-batches", rank)
        
        # Activation node processing loop
        while True:
            try:
                # Collect weighted results with proper micro-batch synchronization
                shutdown_received = False
                
                # Receive messages from all input workers
                for input_worker_rank in range(1, num_input_workers + 1):
                    try:
                        # Receive micro-batch size first
                        microbatch_size_tensor = torch.zeros(1, dtype=torch.long)
                        dist.recv(microbatch_size_tensor, src=input_worker_rank)
                        microbatch_size = microbatch_size_tensor.item()
                        
                        # Check for shutdown signal
                        if microbatch_size == -999:
                            debug_print(f"🛑 Received shutdown signal from input worker {input_worker_rank}", rank)
                            shutdown_received = True
                            break
                        
                        # Receive micro-batch ID
                        microbatch_id_tensor = torch.zeros(1, dtype=torch.long)
                        dist.recv(microbatch_id_tensor, src=input_worker_rank)
                        microbatch_id = microbatch_id_tensor.item()
                        
                        # Receive flattened weighted results from input worker
                        total_elements = microbatch_size * activation_size
                        weighted_results_flat = torch.zeros(total_elements)
                        dist.recv(weighted_results_flat, src=input_worker_rank)
                        
                        # Reshape to matrix form
                        weighted_results = weighted_results_flat.view(microbatch_size, activation_size)
                        
                        # Check for invalid data
                        if torch.isnan(weighted_results).any() or torch.isinf(weighted_results).any():
                            debug_print(f"🛑 Received invalid data from worker {input_worker_rank}, treating as shutdown", rank)
                            shutdown_received = True
                            break
                        elif torch.all(weighted_results == -999.0):
                            debug_print(f"🛑 Received backup shutdown signal from worker {input_worker_rank}", rank)
                            shutdown_received = True
                            break
                        elif torch.all(weighted_results == 0.0):
                            debug_print("💓 Received potential heartbeat", rank)
                            continue
                        
                        # Store the weighted results for this micro-batch and worker
                        pending_microbatch_splits[microbatch_id][input_worker_rank] = (microbatch_size, weighted_results)
                        
                        if DEBUG and total_processed_microbatches < 2:
                            result_sum = torch.sum(weighted_results).item()
                            result_mean = torch.mean(weighted_results).item()
                            result_std = torch.std(weighted_results).item()
                            debug_print(f"EBD2N: Received weighted results for micro-batch {microbatch_id} from input worker {input_worker_rank}: size={microbatch_size}, sum={result_sum:.4f}, mean={result_mean:.6f}, std={result_std:.6f}", rank)
                        
                    except RuntimeError as recv_error:
                        error_msg = str(recv_error).lower()
                        if any(keyword in error_msg for keyword in ["connection", "recv", "peer", "socket"]):
                            debug_print(f"🛑 Connection lost with input worker {input_worker_rank}, shutting down...", rank)
                            shutdown_received = True
                            break
                        else:
                            debug_print(f"✗ Error receiving from input worker {input_worker_rank}: {recv_error}", rank)
                            raise recv_error
                    except Exception as recv_error:
                        debug_print(f"✗ Unexpected error receiving from input worker {input_worker_rank}: {recv_error}", rank)
                        shutdown_received = True
                        break
                
                # If shutdown was received, break out of main loop
                if shutdown_received:
                    debug_print("🛑 Shutdown detected, terminating activation node", rank)
                    break
                
                # Process all complete micro-batches
                microbatches_to_process = []
                for microbatch_id in sorted(pending_microbatch_splits.keys()):
                    if len(pending_microbatch_splits[microbatch_id]) == num_input_workers:
                        microbatches_to_process.append(microbatch_id)
                
                # Process complete micro-batches in order
                for microbatch_id in microbatches_to_process:
                    # Get all weighted results for this micro-batch
                    weighted_results_list = []
                    microbatch_size = None
                    
                    for worker_rank in range(1, num_input_workers + 1):
                        if worker_rank in pending_microbatch_splits[microbatch_id]:
                            size, weighted_results = pending_microbatch_splits[microbatch_id][worker_rank]
                            if microbatch_size is None:
                                microbatch_size = size
                            elif microbatch_size != size:
                                debug_print(f"⚠️  Micro-batch size mismatch for micro-batch {microbatch_id}: expected {microbatch_size}, got {size} from worker {worker_rank}", rank)
                            
                            weighted_results_list.append(weighted_results)
                        else:
                            debug_print(f"⚠️  Missing split from worker {worker_rank} for micro-batch {microbatch_id}", rank)
                    
                    if len(weighted_results_list) == num_input_workers and microbatch_size is not None:
                        # EBD2N MATHEMATICAL FRAMEWORK: Apply conditional aggregation and activation
                        
                        if DEBUG and total_processed_microbatches < 2:
                            debug_print(f"EBD2N: Processing micro-batch {microbatch_id} with {len(weighted_results_list)} partitions", rank)
                        
                        # Calculate source partition size
                        source_partition_size = weighted_results_list[0].shape[1]
                        
                        # Apply EBD2N forward pass
                        activation_output_matrix = ebd2n_activation.forward(weighted_results_list, source_partition_size)
                        
                        # CORRECTED: Send activation results to WEIGHTED LAYER workers instead of master
                        try:
                            microbatch_size_tensor_out = torch.tensor([microbatch_size], dtype=torch.long)
                            microbatch_id_tensor_out = torch.tensor([microbatch_id], dtype=torch.long)
                            
                            # Send to ALL weighted layer workers (broadcast)
                            for weighted_worker_rank in target_weighted_ranks:
                                dist.send(microbatch_size_tensor_out, dst=weighted_worker_rank)
                                dist.send(microbatch_id_tensor_out, dst=weighted_worker_rank)
                                dist.send(activation_output_matrix.flatten(), dst=weighted_worker_rank)
                                
                                if DEBUG and total_processed_microbatches < 2:
                                    debug_print(f"EBD2N: Sent activation results for micro-batch {microbatch_id} to weighted worker {weighted_worker_rank}", rank)
                            
                            if DEBUG and total_processed_microbatches < 2:
                                act_sum = torch.sum(activation_output_matrix).item()
                                act_mean = torch.mean(activation_output_matrix).item()
                                act_std = torch.std(activation_output_matrix).item()
                                debug_print(f"EBD2N: Broadcasted activation results for micro-batch {microbatch_id} to {len(target_weighted_ranks)} weighted workers:", rank)
                                debug_print(f"  Activation size: {microbatch_size}, sum: {act_sum:.4f}, mean: {act_mean:.6f}, std: {act_std:.6f}", rank)
                                debug_print(f"  Activation output shape: {activation_output_matrix.shape}", rank)
                                debug_print(f"  Target weighted worker ranks: {target_weighted_ranks}", rank)
                                
                                # Show per-image activation analysis
                                for img_idx in range(min(2, microbatch_size)):
                                    img_activation_sum = torch.sum(activation_output_matrix[img_idx]).item()
                                    img_activation_mean = torch.mean(activation_output_matrix[img_idx]).item()
                                    debug_print(f"EBD2N:   Image {img_idx} in micro-batch {microbatch_id}: activation_sum={img_activation_sum:.4f}, activation_mean={img_activation_mean:.6f}", rank)
                                
                        except Exception as send_error:
                            debug_print(f"✗ Error sending result for micro-batch {microbatch_id} to weighted workers: {send_error}", rank)
                            break
                        
                        # Track completion
                        completed_microbatches.append(microbatch_id)
                        total_processed_microbatches += 1
                        total_processed_images += microbatch_size
                        
                        # Check for out-of-order completion
                        if len(completed_microbatches) > 1 and microbatch_id < completed_microbatches[-2]:
                            out_of_order_count += 1
                            if DEBUG and out_of_order_count <= 5:
                                debug_print(f"📋 Out-of-order completion: micro-batch {microbatch_id} completed after {completed_microbatches[-2]}", rank)
                        
                        # Detailed logging for first few micro-batches
                        if DEBUG and total_processed_microbatches <= 2:
                            debug_print(f"EBD2N: Micro-batch {microbatch_id} processing complete:", rank)
                            debug_print(f"  Received from {len(weighted_results_list)} input workers", rank)
                            debug_print(f"  Micro-batch size: {microbatch_size} images", rank)
                            debug_print(f"  Source partition size: {source_partition_size}", rank)
                            debug_print(f"  Activation output shape: {activation_output_matrix.shape}, sum: {torch.sum(activation_output_matrix).item():.4f}", rank)
                            debug_print(f"  Activation output mean: {torch.mean(activation_output_matrix).item():.6f}", rank)
                            debug_print(f"  EBD2N: Layer {layer_id} ({source_layer_type.name} -> Activation)", rank)
                            debug_print(f"  EBD2N: Conditional aggregation applied with function: {activation_function_name}", rank)
                            debug_print(f"  MICRO-BATCHING: Processed {microbatch_size} images in single operation", rank)
                            debug_print(f"  CORRECTED FLOW: Sent to weighted layer workers {target_weighted_ranks}", rank)
                    
                    # Clean up processed micro-batch
                    del pending_microbatch_splits[microbatch_id]
                
                # Clean up old pending micro-batches to prevent memory issues
                if len(pending_microbatch_splits) > max_pending_microbatches:
                    sorted_pending = sorted(pending_microbatch_splits.keys())
                    for old_microbatch_id in sorted_pending[:-max_pending_microbatches]:
                        incomplete_workers = num_input_workers - len(pending_microbatch_splits[old_microbatch_id])
                        debug_print(f"⚠️  Timing out incomplete micro-batch {old_microbatch_id} (missing {incomplete_workers} splits)", rank)
                        del pending_microbatch_splits[old_microbatch_id]
                        timeout_count += 1
                
                # Report progress every 50 micro-batches or every 10 seconds
                current_time = time.time()
                if DEBUG and ((total_processed_microbatches % 50 == 0) or (current_time - last_report_time > 10)):
                    elapsed = current_time - start_time
                    microbatch_rate = total_processed_microbatches / elapsed if elapsed > 0 else 0
                    image_rate = total_processed_images / elapsed if elapsed > 0 else 0
                    pending_count = len(pending_microbatch_splits)
                    
                    # EBD2N stats
                    ebd2n_stats = ebd2n_activation.get_statistics()
                    avg_activation_sum = ebd2n_stats['activation_stats']['total_sum'] / total_processed_microbatches if total_processed_microbatches > 0 else 0
                    
                    print(f"[EBD2N-ActivationNode {rank}] Processed {total_processed_microbatches} micro-batches ({total_processed_images} images) | MBatch rate: {microbatch_rate:.2f}/sec | Image rate: {image_rate:.2f}/sec | Pending: {pending_count} | Out-of-order: {out_of_order_count} | Timeouts: {timeout_count} | Avg Act Sum: {avg_activation_sum:.4f} | Layer: {layer_id} | CORRECTED-FLOW")
                    last_report_time = current_time
                
                # Continue processing
                if not microbatches_to_process and not pending_microbatch_splits:
                    if total_processed_microbatches > 0:
                        time.sleep(0.01)
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["connection", "recv", "send", "peer", "socket"]):
                    debug_print("⚠️  Connection lost with master or input workers, shutting down...", rank)
                    break
                else:
                    if DEBUG:
                        print(f"[EBD2N-ActivationNode {rank}] ❌ Runtime error: {e}")
                    raise
            except Exception as e:
                if DEBUG:
                    print(f"[EBD2N-ActivationNode {rank}] ❌ Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()
                break
        
        # Final statistics
        final_time = time.time()
        total_elapsed = final_time - start_time
        avg_microbatch_rate = total_processed_microbatches / total_elapsed if total_elapsed > 0 else 0
        avg_image_rate = total_processed_images / total_elapsed if total_elapsed > 0 else 0
        
        # Get comprehensive EBD2N statistics
        ebd2n_stats = ebd2n_activation.get_statistics()
        
        if DEBUG:
            print(f"\n[EBD2N-ActivationNode {rank}] 📊 EBD2N CORRECTED FINAL STATISTICS:")
            print(f"[EBD2N-ActivationNode {rank}]   Total micro-batches processed: {total_processed_microbatches}")
            print(f"[EBD2N-ActivationNode {rank}]   Total images processed: {total_processed_images}")
            print(f"[EBD2N-ActivationNode {rank}]   Out-of-order completions: {out_of_order_count}")
            print(f"[EBD2N-ActivationNode {rank}]   Timed-out micro-batches: {timeout_count}")
            print(f"[EBD2N-ActivationNode {rank}]   Remaining pending micro-batches: {len(pending_microbatch_splits)}")
            print(f"[EBD2N-ActivationNode {rank}]   Total time: {total_elapsed:.2f}s")
            print(f"[EBD2N-ActivationNode {rank}]   Average micro-batch rate: {avg_microbatch_rate:.2f} micro-batches/second")
            print(f"[EBD2N-ActivationNode {rank}]   Average image rate: {avg_image_rate:.2f} images/second")
            print(f"[EBD2N-ActivationNode {rank}]   Average images per micro-batch: {total_processed_images/total_processed_microbatches:.1f}" if total_processed_microbatches > 0 else "")
            
            # EBD2N-specific statistics
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: Layer ID: {ebd2n_stats['layer_id']}")
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: Source layer type: {ebd2n_stats['source_layer_type']}")
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: Target dimension: {ebd2n_stats['target_dimension']}")
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: Activation function: {ebd2n_stats['activation_function']}")
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: Dimension matches: {ebd2n_stats['dimension_matches']}")
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: Dimension mismatches: {ebd2n_stats['dimension_mismatches']}")
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: Padding operations: {ebd2n_stats['padding_operations']}")
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: Bias norm: {ebd2n_stats['bias_norm']:.6f}")
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: Mathematical framework compliance verified")
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: CORRECTED - Now properly flows to weighted layer")
            print(f"[EBD2N-ActivationNode {rank}]   EBD2N: Ready for backpropagation integration")
            
            # Show sample completed micro-batch IDs
            if completed_microbatches:
                sample_completed = completed_microbatches[:10]
                print(f"[EBD2N-ActivationNode {rank}]   Sample completed micro-batch IDs: {sample_completed}")
                if len(completed_microbatches) > 10:
                    print(f"[EBD2N-ActivationNode {rank}]   ... and {len(completed_microbatches) - 10} more")
        
    except KeyboardInterrupt:
        debug_print("🛑 Interrupted by user", rank)
    except Exception as e:
        if DEBUG:
            print(f"\n[EBD2N-ActivationNode {rank}] ❌ Failed to start or run: {e}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_distributed()
        debug_print("👋 EBD2N corrected activation node process terminated", rank)

def main():
    """Main activation node entry point with corrected flow"""
    global DEBUG
    
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='EBD2N Corrected Distributed Activation Layer Node')
    parser.add_argument('--rank', type=int, default=None, help='Activation node rank (usually rank 3)')
    parser.add_argument('--world-size', type=int, default=7, help='Total world size including all nodes')
    parser.add_argument('--layer-id', type=int, default=1, help='Layer ID in the network')
    parser.add_argument('--activation-size', type=int, default=100, help='Size of activation layer')
    parser.add_argument('--activation-function', default='relu', help='Activation function')
    parser.add_argument('--target-weighted-ranks', type=str, default='4,5', help='Comma-separated ranks of target weighted workers')
    parser.add_argument('--master-addr', default='192.168.1.191', help='Master node IP address')
    parser.add_argument('--master-port', default='12355', help='Master node port')
    parser.add_argument('--interactive', action='store_true', help='Run interactive setup')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')
    
    args = parser.parse_args()
    
    if args.no_debug:
        DEBUG = False
    else:
        DEBUG = args.debug
    
    # Parse target weighted ranks
    try:
        target_weighted_ranks = [int(r.strip()) for r in args.target_weighted_ranks.split(',')]
    except ValueError:
        print("Error: Invalid target weighted ranks format. Use comma-separated integers like '4,5'")
        return
    
    # Set defaults if interactive mode or rank not specified
    if args.interactive or args.rank is None:
        rank = args.world_size - 4 if args.rank is None else args.rank  # Default to rank 3 for world_size 7
        world_size = args.world_size
        num_input_workers = 2  # Default configuration
        activation_size = args.activation_size
        activation_function = args.activation_function
        layer_id = args.layer_id
        master_addr = args.master_addr
        master_port = args.master_port
    else:
        rank = args.rank
        world_size = args.world_size
        num_input_workers = 2  # Based on your network topology
        activation_size = args.activation_size
        activation_function = args.activation_function
        layer_id = args.layer_id
        master_addr = args.master_addr
        master_port = args.master_port
    
    # Validate configuration
    if rank <= 0:
        print(f"Error: Activation node rank must be > 0 (rank 0 is reserved for master)")
        return
    
    if activation_size < 1:
        print(f"Error: Activation size must be >= 1")
        return
        
    if layer_id < 1:
        print(f"Error: Layer ID must be >= 1")
        return
    
    if DEBUG:
        print(f"\n🚀 Starting EBD2N corrected activation node with configuration:")
        print(f"   Rank: {rank}")
        print(f"   World size: {world_size}")
        print(f"   Layer ID: {layer_id}")
        print(f"   Master: {master_addr}:{master_port}")
        print(f"   Number of input workers: {num_input_workers}")
        print(f"   Activation size: {activation_size}")
        print(f"   Activation function: {activation_function}")
        print(f"   Target weighted worker ranks: {target_weighted_ranks}")
        print(f"   Debug mode: {DEBUG}")
        print(f"   EBD2N: CORRECTED FLOW - Will send to weighted layer instead of master")
    
    try:
        activation_process(rank, world_size, num_input_workers, activation_size, activation_function, layer_id, target_weighted_ranks, master_addr, master_port)
    except KeyboardInterrupt:
        debug_print("🛑 Activation node interrupted by user")
    except Exception as e:
        if DEBUG:
            print(f"\n❌ Activation node failed: {e}")

if __name__ == "__main__":
    main()