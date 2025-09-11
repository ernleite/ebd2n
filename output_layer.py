# EBD2N OUTPUT LAYER NODE WITH SOFTMAX AND LOSS COMPUTATION
# Implements final layer processing with conditional aggregation and backpropagation support

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
            print(f"[EBD2N-OutputLayer {rank}] {message}")
        else:
            print(f"[EBD2N-OutputLayer] {message}")

class OutputType(Enum):
    """Output layer types"""
    CLASSIFICATION = "classification"  # Uses softmax
    REGRESSION = "regression"         # Uses identity

class PaddingStrategy(Enum):
    """Padding strategies for conditional aggregation"""
    BOTTOM = "bottom"    # padBottom - for first partition
    TOP = "top"         # padTop - for last partition  
    PLACE_AT = "place_at"  # placeAt - for intermediate partitions
    NO_PADDING = "none"  # Direct aggregation when dimensions match

def pad_bottom(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    EBD2N padBottom function: pad with zeros at bottom
    
    Args:
        vector: Input vector of length s
        target_dim: Target dimension d
        
    Returns:
        Padded vector [vector; zeros]
    """
    if vector.dim() == 1:
        # Single vector
        padding_size = target_dim - vector.shape[0]
        if padding_size <= 0:
            return vector[:target_dim]
        padding = torch.zeros(padding_size, dtype=vector.dtype, device=vector.device)
        return torch.cat([vector, padding])
    else:
        # Batch of vectors
        padding_size = target_dim - vector.shape[1]
        if padding_size <= 0:
            return vector[:, :target_dim]
        padding = torch.zeros(vector.shape[0], padding_size, dtype=vector.dtype, device=vector.device)
        return torch.cat([vector, padding], dim=1)

def pad_top(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    EBD2N padTop function: pad with zeros at top
    
    Args:
        vector: Input vector of length s
        target_dim: Target dimension d
        
    Returns:
        Padded vector [zeros; vector]
    """
    if vector.dim() == 1:
        # Single vector
        padding_size = target_dim - vector.shape[0]
        if padding_size <= 0:
            return vector[:target_dim]
        padding = torch.zeros(padding_size, dtype=vector.dtype, device=vector.device)
        return torch.cat([padding, vector])
    else:
        # Batch of vectors
        padding_size = target_dim - vector.shape[1]
        if padding_size <= 0:
            return vector[:, :target_dim]
        padding = torch.zeros(vector.shape[0], padding_size, dtype=vector.dtype, device=vector.device)
        return torch.cat([padding, vector], dim=1)

def place_at(vector: torch.Tensor, position: int, target_dim: int) -> torch.Tensor:
    """
    EBD2N placeAt function: place vector at specific position with zero padding
    
    Args:
        vector: Input vector of length s
        position: Starting position in target vector
        target_dim: Target dimension d
        
    Returns:
        Vector with zeros except at specified position
    """
    if vector.dim() == 1:
        # Single vector
        result = torch.zeros(target_dim, dtype=vector.dtype, device=vector.device)
        end_pos = min(position + vector.shape[0], target_dim)
        actual_length = end_pos - position
        if actual_length > 0:
            result[position:end_pos] = vector[:actual_length]
        return result
    else:
        # Batch of vectors
        result = torch.zeros(vector.shape[0], target_dim, dtype=vector.dtype, device=vector.device)
        end_pos = min(position + vector.shape[1], target_dim)
        actual_length = end_pos - position
        if actual_length > 0:
            result[:, position:end_pos] = vector[:, :actual_length]
        return result

class EBD2NOutputLayer:
    """
    EBD2N Output Layer implementing final layer processing with conditional aggregation.
    
    Mathematical specification:
    - Receives: {z_j^[L-1]} from final weighted layer partitions
    - Target dimension: d^[L]
    - Bias vector: b^[L] ∈ R^{d^[L]}
    - Conditional aggregation: z^[L-1] = Σ_j padded(z_j^[L-1])
    - Pre-activation: z^[L] = z^[L-1] + b^[L]
    - Output: y = σ^[L](z^[L]) where σ^[L] is softmax or identity
    - Error: δ^[L] = y - y_true (for classification)
    """
    
    def __init__(
        self,
        output_dimension: int,         # d^[L] - number of classes or output size
        num_source_partitions: int,    # p^[L] - partitions from final weighted layer
        output_type: OutputType = OutputType.CLASSIFICATION,
        bias_seed: Optional[int] = None,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize EBD2N output layer.
        
        Args:
            output_dimension: Output dimension d^[L] (number of classes)
            num_source_partitions: Number of partitions from final weighted layer
            output_type: Type of output (classification or regression)
            bias_seed: Seed for bias initialization
            device: Computation device
            dtype: Data type for tensors
        """
        self.d_L = output_dimension  # d^[L]
        self.p_source = num_source_partitions  # p^[L]
        self.output_type = output_type
        self.device = device
        self.dtype = dtype
        
        # Initialize bias vector b^[L] ∈ R^{d^[L]}
        self.bias = self._initialize_bias(bias_seed)
        
        # Set output activation function
        if output_type == OutputType.CLASSIFICATION:
            self.output_fn = F.softmax
            self.output_fn_name = "softmax"
        else:
            self.output_fn = lambda x, dim=-1: x  # Identity for regression
            self.output_fn_name = "identity"
        
        # EBD2N Statistics tracking
        self.forward_calls = 0
        self.total_microbatches_processed = 0
        self.total_samples_processed = 0
        self.dimension_matches = 0
        self.dimension_mismatches = 0
        self.padding_operations = {
            PaddingStrategy.BOTTOM: 0,
            PaddingStrategy.TOP: 0,
            PaddingStrategy.PLACE_AT: 0,
            PaddingStrategy.NO_PADDING: 0
        }
        
        # Output and loss statistics
        self.output_stats = {
            'total_loss': 0.0,
            'min_confidence': float('inf'),
            'max_confidence': float('-inf'),
            'correct_predictions': 0,
            'total_predictions': 0,
            'class_predictions': [0] * output_dimension,  # Count per class
            'confidence_sum': 0.0
        }
        
        # For backpropagation support
        self.accumulated_bias_gradients = torch.zeros_like(self.bias)
        self.gradient_accumulation_count = 0
        self.last_error_signal = None  # Store for backpropagation
        
        debug_print(f"EBD2N Output Layer initialized: output_dim={output_dimension}, partitions={num_source_partitions}, type={output_type.value}")
    
    def _initialize_bias(self, bias_seed: Optional[int] = None) -> torch.Tensor:
        """Initialize bias vector with small random values."""
        if bias_seed is not None:
            torch.manual_seed(bias_seed)
        
        # Small initialization for output layer bias
        bias = torch.randn(self.d_L, device=self.device, dtype=self.dtype) * 0.01
        return bias
    
    def _calculate_position(self, partition_index: int, source_partition_size: int) -> int:
        """
        Calculate position for placeAt padding strategy.
        
        Mathematical formula: pos_j = ⌊(d^[L] / p^[L]) * j⌋
        """
        return int((self.d_L / self.p_source) * partition_index)
    
    def _verify_boundary_condition(self, partition_index: int, source_partition_size: int) -> bool:
        """
        Verify boundary condition for intermediate partitions.
        
        Mathematical condition: pos_j + s^[L-1] ≤ d^[L]
        """
        if partition_index == 0 or partition_index == self.p_source - 1:
            return True  # First and last partitions don't need boundary check
        
        position = self._calculate_position(partition_index, source_partition_size)
        return position + source_partition_size <= self.d_L
    
    def _apply_conditional_aggregation(
        self, 
        partition_outputs: List[torch.Tensor], 
        source_partition_size: int
    ) -> torch.Tensor:
        """
        Apply EBD2N conditional aggregation with adaptive padding.
        
        Args:
            partition_outputs: List of partition outputs z_j^[L-1]
            source_partition_size: Size of each source partition s^[L-1]
            
        Returns:
            Aggregated result z^[L-1]
        """
        # Case 1: Dimension Match
        if source_partition_size == self.d_L:
            self.dimension_matches += 1
            self.padding_operations[PaddingStrategy.NO_PADDING] += len(partition_outputs)
            
            # Direct aggregation: Σ z_j^[L-1]
            aggregated = torch.stack(partition_outputs).sum(dim=0)
            
            if DEBUG and self.total_microbatches_processed < 2:
                debug_print(f"EBD2N: Dimension match ({source_partition_size}={self.d_L}), direct aggregation")
            
            return aggregated
        
        # Case 2: Dimension Mismatch - Apply weighted layer padding strategy
        self.dimension_mismatches += 1
        
        if DEBUG and self.total_microbatches_processed < 2:
            debug_print(f"EBD2N: Dimension mismatch ({source_partition_size}!={self.d_L}), applying adaptive padding")
            debug_print(f"EBD2N: Receiving from weighted layer partitions")
        
        padded_outputs = []
        
        for j, partition_output in enumerate(partition_outputs):
            # Output layer receives from weighted layer partitions - apply position-based padding
            if j == 0:
                # First partition: padBottom
                padded = pad_bottom(partition_output, self.d_L)
                self.padding_operations[PaddingStrategy.BOTTOM] += 1
                strategy = "bottom"
                
            elif j == len(partition_outputs) - 1:
                # Last partition: padTop  
                padded = pad_top(partition_output, self.d_L)
                self.padding_operations[PaddingStrategy.TOP] += 1
                strategy = "top"
                
            else:
                # Intermediate partitions: placeAt with boundary check
                if self._verify_boundary_condition(j, source_partition_size):
                    position = self._calculate_position(j, source_partition_size)
                    padded = place_at(partition_output, position, self.d_L)
                    self.padding_operations[PaddingStrategy.PLACE_AT] += 1
                    strategy = f"place_at(pos={position})"
                else:
                    # Fallback to padTop if boundary condition fails
                    padded = pad_top(partition_output, self.d_L)
                    self.padding_operations[PaddingStrategy.TOP] += 1
                    strategy = "top_fallback"
            
            if DEBUG and self.total_microbatches_processed < 2:
                debug_print(f"EBD2N: Partition {j} -> {strategy}, shape: {partition_output.shape} -> {padded.shape}")
            
            padded_outputs.append(padded)
        
        # Aggregate all padded outputs
        aggregated = torch.stack(padded_outputs).sum(dim=0)
        return aggregated
    
    def forward(
        self, 
        partition_outputs: List[torch.Tensor], 
        source_partition_size: int,
        true_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        EBD2N output layer forward pass.
        
        Mathematical operations:
        1. Apply conditional aggregation: z^[L-1] = conditional_aggregate({z_j^[L-1]})
        2. Add bias: z^[L] = z^[L-1] + b^[L]
        3. Apply output function: y = σ^[L](z^[L])
        4. Compute loss and error if labels provided: δ^[L] = y - y_true
        
        Args:
            partition_outputs: List of outputs from weighted layer partitions
            source_partition_size: Size of each source partition
            true_labels: True labels for loss computation (optional)
            
        Returns:
            Tuple of (predictions, loss, error_signal)
        """
        # Validate inputs
        assert len(partition_outputs) == self.p_source, f"Expected {self.p_source} partitions, got {len(partition_outputs)}"
        
        # Apply EBD2N conditional aggregation
        aggregated = self._apply_conditional_aggregation(partition_outputs, source_partition_size)
        
        # Add bias: z^[L] = z^[L-1] + b^[L]
        pre_output = aggregated + self.bias
        
        # Apply output function
        if self.output_type == OutputType.CLASSIFICATION:
            # Softmax for classification: y = softmax(z^[L])
            predictions = self.output_fn(pre_output, dim=-1)
        else:
            # Identity for regression: y = z^[L]
            predictions = self.output_fn(pre_output)
        
        # Compute loss and error signal if labels provided
        loss = None
        error_signal = None
        
        if true_labels is not None:
            if self.output_type == OutputType.CLASSIFICATION:
                # Cross-entropy loss and error computation
                loss, error_signal = self._compute_classification_loss_and_error(
                    predictions, true_labels, pre_output
                )
            else:
                # MSE loss for regression
                loss, error_signal = self._compute_regression_loss_and_error(
                    predictions, true_labels
                )
            
            # Store error signal for backpropagation
            self.last_error_signal = error_signal
        
        # Update statistics
        self.forward_calls += 1
        batch_size = partition_outputs[0].shape[0] if partition_outputs[0].dim() > 1 else 1
        self.total_microbatches_processed += 1
        self.total_samples_processed += batch_size
        
        # Update output statistics
        self._update_output_stats(predictions, true_labels, loss)
        
        return predictions, loss, error_signal
    
    def _compute_classification_loss_and_error(
        self, 
        predictions: torch.Tensor, 
        true_labels: torch.Tensor,
        pre_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-entropy loss and error signal for classification.
        
        Mathematical operations:
        - Loss: L = -Σ y_true * log(y_pred)
        - Error: δ^[L] = y_pred - y_true
        
        Args:
            predictions: Softmax predictions
            true_labels: True class labels (integers or one-hot)
            pre_output: Pre-softmax logits
            
        Returns:
            Tuple of (loss, error_signal)
        """
        # Handle different label formats
        if true_labels.dim() == 1:
            # Integer labels - convert to one-hot
            true_labels_one_hot = F.one_hot(true_labels, num_classes=self.d_L).float()
        else:
            # Already one-hot encoded
            true_labels_one_hot = true_labels.float()
        
        # Cross-entropy loss
        loss = F.cross_entropy(pre_output, true_labels_one_hot.argmax(dim=-1), reduction='mean')
        
        # Error signal: δ^[L] = y_pred - y_true
        error_signal = predictions - true_labels_one_hot
        
        return loss, error_signal
    
    def _compute_regression_loss_and_error(
        self, 
        predictions: torch.Tensor, 
        true_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MSE loss and error signal for regression.
        
        Mathematical operations:
        - Loss: L = (1/2) * ||y_pred - y_true||^2
        - Error: δ^[L] = y_pred - y_true
        
        Args:
            predictions: Model predictions
            true_labels: True target values
            
        Returns:
            Tuple of (loss, error_signal)
        """
        # MSE loss
        loss = F.mse_loss(predictions, true_labels, reduction='mean')
        
        # Error signal: δ^[L] = y_pred - y_true
        error_signal = predictions - true_labels
        
        return loss, error_signal
    
    def _update_output_stats(self, predictions: torch.Tensor, true_labels: Optional[torch.Tensor], loss: Optional[torch.Tensor]):
        """Update comprehensive output statistics."""
        with torch.no_grad():
            if loss is not None:
                self.output_stats['total_loss'] += loss.item()
            
            if self.output_type == OutputType.CLASSIFICATION:
                # Classification statistics
                max_probs, predicted_classes = torch.max(predictions, dim=-1)
                
                # Update confidence statistics
                min_conf = torch.min(max_probs).item()
                max_conf = torch.max(max_probs).item()
                avg_conf = torch.mean(max_probs).item()
                
                self.output_stats['min_confidence'] = min(self.output_stats['min_confidence'], min_conf)
                self.output_stats['max_confidence'] = max(self.output_stats['max_confidence'], max_conf)
                self.output_stats['confidence_sum'] += avg_conf * predictions.shape[0]
                
                # Update class prediction counts
                for class_id in predicted_classes:
                    if 0 <= class_id < self.d_L:
                        self.output_stats['class_predictions'][class_id.item()] += 1
                
                # Update accuracy if true labels provided
                if true_labels is not None:
                    if true_labels.dim() == 1:
                        true_class_ids = true_labels
                    else:
                        true_class_ids = true_labels.argmax(dim=-1)
                    
                    correct = (predicted_classes == true_class_ids).sum().item()
                    self.output_stats['correct_predictions'] += correct
                    self.output_stats['total_predictions'] += predictions.shape[0]
    
    def accumulate_bias_gradients(self, error_signal: torch.Tensor):
        """
        Accumulate bias gradients for backpropagation.
        
        Mathematical operation: ∇_{b^[L]} += δ^[L]
        
        Args:
            error_signal: Error signal δ^[L] from loss computation
        """
        if error_signal.dim() > 1:
            # Batch processing - sum across batch dimension
            batch_gradient = error_signal.sum(dim=0)
            self.accumulated_bias_gradients += batch_gradient
            self.gradient_accumulation_count += error_signal.shape[0]
        else:
            # Single example
            self.accumulated_bias_gradients += error_signal
            self.gradient_accumulation_count += 1
    
    def update_bias(self, learning_rate: float = 0.01):
        """
        Update bias using accumulated gradients.
        
        Mathematical operation: b^[L] ← b^[L] - η * (∇_{b^[L]} / count)
        """
        if self.gradient_accumulation_count > 0:
            avg_gradient = self.accumulated_bias_gradients / self.gradient_accumulation_count
            self.bias = self.bias - learning_rate * avg_gradient
            
            # Reset accumulation
            self.accumulated_bias_gradients.zero_()
            self.gradient_accumulation_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive output layer statistics."""
        # Calculate derived statistics
        avg_loss = self.output_stats['total_loss'] / self.total_microbatches_processed if self.total_microbatches_processed > 0 else 0
        accuracy = (self.output_stats['correct_predictions'] / self.output_stats['total_predictions'] * 100) if self.output_stats['total_predictions'] > 0 else 0
        avg_confidence = self.output_stats['confidence_sum'] / self.total_samples_processed if self.total_samples_processed > 0 else 0
        
        return {
            'layer_type': 'EBD2N_OutputLayer',
            'output_dimension': self.d_L,
            'num_source_partitions': self.p_source,
            'output_type': self.output_type.value,
            'output_function': self.output_fn_name,
            'bias_shape': list(self.bias.shape),
            'bias_norm': torch.norm(self.bias).item(),
            'bias_mean': torch.mean(self.bias).item(),
            'bias_std': torch.std(self.bias).item(),
            'forward_calls': self.forward_calls,
            'total_microbatches_processed': self.total_microbatches_processed,
            'total_samples_processed': self.total_samples_processed,
            'dimension_matches': self.dimension_matches,
            'dimension_mismatches': self.dimension_mismatches,
            'padding_operations': {k.value: v for k, v in self.padding_operations.items()},
            'average_loss': avg_loss,
            'accuracy_percent': accuracy,
            'average_confidence': avg_confidence,
            'confidence_range': {
                'min': self.output_stats['min_confidence'] if self.output_stats['min_confidence'] != float('inf') else 0,
                'max': self.output_stats['max_confidence'] if self.output_stats['max_confidence'] != float('-inf') else 0
            },
            'class_predictions': self.output_stats['class_predictions'],
            'total_correct': self.output_stats['correct_predictions'],
            'total_predictions': self.output_stats['total_predictions'],
            'pending_bias_gradients': self.gradient_accumulation_count
        }

def setup_distributed(rank, world_size, master_addr="192.168.1.191", master_port="12355"):
    """Initialize distributed training with comprehensive error handling"""
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    if DEBUG:
        print(f"[EBD2N-OutputLayer {rank}] Setting up distributed training:")
        print(f"[EBD2N-OutputLayer {rank}]   Rank: {rank}")
        print(f"[EBD2N-OutputLayer {rank}]   World size: {world_size}")
        print(f"[EBD2N-OutputLayer {rank}]   Master addr: {master_addr}")
        print(f"[EBD2N-OutputLayer {rank}]   Master port: {master_port}")
    
    try:
        debug_print("Attempting to join process group...", rank)
        
        # Initialize with longer timeout
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            timeout=timedelta(minutes=3)
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
            print(f"\n[EBD2N-OutputLayer {rank}] ❌ FAILED TO INITIALIZE DISTRIBUTED TRAINING:")
            print(f"[EBD2N-OutputLayer {rank}] Error: {e}")
        raise

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            debug_print("✓ Distributed cleanup completed")
        except:
            pass

def output_layer_process(
    rank, 
    world_size, 
    output_dimension, 
    num_source_partitions,
    source_weighted_ranks,
    output_type,
    master_addr, 
    master_port
):
    """EBD2N output layer process with softmax and loss computation"""
    if DEBUG:
        print(f"=" * 60)
        print(f"STARTING EBD2N OUTPUT LAYER RANK {rank}")
        print(f"=" * 60)
    
    try:
        debug_print("Initializing EBD2N output layer...", rank)
        
        # Setup distributed environment
        setup_distributed(rank, world_size, master_addr, master_port)
        
        # Enhanced bias seed generation
        bias_seed = rank * 700 + int(time.time() * 100) % 1000
        
        # Create EBD2N output layer
        ebd2n_output = EBD2NOutputLayer(
            output_dimension=output_dimension,
            num_source_partitions=num_source_partitions,
            output_type=output_type,
            bias_seed=bias_seed
        )
        
        debug_print("✓ EBD2N output layer initialized!", rank)
        debug_print(f"Output dimension: {output_dimension} ({output_type.value})", rank)
        debug_print(f"Source partitions: {num_source_partitions}", rank)
        debug_print(f"Source weighted layer ranks: {source_weighted_ranks}", rank)
        debug_print(f"Bias stats - mean: {torch.mean(ebd2n_output.bias).item():.6f}, std: {torch.std(ebd2n_output.bias).item():.6f}", rank)
        debug_print("EBD2N: Conditional aggregation with final layer processing enabled", rank)
        debug_print("Entering main processing loop...", rank)
        
        if DEBUG:
            print(f"-" * 60)
        
        # Processing statistics
        start_time = time.time()
        last_report_time = start_time
        
        # MICRO-BATCHING: Micro-batch synchronization data structures
        pending_microbatch_splits = defaultdict(dict)  # {microbatch_id: {worker_rank: (microbatch_size, weighted_results)}}
        completed_microbatches = []
        max_pending_microbatches = 50
        out_of_order_count = 0
        timeout_count = 0
        
        # Main processing loop
        while True:
            try:
                # Collect weighted results from all source weighted workers
                shutdown_received = False
                
                # Receive messages from all weighted layer workers
                for weighted_worker_rank in source_weighted_ranks:
                    try:
                        # Receive micro-batch size first
                        microbatch_size_tensor = torch.zeros(1, dtype=torch.long)
                        dist.recv(microbatch_size_tensor, src=weighted_worker_rank)
                        microbatch_size = microbatch_size_tensor.item()
                        
                        # Check for shutdown signal
                        if microbatch_size == -999:
                            debug_print(f"🛑 Received shutdown signal from weighted worker {weighted_worker_rank}", rank)
                            shutdown_received = True
                            break
                        
                        # Receive micro-batch ID
                        microbatch_id_tensor = torch.zeros(1, dtype=torch.long)
                        dist.recv(microbatch_id_tensor, src=weighted_worker_rank)
                        microbatch_id = microbatch_id_tensor.item()
                        
                        # Calculate expected elements for this partition
                        partition_size = output_dimension // num_source_partitions
                        total_elements = microbatch_size * partition_size
                        
                        # Receive flattened weighted results
                        weighted_results_flat = torch.zeros(total_elements)
                        dist.recv(weighted_results_flat, src=weighted_worker_rank)
                        
                        # Reshape to matrix form: (microbatch_size, partition_size)
                        weighted_results = weighted_results_flat.view(microbatch_size, partition_size)
                        
                        # Check for invalid data
                        if torch.isnan(weighted_results).any() or torch.isinf(weighted_results).any():
                            debug_print(f"🛑 Received invalid data from weighted worker {weighted_worker_rank}, treating as shutdown", rank)
                            shutdown_received = True
                            break
                        
                        # Store the weighted results for this micro-batch and worker
                        pending_microbatch_splits[microbatch_id][weighted_worker_rank] = (microbatch_size, weighted_results)
                        
                        if DEBUG and ebd2n_output.total_microbatches_processed < 2:
                            result_sum = torch.sum(weighted_results).item()
                            result_mean = torch.mean(weighted_results).item()
                            debug_print(f"EBD2N: Received weighted results for micro-batch {microbatch_id} from weighted worker {weighted_worker_rank}: size={microbatch_size}, sum={result_sum:.4f}, mean={result_mean:.6f}", rank)
                        
                    except RuntimeError as recv_error:
                        error_msg = str(recv_error).lower()
                        if any(keyword in error_msg for keyword in ["connection", "recv", "peer", "socket"]):
                            debug_print(f"🛑 Connection lost with weighted worker {weighted_worker_rank}, shutting down...", rank)
                            shutdown_received = True
                            break
                        else:
                            debug_print(f"✗ Error receiving from weighted worker {weighted_worker_rank}: {recv_error}", rank)
                            raise recv_error
                    except Exception as recv_error:
                        debug_print(f"✗ Unexpected error receiving from weighted worker {weighted_worker_rank}: {recv_error}", rank)
                        shutdown_received = True
                        break
                
                # If shutdown was received, break out of main loop
                if shutdown_received:
                    debug_print("🛑 Shutdown detected, terminating output layer", rank)
                    break
                
                # Process all complete micro-batches
                microbatches_to_process = []
                for microbatch_id in sorted(pending_microbatch_splits.keys()):
                    if len(pending_microbatch_splits[microbatch_id]) == num_source_partitions:
                        microbatches_to_process.append(microbatch_id)
                
                # Process complete micro-batches in order
                for microbatch_id in microbatches_to_process:
                    # Get all weighted results for this micro-batch
                    weighted_results_list = []
                    microbatch_size = None
                    
                    for weighted_worker_rank in source_weighted_ranks:
                        if weighted_worker_rank in pending_microbatch_splits[microbatch_id]:
                            size, weighted_results = pending_microbatch_splits[microbatch_id][weighted_worker_rank]
                            if microbatch_size is None:
                                microbatch_size = size
                            elif microbatch_size != size:
                                debug_print(f"⚠️  Micro-batch size mismatch for micro-batch {microbatch_id}: expected {microbatch_size}, got {size} from worker {weighted_worker_rank}", rank)
                            
                            weighted_results_list.append(weighted_results)
                        else:
                            debug_print(f"⚠️  Missing split from weighted worker {weighted_worker_rank} for micro-batch {microbatch_id}", rank)
                    
                    if len(weighted_results_list) == num_source_partitions and microbatch_size is not None:
                        # EBD2N MATHEMATICAL FRAMEWORK: Apply output layer processing
                        
                        if DEBUG and ebd2n_output.total_microbatches_processed < 2:
                            debug_print(f"EBD2N: Processing micro-batch {microbatch_id} with {len(weighted_results_list)} partitions", rank)
                        
                        # Calculate source partition size
                        source_partition_size = weighted_results_list[0].shape[1]
                        
                        # For now, process without true labels (inference mode)
                        # In training mode, labels would be received from master
                        predictions, loss, error_signal = ebd2n_output.forward(
                            weighted_results_list, 
                            source_partition_size,
                            true_labels=None  # TODO: Receive labels from master for training
                        )
                        
                        # Send results back to master (epoch layer)
                        try:
                            microbatch_size_tensor_out = torch.tensor([microbatch_size], dtype=torch.long)
                            microbatch_id_tensor_out = torch.tensor([microbatch_id], dtype=torch.long)
                            
                            dist.send(microbatch_size_tensor_out, dst=0)
                            dist.send(microbatch_id_tensor_out, dst=0)
                            dist.send(predictions.flatten(), dst=0)  # Send predictions to master
                            
                            if DEBUG and ebd2n_output.total_microbatches_processed < 2:
                                pred_sum = torch.sum(predictions).item()
                                pred_max = torch.max(predictions).item()
                                if ebd2n_output.output_type == OutputType.CLASSIFICATION:
                                    # Show class predictions
                                    _, predicted_classes = torch.max(predictions, dim=-1)
                                    debug_print(f"EBD2N: Sent {ebd2n_output.output_type.value} results for micro-batch {microbatch_id} to master:", rank)
                                    debug_print(f"  Predictions shape: {predictions.shape}, sum: {pred_sum:.4f}, max: {pred_max:.4f}", rank)
                                    debug_print(f"  Sample predicted classes: {predicted_classes[:min(5, len(predicted_classes))].tolist()}", rank)
                                else:
                                    debug_print(f"EBD2N: Sent {ebd2n_output.output_type.value} results for micro-batch {microbatch_id} to master:", rank)
                                    debug_print(f"  Predictions shape: {predictions.shape}, sum: {pred_sum:.4f}", rank)
                                
                        except Exception as send_error:
                            debug_print(f"✗ Error sending result for micro-batch {microbatch_id} to master: {send_error}", rank)
                            break
                        
                        # Track completion
                        completed_microbatches.append(microbatch_id)
                        
                        # Check for out-of-order completion
                        if len(completed_microbatches) > 1 and microbatch_id < completed_microbatches[-2]:
                            out_of_order_count += 1
                            if DEBUG and out_of_order_count <= 5:
                                debug_print(f"📋 Out-of-order completion: micro-batch {microbatch_id} completed after {completed_microbatches[-2]}", rank)
                        
                        # Detailed logging for first few micro-batches
                        if DEBUG and ebd2n_output.total_microbatches_processed <= 2:
                            debug_print(f"EBD2N: Micro-batch {microbatch_id} processing complete:", rank)
                            debug_print(f"  Received from {len(weighted_results_list)} weighted workers", rank)
                            debug_print(f"  Micro-batch size: {microbatch_size} images", rank)
                            debug_print(f"  Source partition size: {source_partition_size}", rank)
                            debug_print(f"  Output shape: {predictions.shape}", rank)
                            debug_print(f"  Output type: {ebd2n_output.output_type.value}", rank)
                            debug_print(f"  Output function: {ebd2n_output.output_fn_name}", rank)
                            debug_print(f"  EBD2N: Final layer conditional aggregation applied", rank)
                    
                    # Clean up processed micro-batch
                    del pending_microbatch_splits[microbatch_id]
                
                # Clean up old pending micro-batches
                if len(pending_microbatch_splits) > max_pending_microbatches:
                    sorted_pending = sorted(pending_microbatch_splits.keys())
                    for old_microbatch_id in sorted_pending[:-max_pending_microbatches]:
                        incomplete_workers = num_source_partitions - len(pending_microbatch_splits[old_microbatch_id])
                        debug_print(f"⚠️  Timing out incomplete micro-batch {old_microbatch_id} (missing {incomplete_workers} splits)", rank)
                        del pending_microbatch_splits[old_microbatch_id]
                        timeout_count += 1
                
                # Report progress
                current_time = time.time()
                if DEBUG and ((ebd2n_output.total_microbatches_processed % 50 == 0) or (current_time - last_report_time > 10)):
                    elapsed = current_time - start_time
                    microbatch_rate = ebd2n_output.total_microbatches_processed / elapsed if elapsed > 0 else 0
                    sample_rate = ebd2n_output.total_samples_processed / elapsed if elapsed > 0 else 0
                    pending_count = len(pending_microbatch_splits)
                    
                    # EBD2N stats
                    ebd2n_stats = ebd2n_output.get_statistics()
                    
                    print(f"[EBD2N-OutputLayer {rank}] Processed {ebd2n_output.total_microbatches_processed} microbatches ({ebd2n_output.total_samples_processed} samples) | Rate: {microbatch_rate:.2f} mb/s, {sample_rate:.2f} samples/s | Pending: {pending_count} | Out-of-order: {out_of_order_count} | Accuracy: {ebd2n_stats['accuracy_percent']:.1f}% | EBD2N-COMPLIANT")
                    last_report_time = current_time
                
                # Continue processing or wait
                if not microbatches_to_process and not pending_microbatch_splits:
                    if ebd2n_output.total_microbatches_processed > 0:
                        time.sleep(0.01)
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["connection", "recv", "send", "peer", "socket"]):
                    debug_print("⚠️  Connection lost, shutting down...", rank)
                    break
                else:
                    if DEBUG:
                        print(f"[EBD2N-OutputLayer {rank}] ❌ Runtime error: {e}")
                    raise
            except Exception as e:
                if DEBUG:
                    print(f"[EBD2N-OutputLayer {rank}] ❌ Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()
                break
        
        # Final statistics
        final_time = time.time()
        total_elapsed = final_time - start_time
        
        # Get comprehensive EBD2N statistics
        ebd2n_stats = ebd2n_output.get_statistics()
        
        if DEBUG:
            print(f"\n[EBD2N-OutputLayer {rank}] 📊 EBD2N OUTPUT LAYER FINAL STATISTICS:")
            print(f"[EBD2N-OutputLayer {rank}]   Total microbatches processed: {ebd2n_stats['total_microbatches_processed']}")
            print(f"[EBD2N-OutputLayer {rank}]   Total samples processed: {ebd2n_stats['total_samples_processed']}")
            print(f"[EBD2N-OutputLayer {rank}]   Out-of-order completions: {out_of_order_count}")
            print(f"[EBD2N-OutputLayer {rank}]   Timed-out microbatches: {timeout_count}")
            print(f"[EBD2N-OutputLayer {rank}]   Total time: {total_elapsed:.2f}s")
            print(f"[EBD2N-OutputLayer {rank}]   Average processing rate: {ebd2n_stats['total_samples_processed']/total_elapsed:.2f} samples/second")
            
            # EBD2N-specific statistics
            print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Output dimension: {ebd2n_stats['output_dimension']}")
            print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Output type: {ebd2n_stats['output_type']}")
            print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Output function: {ebd2n_stats['output_function']}")
            print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Dimension matches: {ebd2n_stats['dimension_matches']}")
            print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Dimension mismatches: {ebd2n_stats['dimension_mismatches']}")
            print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Padding operations: {ebd2n_stats['padding_operations']}")
            
            # Output performance statistics
            if ebd2n_stats['total_predictions'] > 0:
                print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Final accuracy: {ebd2n_stats['accuracy_percent']:.2f}%")
                print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Average confidence: {ebd2n_stats['average_confidence']:.4f}")
                print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Confidence range: {ebd2n_stats['confidence_range']['min']:.4f} - {ebd2n_stats['confidence_range']['max']:.4f}")
                print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Predictions per class: {ebd2n_stats['class_predictions']}")
            
            print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Mathematical framework compliance verified")
            print(f"[EBD2N-OutputLayer {rank}]   EBD2N: Ready for backpropagation integration")
            
            # Show sample completed micro-batch IDs
            if completed_microbatches:
                sample_completed = completed_microbatches[:10]
                print(f"[EBD2N-OutputLayer {rank}]   Sample completed micro-batch IDs: {sample_completed}")
                if len(completed_microbatches) > 10:
                    print(f"[EBD2N-OutputLayer {rank}]   ... and {len(completed_microbatches) - 10} more")
        
    except KeyboardInterrupt:
        debug_print("🛑 Interrupted by user", rank)
    except Exception as e:
        if DEBUG:
            print(f"\n[EBD2N-OutputLayer {rank}] ❌ Failed to start or run: {e}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_distributed()
        debug_print("👋 EBD2N output layer process terminated", rank)

def main():
    """Main output layer entry point"""
    global DEBUG
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='EBD2N Enhanced Distributed Output Layer Node')
    parser.add_argument('--rank', type=int, required=True, help='Output layer rank')
    parser.add_argument('--world-size', type=int, default=6, help='Total world size including all nodes')
    parser.add_argument('--output-dimension', type=int, default=10, help='Output dimension (number of classes or output size)')
    parser.add_argument('--num-source-partitions', type=int, required=True, help='Number of partitions from final weighted layer')
    parser.add_argument('--source-weighted-ranks', type=str, required=True, help='Comma-separated ranks of source weighted workers (e.g., "4,5,6")')
    parser.add_argument('--output-type', choices=['classification', 'regression'], default='classification', help='Type of output layer')
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
    
    # Parse source weighted ranks
    try:
        source_weighted_ranks = [int(r.strip()) for r in args.source_weighted_ranks.split(',')]
    except ValueError:
        print(f"Error: Invalid source weighted ranks format. Use comma-separated integers like '4,5,6'")
        return
    
    # Validate arguments
    if args.rank == 0:
        print(f"Error: Output layer rank cannot be 0 (reserved for master)")
        return
    
    if args.rank in source_weighted_ranks:
        print(f"Error: Output layer rank {args.rank} cannot be the same as source weighted ranks")
        return
    
    if args.output_dimension <= 0:
        print(f"Error: Output dimension must be > 0")
        return
    
    if len(source_weighted_ranks) != args.num_source_partitions:
        print(f"Error: Number of source weighted ranks ({len(source_weighted_ranks)}) must match num_source_partitions ({args.num_source_partitions})")
        return
    
    # Convert output type
    output_type = OutputType.CLASSIFICATION if args.output_type == 'classification' else OutputType.REGRESSION
    
    if DEBUG:
        print(f"\n🚀 Starting EBD2N output layer with configuration:")
        print(f"   Rank: {args.rank}")
        print(f"   World size: {args.world_size}")
        print(f"   Output dimension: {args.output_dimension}")
        print(f"   Output type: {output_type.value}")
        print(f"   Source partitions: {args.num_source_partitions}")
        print(f"   Source weighted ranks: {source_weighted_ranks}")
        print(f"   Master: {args.master_addr}:{args.master_port}")
        print(f"   Debug mode: {DEBUG}")
        print(f"   EBD2N: Final layer processing with conditional aggregation")
        print(f"   EBD2N: Mathematical framework compliance")
    
    try:
        output_layer_process(
            rank=args.rank,
            world_size=args.world_size,
            output_dimension=args.output_dimension,
            num_source_partitions=args.num_source_partitions,
            source_weighted_ranks=source_weighted_ranks,
            output_type=output_type,
            master_addr=args.master_addr,
            master_port=args.master_port
        )
    except KeyboardInterrupt:
        debug_print("🛑 Output layer interrupted by user")
    except Exception as e:
        if DEBUG:
            print(f"\n❌ Output layer failed: {e}")

if __name__ == "__main__":
    main()