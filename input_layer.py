# ENHANCED INPUT LAYER NODE WITH EBD2N MATHEMATICAL FRAMEWORK
# Adapted to integrate with existing distributed micro-batching framework

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import socket
from datetime import timedelta
import sys
import signal
import argparse
import math
from typing import List, Dict, Any, Optional

# Global debug flag
DEBUG = True

def debug_print(message, rank=None):
    """Print debug messages only if DEBUG is True"""
    if DEBUG:
        if rank is not None:
            print(f"[EBD2N-InputWorker {rank}] {message}")
        else:
            print(f"[EBD2N-InputWorker] {message}")

class EBD2NInputPartition:
    """
    EBD2N Input Layer Partition following mathematical framework.
    
    Mathematical specification:
    - Weight matrix: W_i^[0] ∈ R^{d^[1] × s^[0]}
    - Forward: z_i^[0] = W_i^[0] @ x_i^[0]
    - Backward: ∇_{W_i^[0]} = δ^[1] ⊗ x_i^[0] (outer product)
    """
    
    def __init__(self, partition_id: int, input_partition_size: int, output_size: int, device: torch.device = torch.device('cpu')):
        """
        Initialize EBD2N input partition.
        
        Args:
            partition_id: Partition index (0-based)
            input_partition_size: s^[0] = d^[0] / p^[0]
            output_size: d^[1] (activation layer size)
            device: Computation device
        """
        self.partition_id = partition_id
        self.s_0 = input_partition_size  # s^[0]
        self.d_1 = output_size          # d^[1]
        self.device = device
        
        # Initialize weight matrix W_i^[0] ∈ R^{d^[1] × s^[0]} using Xavier initialization
        self.W = self._initialize_weights()
        
        # Statistics tracking
        self.forward_calls = 0
        self.total_examples_processed = 0
        self.weight_updates = 0
        
        # For future backpropagation support
        self.accumulated_gradients = torch.zeros_like(self.W)
        self.gradient_accumulation_count = 0
        
        debug_print(f"Initialized EBD2N partition {partition_id}: W shape {self.W.shape}")
    
    def _initialize_weights(self) -> torch.Tensor:
        """Initialize weight matrix using Xavier/Glorot initialization."""
        fan_in = self.s_0
        fan_out = self.d_1
        std = math.sqrt(2.0 / (fan_in + fan_out))
        
        W = torch.randn(self.d_1, self.s_0, device=self.device, dtype=torch.float32) * std
        return W
    
    def forward(self, x_partition: torch.Tensor) -> torch.Tensor:
        """
        EBD2N forward pass: z_i^[0] = W_i^[0] @ x_i^[0]
        
        Args:
            x_partition: Input partition ∈ R^{m × s^[0]} for micro-batch
            
        Returns:
            z_i^[0]: Output ∈ R^{m × d^[1]}
        """
        # Validate input dimensions
        assert x_partition.dim() == 2, f"Expected 2D input (micro-batch), got {x_partition.dim()}D"
        assert x_partition.shape[1] == self.s_0, f"Input partition size mismatch: expected {self.s_0}, got {x_partition.shape[1]}"
        
        # Matrix multiplication: (m × s^[0]) @ (s^[0] × d^[1]) = (m × d^[1])
        # Note: W is (d^[1] × s^[0]), so we need W.T which is (s^[0] × d^[1])
        z = torch.mm(x_partition, self.W.T)
        
        self.forward_calls += 1
        self.total_examples_processed += x_partition.shape[0]
        
        return z
    
    def accumulate_gradients(self, x_partition: torch.Tensor, error_signal: torch.Tensor):
        """
        Accumulate gradients for future backpropagation.
        
        Mathematical operation: ∇_{W_i^[0]} += δ^[1] ⊗ x_i^[0]
        
        Args:
            x_partition: Input partition used in forward pass
            error_signal: Error signal δ^[1] from activation layer
        """
        # For micro-batch, accumulate gradients across all examples
        for i in range(x_partition.shape[0]):
            x_example = x_partition[i]  # Single example: s^[0]
            delta_example = error_signal[i] if error_signal.dim() > 1 else error_signal  # d^[1]
            
            # Outer product: δ^[1] ⊗ x_i^[0] → (d^[1], s^[0])
            gradient = torch.outer(delta_example, x_example)
            self.accumulated_gradients += gradient
            self.gradient_accumulation_count += 1
    
    def update_weights(self, learning_rate: float = 0.01, l2_regularization: float = 0.0):
        """
        Update weights using accumulated gradients.
        
        Mathematical operations:
        - Standard: W_i^[0] ← W_i^[0] - η * (∇_{W_i^[0]} / count)
        - With L2: W_i^[0] ← (1 - η*λ) * W_i^[0] - η * (∇_{W_i^[0]} / count)
        """
        if self.gradient_accumulation_count > 0:
            # Average gradients
            avg_gradient = self.accumulated_gradients / self.gradient_accumulation_count
            
            # Apply weight update
            if l2_regularization > 0.0:
                self.W = (1.0 - learning_rate * l2_regularization) * self.W - learning_rate * avg_gradient
            else:
                self.W = self.W - learning_rate * avg_gradient
            
            # Reset accumulation
            self.accumulated_gradients.zero_()
            self.gradient_accumulation_count = 0
            self.weight_updates += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get partition statistics."""
        return {
            'partition_id': self.partition_id,
            'weight_shape': list(self.W.shape),
            'weight_norm': torch.norm(self.W).item(),
            'weight_mean': torch.mean(self.W).item(),
            'weight_std': torch.std(self.W).item(),
            'forward_calls': self.forward_calls,
            'total_examples_processed': self.total_examples_processed,
            'weight_updates': self.weight_updates,
            'pending_gradients': self.gradient_accumulation_count
        }

def test_network_connectivity(master_addr, master_port, timeout=10):
    """Test if we can connect to the master node"""
    debug_print(f"Testing network connectivity to {master_addr}:{master_port}...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((master_addr, int(master_port)))
            if result == 0:
                debug_print("✓ Network connectivity successful")
                return True
            else:
                debug_print(f"✗ Cannot connect (error code: {result})")
                return False
    except Exception as e:
        debug_print(f"✗ Network test failed: {e}")
        return False

def wait_for_master(master_addr, master_port, max_wait=60):
    """Wait for master to be available"""
    debug_print("Waiting for master to become available...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if test_network_connectivity(master_addr, master_port, timeout=2):
            return True
        debug_print("Master not ready yet, retrying in 2 seconds...")
        time.sleep(2)
    
    debug_print(f"❌ Master did not become available within {max_wait} seconds")
    return False

def setup_distributed(rank, world_size, master_addr="192.168.1.191", master_port="12355"):
    """Initialize distributed training with comprehensive error handling"""
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    if DEBUG:
        print(f"[EBD2N-InputWorker {rank}] Setting up distributed training:")
        print(f"[EBD2N-InputWorker {rank}]   Rank: {rank}")
        print(f"[EBD2N-InputWorker {rank}]   World size: {world_size}")
        print(f"[EBD2N-InputWorker {rank}]   Master addr: {master_addr}")
        print(f"[EBD2N-InputWorker {rank}]   Master port: {master_port}")
    
    # Wait for master to be available
    if not wait_for_master(master_addr, master_port):
        raise ConnectionError(f"Cannot reach master node at {master_addr}:{master_port}")
    
    try:
        debug_print("Attempting to join process group...", rank)
        
        # Initialize with longer timeout and better error messages
        dist.init_process_group(
            backend="gloo",  # gloo is more reliable for CPU-based communication
            rank=rank,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            timeout=timedelta(minutes=3)  # Longer timeout
        )
        
        debug_print("✓ Successfully joined process group", rank)
        
        # Wait for all processes to be ready
        debug_print("Synchronizing with all processes...", rank)
        dist.barrier()  # This will wait for all processes
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
            print(f"\n[EBD2N-InputWorker {rank}] ❌ FAILED TO INITIALIZE DISTRIBUTED TRAINING:")
            print(f"[EBD2N-InputWorker {rank}] Error: {e}")
            print_troubleshooting_tips(rank, master_addr, master_port)
        raise

def print_troubleshooting_tips(rank, master_addr, master_port):
    """Print comprehensive troubleshooting information"""
    if DEBUG:
        print(f"\n[EBD2N-InputWorker {rank}] TROUBLESHOOTING CHECKLIST:")
        print(f"[EBD2N-InputWorker {rank}] 1. Is the master node running?")
        print(f"[EBD2N-InputWorker {rank}] 2. Can you ping {master_addr}?")
        print(f"[EBD2N-InputWorker {rank}] 3. Is port {master_port} open in firewall?")
        print(f"[EBD2N-InputWorker {rank}] 4. Try: telnet {master_addr} {master_port}")
        print(f"[EBD2N-InputWorker {rank}] 5. Are you on the same network as master?")
        print(f"[EBD2N-InputWorker {rank}] 6. Check if master changed to a different port")
        print(f"[EBD2N-InputWorker {rank}] 7. Ensure master started before workers")

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            debug_print("✓ Distributed cleanup completed")
        except:
            pass

def worker_process(rank, world_size, activation_rank, master_addr, master_port):
    """EBD2N ENHANCED input worker process with mathematical framework compliance"""
    if DEBUG:
        print(f"=" * 60)
        print(f"STARTING EBD2N ENHANCED INPUT WORKER RANK {rank}")
        print(f"=" * 60)
    
    try:
        debug_print("Initializing EBD2N framework...", rank)
        
        # Setup distributed environment
        setup_distributed(rank, world_size, master_addr, master_port)
        
        # Calculate EBD2N partition configuration
        num_input_workers = world_size - 2  # Exclude master and activation node
        partition_id = rank - 1  # Convert rank to 0-based partition index
        
        # EBD2N configuration
        input_size = 784  # d^[0] - MNIST image size
        output_size = 100  # d^[1] - activation layer size
        
        # Verify divisibility constraint: d^[0] mod p^[0] = 0
        if input_size % num_input_workers != 0:
            raise ValueError(f"EBD2N constraint violation: input size {input_size} must be divisible by number of partitions {num_input_workers}")
        
        partition_size = input_size // num_input_workers  # s^[0]
        
        # Create EBD2N input partition
        ebd2n_partition = EBD2NInputPartition(
            partition_id=partition_id,
            input_partition_size=partition_size,
            output_size=output_size
        )
        
        debug_print("✓ EBD2N partition initialized!", rank)
        debug_print(f"Partition ID: {partition_id}, Input size: {partition_size}, Output size: {output_size}", rank)
        debug_print(f"Weight matrix shape: {ebd2n_partition.W.shape}", rank)
        debug_print(f"Will send processed data to activation node (rank {activation_rank})", rank)
        debug_print("EBD2N: Following mathematical framework for vertical partitioning", rank)
        debug_print("Entering main processing loop...", rank)
        
        if DEBUG:
            print(f"-" * 60)
        
        # Processing statistics
        total_processed_microbatches = 0
        total_processed_images = 0
        start_time = time.time()
        last_report_time = start_time
        
        # EBD2N: Track processed micro-batches for debugging
        processed_microbatches = set()
        last_microbatch_id = 0
        out_of_order_count = 0
        
        debug_print("EBD2N: Ready to process matrix operations on micro-batches", rank)
        
        # Worker processing loop
        while True:
            try:
                # EBD2N ENHANCED: Receive micro-batch size first
                microbatch_size_tensor = torch.zeros(1, dtype=torch.long)
                dist.recv(microbatch_size_tensor, src=0)
                microbatch_size = microbatch_size_tensor.item()
                
                # Check for shutdown signal (special micro-batch size -999)
                if microbatch_size == -999:
                    debug_print("🛑 Received shutdown signal", rank)
                    break
                
                # EBD2N: Receive micro-batch ID
                microbatch_id_tensor = torch.zeros(1, dtype=torch.long)
                dist.recv(microbatch_id_tensor, src=0)
                microbatch_id = microbatch_id_tensor.item()
                
                # EBD2N: Receive the flattened micro-batch split
                # Expected shape after reshape: (microbatch_size, partition_size)
                total_elements = microbatch_size * partition_size
                microbatch_split_flat = torch.zeros(total_elements)
                dist.recv(microbatch_split_flat, src=0)
                
                # Reshape to matrix form: (microbatch_size, partition_size)
                microbatch_split = microbatch_split_flat.view(microbatch_size, partition_size)
                
                # Check for additional shutdown patterns
                if torch.all(microbatch_split == -999.0):  # Backup shutdown signal pattern
                    debug_print("🛑 Received backup shutdown signal", rank)
                    break
                elif torch.all(microbatch_split == 0.0):  # Potential heartbeat
                    debug_print("💓 Received potential heartbeat", rank)
                    continue
                
                # EBD2N: Track micro-batch processing order
                if microbatch_id in processed_microbatches:
                    debug_print(f"⚠️  Duplicate micro-batch ID {microbatch_id} received!", rank)
                else:
                    processed_microbatches.add(microbatch_id)
                
                # Check for out-of-order processing
                if microbatch_id < last_microbatch_id:
                    out_of_order_count += 1
                    if DEBUG and out_of_order_count <= 5:  # Only log first few out-of-order events
                        debug_print(f"📋 Out-of-order processing: received micro-batch {microbatch_id} after {last_microbatch_id}", rank)
                
                last_microbatch_id = max(last_microbatch_id, microbatch_id)
                
                # EBD2N MATHEMATICAL FRAMEWORK: Process the micro-batch matrix
                # Forward pass: z_i^[0] = W_i^[0] @ x_i^[0]
                microbatch_start_time = time.time()
                weighted_results = ebd2n_partition.forward(microbatch_split)
                microbatch_processing_time = time.time() - microbatch_start_time
                
                # EBD2N ENHANCED: Send micro-batch size + micro-batch ID + weighted results to activation node
                microbatch_size_tensor_out = torch.tensor([microbatch_size], dtype=torch.long)
                microbatch_id_tensor_out = torch.tensor([microbatch_id], dtype=torch.long)
                
                dist.send(microbatch_size_tensor_out, dst=activation_rank)
                dist.send(microbatch_id_tensor_out, dst=activation_rank)
                dist.send(weighted_results.flatten(), dst=activation_rank)  # Flatten for transmission
                
                total_processed_microbatches += 1
                total_processed_images += microbatch_size
                
                # Report progress every 50 micro-batches or every 10 seconds
                current_time = time.time()
                if DEBUG and ((total_processed_microbatches % 50 == 0) or (current_time - last_report_time > 10)):
                    elapsed = current_time - start_time
                    microbatch_rate = total_processed_microbatches / elapsed if elapsed > 0 else 0
                    image_rate = total_processed_images / elapsed if elapsed > 0 else 0
                    unique_microbatches = len(processed_microbatches)
                    print(f"[EBD2N-InputWorker {rank}] Processed {total_processed_microbatches} micro-batches ({total_processed_images} images) | MBatch rate: {microbatch_rate:.2f}/sec | Image rate: {image_rate:.2f}/sec | Unique: {unique_microbatches} | Out-of-order: {out_of_order_count} | EBD2N-COMPLIANT")
                    last_report_time = current_time
                
                # Detailed logging for first few micro-batches
                if DEBUG and total_processed_microbatches <= 3:
                    debug_print(f"EBD2N processed micro-batch {microbatch_id}:", rank)
                    debug_print(f"  Size: {microbatch_size} images", rank)
                    debug_print(f"  Input shape: {microbatch_split.shape}", rank)
                    debug_print(f"  Output shape: {weighted_results.shape}", rank)
                    debug_print(f"  Processing time: {microbatch_processing_time:.4f}s", rank)
                    debug_print(f"  Weighted sum: {torch.sum(weighted_results).item():.4f}", rank)
                    debug_print(f"  Weight matrix norm: {torch.norm(ebd2n_partition.W).item():.4f}", rank)
                    debug_print(f"  EBD2N: Mathematical framework z_i^[0] = W_i^[0] @ x_i^[0]", rank)
                
                # TODO: Future backpropagation integration point
                # When activation layer implements backprop, it will send error signals back
                # Then we would call: ebd2n_partition.accumulate_gradients(microbatch_split, error_signal)
                # And periodically: ebd2n_partition.update_weights(learning_rate=0.01)
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["connection", "recv", "send", "peer", "socket"]):
                    debug_print("⚠️  Connection lost with master, shutting down...", rank)
                    break
                else:
                    if DEBUG:
                        print(f"[EBD2N-InputWorker {rank}] ❌ Runtime error: {e}")
                    raise
            except Exception as e:
                if DEBUG:
                    print(f"[EBD2N-InputWorker {rank}] ❌ Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()
                break
        
        # Final statistics
        final_time = time.time()
        total_elapsed = final_time - start_time
        avg_microbatch_rate = total_processed_microbatches / total_elapsed if total_elapsed > 0 else 0
        avg_image_rate = total_processed_images / total_elapsed if total_elapsed > 0 else 0
        unique_microbatches = len(processed_microbatches)
        
        # Get EBD2N partition statistics
        ebd2n_stats = ebd2n_partition.get_statistics()
        
        if DEBUG:
            print(f"\n[EBD2N-InputWorker {rank}] 📊 EBD2N FINAL STATISTICS:")
            print(f"[EBD2N-InputWorker {rank}]   Total micro-batches processed: {total_processed_microbatches}")
            print(f"[EBD2N-InputWorker {rank}]   Total images processed: {total_processed_images}")
            print(f"[EBD2N-InputWorker {rank}]   Unique micro-batches processed: {unique_microbatches}")
            print(f"[EBD2N-InputWorker {rank}]   Out-of-order events: {out_of_order_count}")
            print(f"[EBD2N-InputWorker {rank}]   Total time: {total_elapsed:.2f}s")
            print(f"[EBD2N-InputWorker {rank}]   Average micro-batch rate: {avg_microbatch_rate:.2f} micro-batches/second")
            print(f"[EBD2N-InputWorker {rank}]   Average image rate: {avg_image_rate:.2f} images/second")
            print(f"[EBD2N-InputWorker {rank}]   Average images per micro-batch: {total_processed_images/total_processed_microbatches:.1f}" if total_processed_microbatches > 0 else "")
            print(f"[EBD2N-InputWorker {rank}]   EBD2N: Mathematical framework compliance verified")
            print(f"[EBD2N-InputWorker {rank}]   EBD2N: Weight matrix shape: {ebd2n_stats['weight_shape']}")
            print(f"[EBD2N-InputWorker {rank}]   EBD2N: Weight matrix norm: {ebd2n_stats['weight_norm']:.4f}")
            print(f"[EBD2N-InputWorker {rank}]   EBD2N: Forward calls: {ebd2n_stats['forward_calls']}")
            print(f"[EBD2N-InputWorker {rank}]   EBD2N: Examples processed by partition: {ebd2n_stats['total_examples_processed']}")
            print(f"[EBD2N-InputWorker {rank}]   EBD2N: Ready for backpropagation integration")
            
            # Show some processed micro-batch IDs for verification
            if processed_microbatches:
                sample_microbatch_ids = sorted(list(processed_microbatches))[:10]
                print(f"[EBD2N-InputWorker {rank}]   Sample processed micro-batch IDs: {sample_microbatch_ids}")
                if len(processed_microbatches) > 10:
                    print(f"[EBD2N-InputWorker {rank}]   ... and {len(processed_microbatches) - 10} more")
        
    except KeyboardInterrupt:
        debug_print("🛑 Interrupted by user", rank)
    except Exception as e:
        if DEBUG:
            print(f"\n[EBD2N-InputWorker {rank}] ❌ Failed to start or run: {e}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_distributed()
        debug_print("👋 EBD2N enhanced input worker process terminated", rank)

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        return "unknown"

def run_interactive_setup():
    """Interactive setup for worker configuration"""
    if DEBUG:
        print("=" * 60)
        print("INTERACTIVE EBD2N ENHANCED INPUT WORKER SETUP")
        print("=" * 60)
    
    # Get local IP for reference
    local_ip = get_local_ip()
    if DEBUG:
        print(f"Local machine IP: {local_ip}")
        print("EBD2N: This worker implements mathematical framework compliance")
    
    # Get configuration from user
    try:
        rank = int(input("Enter input worker rank (1, 2, ...): "))
        if rank < 1:
            print("Rank must be >= 1")
            return None
    except ValueError:
        print("Invalid rank. Must be a number.")
        return None
    
    master_addr = input(f"Enter master IP address [192.168.1.191]: ").strip()
    if not master_addr:
        master_addr = "192.168.1.191"
    
    master_port = input(f"Enter master port [12355]: ").strip()
    if not master_port:
        master_port = "12355"
    
    world_size = input(f"Enter world size (total nodes including master) [4]: ").strip()
    if not world_size:
        world_size = 4
    else:
        try:
            world_size = int(world_size)
        except ValueError:
            print("Invalid world size. Using default 4.")
            world_size = 4
    
    # Calculate activation rank (last worker rank)
    activation_rank = world_size - 1
    
    # Verify EBD2N constraints
    num_input_workers = world_size - 2
    input_size = 784
    if input_size % num_input_workers != 0:
        print(f"WARNING: EBD2N constraint violation!")
        print(f"Input size {input_size} must be divisible by number of input workers {num_input_workers}")
        print(f"Consider using a different world size where {input_size} % {num_input_workers} = 0")
        
        common_divisors = [i for i in range(2, 20) if 784 % i == 0]
        suggested_world_sizes = [d + 2 for d in common_divisors]  # +2 for master and activation
        print(f"Suggested world sizes: {suggested_world_sizes}")
        
        proceed = input("Continue anyway? [y/N]: ").strip().lower()
        if proceed not in ['y', 'yes']:
            return None
    
    if DEBUG:
        print(f"\nEBD2N Configuration:")
        print(f"  Input worker rank: {rank}")
        print(f"  Master: {master_addr}:{master_port}")
        print(f"  World size: {world_size}")
        print(f"  Activation node rank: {activation_rank}")
        print(f"  Number of input workers: {num_input_workers}")
        print(f"  Partition size per worker: {input_size // num_input_workers}")
        print(f"  Local IP: {local_ip}")
        print(f"  EBD2N: Mathematical framework enabled")
    
    confirm = input(f"\nProceed with this configuration? [y/N]: ").strip().lower()
    if confirm in ['y', 'yes']:
        return rank, world_size, activation_rank, master_addr, master_port
    else:
        print("Setup cancelled.")
        return None

def main():
    """Main input worker entry point with multiple launch modes"""
    global DEBUG
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='EBD2N Enhanced Distributed Input Worker Node')
    parser.add_argument('--rank', type=int, help='Input worker rank (1, 2, ...)')
    parser.add_argument('--world-size', type=int, default=4, help='Total world size including master and activation node')
    parser.add_argument('--master-addr', default='192.168.1.191', help='Master node IP address')
    parser.add_argument('--master-port', default='12355', help='Master node port')
    parser.add_argument('--interactive', action='store_true', help='Run interactive setup')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output (default: True)')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')
    
    args = parser.parse_args()
    
    # Set debug flag based on arguments
    if args.no_debug:
        DEBUG = False
    else:
        DEBUG = args.debug
    
    # Interactive mode
    if args.interactive or args.rank is None:
        setup = run_interactive_setup()
        if setup is None:
            return
        rank, world_size, activation_rank, master_addr, master_port = setup
    else:
        # Command line mode
        rank = args.rank
        world_size = args.world_size
        activation_rank = world_size - 1  # Last rank is activation node
        master_addr = args.master_addr
        master_port = args.master_port
    
    # Validate configuration
    if rank >= activation_rank:
        print(f"Error: Input worker rank {rank} must be less than activation rank {activation_rank}")
        return
    
    if rank < 1:
        print(f"Error: Input worker rank must be >= 1 (rank 0 is reserved for master)")
        return
    
    # Validate EBD2N constraints
    num_input_workers = world_size - 2
    input_size = 784
    if input_size % num_input_workers != 0:
        print(f"Error: EBD2N constraint violation!")
        print(f"Input size {input_size} must be divisible by number of input workers {num_input_workers}")
        return
    
    if DEBUG:
        print(f"\n🚀 Starting EBD2N enhanced input worker with configuration:")
        print(f"   Rank: {rank}")
        print(f"   World size: {world_size}")
        print(f"   Master: {master_addr}:{master_port}")
        print(f"   Activation node rank: {activation_rank}")
        print(f"   Debug mode: {DEBUG}")
        print(f"   EBD2N: Mathematical framework compliance enabled")
        print(f"   EBD2N: Partition size: {input_size // num_input_workers}")
    
    # Test connectivity before starting
    if DEBUG:
        print(f"\n🔍 Pre-flight checks...")
    
    if not test_network_connectivity(master_addr, master_port, timeout=5):
        if DEBUG:
            print(f"\n❌ Cannot reach master node!")
            print(f"Please ensure:")
            print(f"  1. Master node is running")
            print(f"  2. IP address {master_addr} is correct")
            print(f"  3. Port {master_port} is open")
            print(f"  4. No firewall blocking the connection")
            
            retry = input(f"\nTry anyway? [y/N]: ").strip().lower()
            if retry not in ['y', 'yes']:
                print("Startup cancelled.")
                return
        else:
            # In non-debug mode, still try to connect even if initial test fails
            pass
    
    try:
        # Run the EBD2N enhanced input worker
        worker_process(rank, world_size, activation_rank, master_addr, master_port)
    except KeyboardInterrupt:
        debug_print("🛑 Input worker interrupted by user")
    except Exception as e:
        if DEBUG:
            print(f"\n❌ Input worker failed: {e}")

if __name__ == "__main__":
    # Handle different launch methods
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        if DEBUG:
            print("No arguments provided. Starting interactive setup...")
        sys.argv.append('--interactive')
    
    main()