# EBD2N INPUT LAYER NODE WITH DYNAMIC CONFIGURATION
# ENHANCED: Reads configuration from environment variables with fallback chain
# Priority: Command-line args → Environment variables → Hardcoded defaults
#filename input_layer.py

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

def get_env_or_default(env_var, default_value, var_type=str):
    """
    Get value from environment variable with type conversion and default fallback.
    
    Args:
        env_var: Environment variable name
        default_value: Default value if env var not set
        var_type: Type to convert to (str, int, bool)
    
    Returns:
        Value from environment or default
    """
    value = os.getenv(env_var)
    if value is None:
        return default_value
    
    try:
        if var_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif var_type == int:
            return int(value)
        else:
            return str(value)
    except (ValueError, AttributeError):
        debug_print(f"Warning: Invalid value for {env_var}='{value}', using default: {default_value}")
        return default_value

def calculate_network_topology(world_size, num_input_workers):
    """
    Calculate network topology based on world size and number of input workers.
    
    EBD2N Network Structure:
    - Rank 0: Master/Epoch node
    - Ranks 1 to num_input_workers: Input workers
    - Rank num_input_workers+1: Activation layer
    - Ranks num_input_workers+2 to world_size-2: Weighted workers (at least 1)
    - Rank world_size-1: Output layer
    
    Args:
        world_size: Total number of nodes
        num_input_workers: Number of input layer workers
    
    Returns:
        tuple: (num_input_workers, activation_rank, num_weighted_workers, output_rank)
    """
    min_world_size = num_input_workers + 4  # input_workers + master + activation + weighted(1) + output
    if world_size < min_world_size:
        raise ValueError(f"World size must be at least {min_world_size} for {num_input_workers} input workers (got {world_size})")
    
    activation_rank = num_input_workers + 1
    num_weighted_workers = world_size - num_input_workers - 3  # -3 for master, activation, output
    output_rank = world_size - 1
    
    return num_input_workers, activation_rank, num_weighted_workers, output_rank

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

def setup_distributed(rank, world_size, master_addr, master_port):
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
        
        # Use file-based initialization (most reliable on HPC/Cray)
        import pathlib
        
        # Use same init file as master
        init_file = f"/tmp/ebd2n_init_{master_port}"
        
        debug_print(f"Using file-based init: {init_file}", rank)
        
        # Initialize with file:// method (no network needed!)
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
        print(f"[EBD2N-InputWorker {rank}] 1. Is master node running at {master_addr}?")
        print(f"[EBD2N-InputWorker {rank}] 2. Can you ping {master_addr}?")
        print(f"[EBD2N-InputWorker {rank}] 3. Is port {master_port} open on master?")
        print(f"[EBD2N-InputWorker {rank}] 4. Try: telnet {master_addr} {master_port}")
        print(f"[EBD2N-InputWorker {rank}] 5. Check firewall rules")
        print(f"[EBD2N-InputWorker {rank}] 6. Verify MASTER_ADDR and MASTER_PORT environment variables")
        print(f"[EBD2N-InputWorker {rank}] 7. Ensure all nodes use the same world_size")

def cleanup_distributed(rank):
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            debug_print("✓ Distributed cleanup completed", rank)
        except:
            pass

def worker_process(rank, world_size, num_input_workers, activation_rank, master_addr, master_port):
    """Main worker process for input layer node"""
    
    try:
        print("=" * 60)
        print(f"STARTING EBD2N FIXED INPUT WORKER RANK {rank}")
        print("=" * 60)
        
        debug_print("Initializing EBD2N framework...", rank)
        
        # Setup distributed training
        setup_distributed(rank, world_size, master_addr, master_port)
        
        # Calculate partition parameters
        input_size = 784
        partition_size = input_size // num_input_workers
        partition_id = rank - 1  # 0-indexed partition ID
        
        # Initialize EBD2N partition
        activation_size = 100  # Default, should match master
        partition = EBD2NInputPartition(
            partition_id=partition_id,
            input_partition_size=partition_size,
            output_size=activation_size
        )
        
        debug_print(f"✓ Input worker initialized:", rank)
        debug_print(f"  Partition ID: {partition_id}", rank)
        debug_print(f"  Partition size: {partition_size}", rank)
        debug_print(f"  Output size: {activation_size}", rank)
        debug_print(f"  Sends to activation rank: {activation_rank}", rank)
        
        # Main processing loop
        debug_print("Entering main processing loop...", rank)
        while True:
            try:
                # Receive micro-batch metadata
                microbatch_size_tensor = torch.zeros(1, dtype=torch.long)
                microbatch_id_tensor = torch.zeros(1, dtype=torch.long)
                
                dist.recv(microbatch_size_tensor, src=0)
                dist.recv(microbatch_id_tensor, src=0)
                
                microbatch_size = int(microbatch_size_tensor.item())
                microbatch_id = int(microbatch_id_tensor.item())
                
                # Check for shutdown signal
                if microbatch_size == -999 or microbatch_id == -999:
                    debug_print("Received shutdown signal", rank)
                    break
                
                # Receive data partition
                data_partition = torch.zeros(microbatch_size * partition_size)
                dist.recv(data_partition, src=0)
                data_partition = data_partition.view(microbatch_size, partition_size)
                
                # Forward pass
                output = partition.forward(data_partition)
                
                # Send to activation layer
                dist.send(output.flatten(), dst=activation_rank)
                
                if DEBUG and microbatch_id % 10 == 0:
                    debug_print(f"Processed micro-batch {microbatch_id}: {microbatch_size} samples", rank)
                
            except Exception as e:
                debug_print(f"Error in processing loop: {e}", rank)
                break
        
        debug_print("Exiting main processing loop", rank)
        
    except KeyboardInterrupt:
        debug_print("Interrupted by user", rank)
    except Exception as e:
        debug_print(f"Worker error: {e}", rank)
        if DEBUG:
            import traceback
            traceback.print_exc()
    finally:
        cleanup_distributed(rank)
        debug_print("Worker shutdown complete", rank)

def get_local_ip():
    """Get local IP address"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        return "unknown"

def main():
    """Main input worker entry point with dynamic configuration"""
    global DEBUG
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    # Get defaults from environment variables
    default_world_size = get_env_or_default('WORLD_SIZE', 7, int)
    default_num_input_workers = get_env_or_default('NUM_INPUT_WORKERS', 2, int)
    default_master_addr = get_env_or_default('MASTER_ADDR', '10.150.0.17', str)
    default_master_port = get_env_or_default('MASTER_PORT', '12355', str)
    default_debug = get_env_or_default('DEBUG', True, bool)
    
    # Check for RANK environment variable (common in distributed setups)
    default_rank = get_env_or_default('RANK', None, int)
    
    parser = argparse.ArgumentParser(
        description='EBD2N Input Worker - Reads from environment variables or command-line',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--rank', type=int, default=default_rank,
                       help='Input worker rank (1, 2, ...) (env: RANK)')
    parser.add_argument('--world-size', type=int, default=default_world_size,
                       help='Total world size (env: WORLD_SIZE)')
    parser.add_argument('--num-input-workers', type=int, default=default_num_input_workers,
                       help='Number of input workers (env: NUM_INPUT_WORKERS)')
    parser.add_argument('--master-addr', default=default_master_addr,
                       help='Master IP address (env: MASTER_ADDR)')
    parser.add_argument('--master-port', default=default_master_port,
                       help='Master port (env: MASTER_PORT)')
    parser.add_argument('--debug', action='store_true', default=default_debug,
                       help='Enable debug output (env: DEBUG)')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug output')
    
    args = parser.parse_args()
    
    # Set debug flag
    if args.no_debug:
        DEBUG = False
    else:
        DEBUG = args.debug
    
    # Show configuration source
    if DEBUG:
        local_ip = get_local_ip()
        print("\n" + "=" * 60)
        print("CONFIGURATION SOURCE PRIORITY:")
        print("  1. Command-line arguments (highest)")
        print("  2. Environment variables")
        print("  3. Hardcoded defaults (lowest)")
        print("=" * 60)
        print(f"\nActive Configuration:")
        print(f"  RANK: {args.rank} (from env: {os.getenv('RANK', 'not set')})")
        print(f"  MASTER_ADDR: {args.master_addr} (from env: {os.getenv('MASTER_ADDR', 'not set')})")
        print(f"  MASTER_PORT: {args.master_port} (from env: {os.getenv('MASTER_PORT', 'not set')})")
        print(f"  WORLD_SIZE: {args.world_size} (from env: {os.getenv('WORLD_SIZE', 'not set')})")
        print(f"  NUM_INPUT_WORKERS: {args.num_input_workers} (from env: {os.getenv('NUM_INPUT_WORKERS', 'not set')})")
        print(f"  Local IP: {local_ip}")
        print("=" * 60 + "\n")
    
    # Validate rank
    if args.rank is None:
        print("Error: --rank is required (or set RANK environment variable)")
        print("Example: python worker.py --rank 1")
        print("Or: export RANK=1 && python worker.py")
        return
    
    if args.rank < 1:
        print(f"Error: Worker rank must be >= 1 (rank 0 is master)")
        return
    
    if args.rank >= args.world_size:
        print(f"Error: Worker rank {args.rank} must be < world size {args.world_size}")
        return
    
    # Calculate topology
    num_input_workers, activation_rank, num_weighted_workers, output_rank = calculate_network_topology(args.world_size, args.num_input_workers)
    
    # Validate this is an input worker rank
    if args.rank > num_input_workers:
        print(f"Error: Rank {args.rank} is not an input worker rank")
        print(f"Input workers for world_size {args.world_size}: ranks 1-{num_input_workers}")
        print(f"Other ranks: activation={activation_rank}, weighted={num_input_workers+2}-{output_rank-1}, output={output_rank}")
        return
    
    # Validate EBD2N constraints
    input_size = 784
    if input_size % num_input_workers != 0:
        print(f"Error: EBD2N constraint violation!")
        print(f"Input size {input_size} must be divisible by {num_input_workers} input workers")
        print(f"Partition size would be: {input_size / num_input_workers} (not an integer)")
        return
    
    if DEBUG:
        print(f"🚀 Starting EBD2N input worker:")
        print(f"   Rank: {args.rank}")
        print(f"   World size: {args.world_size}")
        print(f"   Master: {args.master_addr}:{args.master_port}")
        print(f"   Activation rank: {activation_rank}")
        print(f"   Number of input workers: {num_input_workers}")
        print(f"   Partition size: {input_size // num_input_workers}")
    
    # Test connectivity
    if DEBUG:
        print(f"\n🔍 Pre-flight checks...")
    
    if not test_network_connectivity(args.master_addr, args.master_port, timeout=5):
        if DEBUG:
            print(f"\n❌ Cannot reach master node!")
            print(f"Please ensure:")
            print(f"  1. Master node is running")
            print(f"  2. MASTER_ADDR={args.master_addr} is correct")
            print(f"  3. MASTER_PORT={args.master_port} is correct")
            print(f"  4. No firewall blocking the connection")
    
    try:
        worker_process(args.rank, args.world_size, args.num_input_workers, activation_rank, args.master_addr, args.master_port)
    except KeyboardInterrupt:
        debug_print("🛑 Input worker interrupted by user")
    except Exception as e:
        if DEBUG:
            print(f"\n❌ Input worker failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()