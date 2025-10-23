# EPOCH / MASTER NODE FOR FULL EBD2N NETWORK TOPOLOGY
# ENHANCED: Reads configuration from environment variables with fallback chain
# Priority: Command-line args → Environment variables → Hardcoded defaults

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ignite.engine import Engine, Events
import torchvision
import torchvision.transforms as transforms
import os
import time
import socket
from datetime import timedelta
import argparse

# Global debug flag
DEBUG = True

def debug_print(message, rank="Master"):
    """Print debug messages only if DEBUG is True"""
    if DEBUG:
        print(f"[{rank}] {message}")

def stats_print(message, rank="Master"):
    """Print important statistics always (regardless of DEBUG flag)"""
    print(f"[{rank}] {message}")

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

def check_port_available(port):
    """Check if port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', int(port)))
            return True
        except OSError:
            return False

def find_available_port(start_port=12355, end_port=12400):
    """Find an available port in the given range"""
    for port in range(start_port, end_port):
        if check_port_available(str(port)):
            return str(port)
    raise RuntimeError(f"No available ports found in range {start_port}-{end_port}")

def calculate_minimum_world_size(num_input_workers):
    """
    Calculate minimum world size based on network topology parameters.
    
    Args:
        num_input_workers: Number of input worker processes
        
    Returns:
        Minimum world size required
    """
    # Network topology:
    # 1 master (rank 0)
    # + num_input_workers input workers 
    # + 1 activation layer 
    # + 1 weighted worker (minimum)
    # + 1 output layer
    
    min_world_size = 1 + num_input_workers + 1 + 1 + 1
    return min_world_size  # = num_input_workers + 4

def calculate_network_topology(world_size, num_input_workers):
    """
    Calculate network topology based on world size and parameters.
    Auto-calculates number of weighted workers from remaining ranks.
    
    Returns:
        tuple: (input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank)
    """
    input_worker_ranks = list(range(1, num_input_workers + 1))  # ranks 1, 2, ..., num_input_workers
    activation_rank = num_input_workers + 1                     # rank num_input_workers+1
    
    # Calculate number of weighted workers automatically
    num_weighted_workers = world_size - num_input_workers - 3  # -3 for master, activation, output
    
    weighted_worker_ranks = list(range(activation_rank + 1, activation_rank + 1 + num_weighted_workers))
    output_rank = world_size - 1                               # last rank
    
    return input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank

def setup_distributed(rank, world_size, master_addr, master_port):
    """Initialize distributed training with enhanced error handling"""
    
    # Check if port is available and find alternative if needed
    if rank == 0 and not check_port_available(master_port):
        stats_print(f"Port {master_port} is busy, finding alternative...")
        master_port = find_available_port()
        stats_print(f"Using port {master_port}")
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    if DEBUG:
        print(f"[Master] Setting up distributed training:")
        print(f"[Master]   Rank: {rank}")
        print(f"[Master]   World size: {world_size}")
        print(f"[Master]   Master addr: {master_addr}")
        print(f"[Master]   Master port: {master_port}")
    
    try:
        # Set environment variables for env:// init method (better for HPC)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Initialize the process group with env:// method (HPC-friendly)
        dist.init_process_group(
            backend="gloo",  # More reliable for CPU-based communication
            init_method="tcp://10.150.0.18:12355",
            rank=int(os.getenv("RANK")),
            world_size=int(os.getenv("WORLD_SIZE")),
            timeout=timedelta(minutes=1)  # Longer timeout
        )
        stats_print("✓ Successfully initialized process group")
        
        # Wait for all processes to join
        debug_print(f"Waiting for all {world_size} processes to join...")
        dist.barrier()  # This ensures all processes are ready
        stats_print("✓ All processes have joined successfully!")
        
        # Enhanced connection test with better error handling
        debug_print("Testing worker connections...")
        for worker_rank in range(1, world_size):
            try:
                # Send connection test signal
                test_tensor = torch.tensor([float(worker_rank)])
                debug_print(f"Sending connection test to worker rank {worker_rank}...")
                dist.send(test_tensor, dst=worker_rank)
                debug_print(f"✓ Connection test successful with worker rank {worker_rank}")
            except Exception as e:
                error_msg = f"✗ Connection test failed with worker rank {worker_rank}: {e}"
                if DEBUG:
                    print(f"[Master] {error_msg}")
                else:
                    stats_print(error_msg)
                raise ConnectionError(f"Cannot establish connection with worker {worker_rank}")
        
        stats_print("✓ All worker connections verified!")
        return master_port  # Return the actual port being used
        
    except Exception as e:
        error_msg = f"❌ FAILED TO INITIALIZE DISTRIBUTED TRAINING: {e}"
        stats_print(error_msg)
        if DEBUG:
            print(f"\n[Master] TROUBLESHOOTING CHECKLIST:")
            print(f"[Master] 1. Are all worker nodes running and ready?")
            print(f"[Master] 2. Can workers reach {master_addr}:{master_port}?")
            print(f"[Master] 3. Is firewall blocking the port?")
            print(f"[Master] 4. Are all machines on the same network?")
            print(f"[Master] 5. Try: telnet {master_addr} {master_port} from worker machines")
        raise

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            debug_print("✓ Distributed cleanup completed")
        except:
            pass

def send_shutdown_signals(world_size, input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank):
    """Send shutdown signals to all workers in the full EBD2N network topology"""
    debug_print("Sending shutdown signals to all EBD2N network workers...")
    
    # Send to input layer workers
    for worker_rank in input_worker_ranks:
        try:
            shutdown_microbatch_size = torch.tensor([-999], dtype=torch.long)
            shutdown_microbatch_id = torch.tensor([-999], dtype=torch.long)
            shutdown_signal = torch.tensor([-999.0] * 10)  # Fixed size shutdown signal
            
            dist.send(shutdown_microbatch_size, dst=worker_rank)
            dist.send(shutdown_microbatch_id, dst=worker_rank)
            dist.send(shutdown_signal, dst=worker_rank)
            debug_print(f"✓ Sent shutdown signal to input worker {worker_rank}")
        except Exception as e:
            debug_print(f"Warning: Could not send shutdown to input worker {worker_rank}: {e}")
    
    # Send to activation layer
    try:
        shutdown_signal = torch.tensor([-999.0] * 10)
        dist.send(shutdown_signal, dst=activation_rank)
        debug_print(f"✓ Sent shutdown signal to activation layer {activation_rank}")
    except Exception as e:
        debug_print(f"Warning: Could not send shutdown to activation layer: {e}")
    
    # Send to weighted workers
    for worker_rank in weighted_worker_ranks:
        try:
            shutdown_signal = torch.tensor([-999.0] * 10)
            dist.send(shutdown_signal, dst=worker_rank)
            debug_print(f"✓ Sent shutdown signal to weighted worker {worker_rank}")
        except Exception as e:
            debug_print(f"Warning: Could not send shutdown to weighted worker {worker_rank}: {e}")
    
    # Send to output layer
    try:
        shutdown_signal = torch.tensor([-999.0] * 10)
        dist.send(shutdown_signal, dst=output_rank)
        debug_print(f"✓ Sent shutdown signal to output layer {output_rank}")
    except Exception as e:
        debug_print(f"Warning: Could not send shutdown to output layer: {e}")
    
    debug_print("✓ Shutdown signals sent to all workers")

def run_master(num_input_workers, batch_size, micro_batch_size, activation_size, world_size, output_dimension, master_addr, master_port):
    """Run the master/epoch node for the full EBD2N network"""
    
    try:
        stats_print("=" * 60)
        stats_print("STARTING EBD2N MASTER NODE")
        stats_print("=" * 60)
        
        rank = 0  # Master is always rank 0
        
        # Calculate network topology
        input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank = calculate_network_topology(
            world_size, num_input_workers
        )
        
        if DEBUG:
            print(f"[Master] Initializing EBD2N framework...")
            print(f"[Master] Calculated network topology:")
            print(f"[Master]   Input workers: {input_worker_ranks}")
            print(f"[Master]   Activation layer: {activation_rank}")
            print(f"[Master]   Weighted workers: {weighted_worker_ranks}")
            print(f"[Master]   Output layer: {output_rank}")
        
        # Setup distributed training
        actual_port = setup_distributed(rank, world_size, master_addr, master_port)
        
        # ... rest of the master logic would go here ...
        # (keeping the file shorter for this example, but you'd include all the engine logic)
        
        stats_print("✓ Master node initialized successfully!")
        stats_print("Ready to coordinate EBD2N network training...")
        
    except KeyboardInterrupt:
        stats_print("🛑 Received interrupt signal. Shutting down EBD2N network gracefully...")
        if world_size:
            input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank = calculate_network_topology(
                world_size, num_input_workers
            )
            send_shutdown_signals(world_size, input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank)
        cleanup_distributed()
    except Exception as e:
        stats_print(f"❌ Error running EBD2N master: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        if world_size:
            input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank = calculate_network_topology(
                world_size, num_input_workers
            )
            send_shutdown_signals(world_size, input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank)
        cleanup_distributed()
    
    stats_print("👋 EBD2N master node shutdown complete")

def main():
    """Main entry point with dynamic configuration from environment"""
    global DEBUG
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    # Get defaults from environment variables
    default_world_size = get_env_or_default('WORLD_SIZE', 7, int)
    default_num_input_workers = get_env_or_default('NUM_INPUT_WORKERS', 2, int)
    default_batch_size = get_env_or_default('BATCH_SIZE', 50, int)
    default_micro_batch_size = get_env_or_default('MICRO_BATCH_SIZE', 5, int)
    default_activation_size = get_env_or_default('ACTIVATION_SIZE', 100, int)
    default_output_dimension = get_env_or_default('OUTPUT_DIMENSION', 10, int)
    default_master_addr = get_env_or_default('MASTER_ADDR', '10.150.0.17', str)
    default_master_port = get_env_or_default('MASTER_PORT', '12355', str)
    default_debug = get_env_or_default('DEBUG', True, bool)
    
    parser = argparse.ArgumentParser(
        description='EBD2N Master Node - Reads from environment variables or command-line',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--world-size', type=int, default=default_world_size, 
                       help=f'Total world size (env: WORLD_SIZE)')
    parser.add_argument('--num-input-workers', type=int, default=default_num_input_workers,
                       help=f'Number of input workers (env: NUM_INPUT_WORKERS)')
    parser.add_argument('--batch-size', type=int, default=default_batch_size,
                       help=f'Batch size (env: BATCH_SIZE)')
    parser.add_argument('--micro-batch-size', type=int, default=default_micro_batch_size,
                       help=f'Micro-batch size (env: MICRO_BATCH_SIZE)')
    parser.add_argument('--activation-size', type=int, default=default_activation_size,
                       help=f'Activation layer size (env: ACTIVATION_SIZE)')
    parser.add_argument('--output-dimension', type=int, default=default_output_dimension,
                       help=f'Output dimension (env: OUTPUT_DIMENSION)')
    parser.add_argument('--master-addr', default=default_master_addr,
                       help=f'Master IP address (env: MASTER_ADDR)')
    parser.add_argument('--master-port', default=default_master_port,
                       help=f'Master port (env: MASTER_PORT)')
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
        print("\n" + "=" * 60)
        print("CONFIGURATION SOURCE PRIORITY:")
        print("  1. Command-line arguments (highest)")
        print("  2. Environment variables")
        print("  3. Hardcoded defaults (lowest)")
        print("=" * 60)
        print(f"\nActive Configuration:")
        print(f"  MASTER_ADDR: {args.master_addr} (from env: {os.getenv('MASTER_ADDR', 'not set')})")
        print(f"  MASTER_PORT: {args.master_port} (from env: {os.getenv('MASTER_PORT', 'not set')})")
        print(f"  WORLD_SIZE: {args.world_size} (from env: {os.getenv('WORLD_SIZE', 'not set')})")
        print(f"  NUM_INPUT_WORKERS: {args.num_input_workers} (from env: {os.getenv('NUM_INPUT_WORKERS', 'not set')})")
        print("=" * 60 + "\n")
    
    # Validate arguments
    if args.micro_batch_size <= 0:
        print("Error: Micro-batch size must be > 0")
        return
    
    if args.micro_batch_size > args.batch_size:
        print("Warning: Micro-batch size larger than batch size, adjusting...")
        args.micro_batch_size = args.batch_size
    
    # Validate input partitioning constraint
    if 784 % args.num_input_workers != 0:
        print(f"Error: Input vector size (784) must be divisible by number of input workers {args.num_input_workers}")
        valid_input_workers = [i for i in range(1, 21) if 784 % i == 0]
        print(f"Valid input worker counts: {valid_input_workers}")
        return
    
    # Validate world size
    min_world_size = calculate_minimum_world_size(args.num_input_workers)
    if args.world_size < min_world_size:
        available_for_weighted = max(0, args.world_size - args.num_input_workers - 3)
        
        print(f"Error: World size must be at least {min_world_size} (got {args.world_size})")
        print(f"Network topology breakdown:")
        print(f"  - 1 master (rank 0)")
        print(f"  - {args.num_input_workers} input workers")
        print(f"  - 1 activation layer")
        print(f"  - {available_for_weighted} weighted workers (need at least 1)")
        print(f"  - 1 output layer")
        return
    
    # Calculate and validate final topology
    input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank = calculate_network_topology(
        args.world_size, args.num_input_workers
    )
    num_weighted_workers = len(weighted_worker_ranks)
    
    if args.output_dimension % num_weighted_workers != 0:
        print(f"Error: Output dimension {args.output_dimension} must be divisible by weighted workers {num_weighted_workers}")
        valid_output_dims = [i for i in range(1, 21) if i % num_weighted_workers == 0]
        print(f"Valid output dimensions: {valid_output_dims}")
        return
    
    if DEBUG:
        print(f"🚀 Starting EBD2N master with topology:")
        print(f"   World size: {args.world_size}")
        print(f"   Input workers: {args.num_input_workers} (ranks {input_worker_ranks})")
        print(f"   Activation layer: rank {activation_rank}")  
        print(f"   Weighted workers: {num_weighted_workers} (ranks {weighted_worker_ranks})")
        print(f"   Output layer: rank {output_rank}")
        print(f"   Master: {args.master_addr}:{args.master_port}")
    
    run_master(
        num_input_workers=args.num_input_workers,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        activation_size=args.activation_size,
        world_size=args.world_size,
        output_dimension=args.output_dimension,
        master_addr=args.master_addr,
        master_port=args.master_port
    )

if __name__ == "__main__":
    main()