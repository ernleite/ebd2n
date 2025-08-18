# ENHANCED WEIGHTED LAYER NODE WITH MICRO-BATCHING AND CONFIGURABLE WEIGHT MATRIX - FIXED VERSION

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
from collections import defaultdict

# Global debug flag
DEBUG = True

def debug_print(message, rank=None):
    """Print debug messages only if DEBUG is True"""
    if DEBUG:
        if rank is not None:
            print(f"[WeightedWorker {rank}] {message}")
        else:
            print(f"[WeightedWorker] {message}")

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
        print(f"[WeightedWorker {rank}] Setting up distributed training:")
        print(f"[WeightedWorker {rank}]   Rank: {rank}")
        print(f"[WeightedWorker {rank}]   World size: {world_size}")
        print(f"[WeightedWorker {rank}]   Master addr: {master_addr}")
        print(f"[WeightedWorker {rank}]   Master port: {master_port}")
    
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
            print(f"\n[WeightedWorker {rank}] ❌ FAILED TO INITIALIZE DISTRIBUTED TRAINING:")
            print(f"[WeightedWorker {rank}] Error: {e}")
            print_troubleshooting_tips(rank, master_addr, master_port)
        raise

def print_troubleshooting_tips(rank, master_addr, master_port):
    """Print comprehensive troubleshooting information"""
    if DEBUG:
        print(f"\n[WeightedWorker {rank}] TROUBLESHOOTING CHECKLIST:")
        print(f"[WeightedWorker {rank}] 1. Is the master node running?")
        print(f"[WeightedWorker {rank}] 2. Can you ping {master_addr}?")
        print(f"[WeightedWorker {rank}] 3. Is port {master_port} open in firewall?")
        print(f"[WeightedWorker {rank}] 4. Try: telnet {master_addr} {master_port}")
        print(f"[WeightedWorker {rank}] 5. Are you on the same network as master?")
        print(f"[WeightedWorker {rank}] 6. Check if master changed to a different port")
        print(f"[WeightedWorker {rank}] 7. Ensure master started before workers")

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            debug_print("✓ Distributed cleanup completed")
        except:
            pass

def weighted_worker_process(rank, world_size, activation_rank, output_rank, weight_matrix_size, activation_size, master_addr, master_port):
    """MICRO-BATCHING ENHANCED weighted worker process with configurable weight matrix size - FIXED VERSION"""
    if DEBUG:
        print(f"=" * 60)
        print(f"STARTING MICRO-BATCHING ENHANCED WEIGHTED WORKER RANK {rank} - FIXED VERSION")
        print(f"=" * 60)
    
    try:
        debug_print("Initializing...", rank)
        
        # Setup distributed environment
        setup_distributed(rank, world_size, master_addr, master_port)
        
        debug_print("✓ Ready for processing!", rank)
        debug_print(f"Will receive activation data from activation node (rank {activation_rank})", rank)
        debug_print(f"Will send processed data to output node (rank {output_rank})", rank)
        debug_print("MICRO-BATCHING: Expecting micro-batch size + micro-batch ID + activation splits", rank)
        debug_print(f"Weight matrix size: {weight_matrix_size} x {activation_size}", rank)
        debug_print("Entering main processing loop...", rank)
        
        if DEBUG:
            print(f"-" * 60)
        
        # Processing statistics
        total_processed_microbatches = 0
        total_processed_images = 0
        start_time = time.time()
        last_report_time = start_time
        
        # MICRO-BATCHING: Track processed micro-batches for debugging
        processed_microbatches = set()
        last_microbatch_id = 0
        out_of_order_count = 0
        
        # Calculate number of weighted workers
        num_weighted_workers = output_rank - activation_rank - 1  # Workers between activation and output
        
        # Calculate expected split size (activation_size / number of weighted workers)
        expected_split_size = activation_size // num_weighted_workers
        remaining_elements = activation_size % num_weighted_workers
        
        # This worker gets extra elements if it's one of the first workers
        worker_index = rank - activation_rank - 1  # 0-based index among weighted workers
        if worker_index < remaining_elements:
            my_split_size = expected_split_size + 1
        else:
            my_split_size = expected_split_size
        
        debug_print(f"Calculated split sizes: expected={expected_split_size}, my_split_size={my_split_size}", rank)
        debug_print(f"Worker index among weighted workers: {worker_index}/{num_weighted_workers}", rank)
        
        # FIXED: Generate unique random weight matrix for each weighted worker
        # Use a combination of rank and time for better randomness
        base_seed = rank * 2000 + int(time.time() * 1000) % 1000  # Different base than input workers
        torch.manual_seed(base_seed)
        
        # FIXED: Generate truly random weight matrix with proper variance scaling
        # Use Xavier/Glorot initialization for better weight distribution
        weight_matrix = torch.randn(weight_matrix_size, my_split_size) * torch.sqrt(torch.tensor(2.0 / (weight_matrix_size + my_split_size)))
        
        # FIXED: Add rank-specific bias to weights to ensure different outputs
        rank_bias = (rank - activation_rank - 1) * 0.15  # Different bias pattern
        weight_matrix += rank_bias
        
        # FIXED: Apply different scaling factor for each weighted worker
        scale_factor = 1.0 + (rank - activation_rank - 1) * 0.25  # Different scaling pattern
        weight_matrix *= scale_factor
        
        debug_print(f"Generated weight matrix: {weight_matrix.shape} for split size {my_split_size}", rank)
        debug_print(f"Weight matrix dimensions: {weight_matrix_size} x {my_split_size}", rank)
        debug_print(f"FIXED: Using seed {base_seed}, rank bias {rank_bias:.3f}, scale factor {scale_factor:.3f}", rank)
        debug_print(f"FIXED: Weight matrix stats - mean: {torch.mean(weight_matrix).item():.6f}, std: {torch.std(weight_matrix).item():.6f}", rank)
        debug_print(f"FIXED: Weight matrix range - min: {torch.min(weight_matrix).item():.6f}, max: {torch.max(weight_matrix).item():.6f}", rank)
        debug_print("MICRO-BATCHING: Ready to process matrix operations on micro-batches", rank)
        
        # Weighted worker processing loop
        while True:
            try:
                # MICRO-BATCHING ENHANCED: Receive micro-batch size first
                microbatch_size_tensor = torch.zeros(1, dtype=torch.long)
                dist.recv(microbatch_size_tensor, src=activation_rank)
                microbatch_size = microbatch_size_tensor.item()
                
                # Check for shutdown signal (special micro-batch size -999)
                if microbatch_size == -999:
                    debug_print("🛑 Received shutdown signal", rank)
                    break
                
                # MICRO-BATCHING: Receive micro-batch ID
                microbatch_id_tensor = torch.zeros(1, dtype=torch.long)
                dist.recv(microbatch_id_tensor, src=activation_rank)
                microbatch_id = microbatch_id_tensor.item()
                
                # MICRO-BATCHING: Receive the flattened activation split for this worker
                # Expected shape after reshape: (microbatch_size, my_split_size)
                total_elements = microbatch_size * my_split_size
                activation_split_flat = torch.zeros(total_elements)
                dist.recv(activation_split_flat, src=activation_rank)
                
                # Reshape to matrix form: (microbatch_size, my_split_size)
                activation_split = activation_split_flat.view(microbatch_size, my_split_size)
                
                # Check for additional shutdown patterns
                if torch.all(activation_split == -999.0):  # Backup shutdown signal pattern
                    debug_print("🛑 Received backup shutdown signal", rank)
                    break
                elif torch.all(activation_split == 0.0):  # Potential heartbeat
                    debug_print("💓 Received potential heartbeat", rank)
                    continue
                
                # MICRO-BATCHING: Track micro-batch processing order
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
                
                # MICRO-BATCHING: Process the micro-batch matrix with weight matrix
                # Matrix multiplication: (microbatch_size, my_split_size) @ (my_split_size, weight_matrix_size) = (microbatch_size, weight_matrix_size)
                # But our weight matrix is (weight_matrix_size, my_split_size), so we need: (microbatch_size, my_split_size) @ (my_split_size, weight_matrix_size)
                # We transpose: weight_matrix.T is (my_split_size, weight_matrix_size)
                microbatch_start_time = time.time()
                weighted_results = torch.mm(activation_split, weight_matrix.T)  # (microbatch_size, weight_matrix_size)
                microbatch_processing_time = time.time() - microbatch_start_time
                
                # MICRO-BATCHING ENHANCED: Send micro-batch size + micro-batch ID + weighted results to output node
                microbatch_size_tensor_out = torch.tensor([microbatch_size], dtype=torch.long)
                microbatch_id_tensor_out = torch.tensor([microbatch_id], dtype=torch.long)
                
                dist.send(microbatch_size_tensor_out, dst=output_rank)
                dist.send(microbatch_id_tensor_out, dst=output_rank)
                dist.send(weighted_results.flatten(), dst=output_rank)  # Flatten for transmission
                
                total_processed_microbatches += 1
                total_processed_images += microbatch_size
                
                # Report progress every 50 micro-batches or every 10 seconds
                current_time = time.time()
                if DEBUG and ((total_processed_microbatches % 50 == 0) or (current_time - last_report_time > 10)):
                    elapsed = current_time - start_time
                    microbatch_rate = total_processed_microbatches / elapsed if elapsed > 0 else 0
                    image_rate = total_processed_images / elapsed if elapsed > 0 else 0
                    unique_microbatches = len(processed_microbatches)
                    print(f"[WeightedWorker {rank}] Processed {total_processed_microbatches} micro-batches ({total_processed_images} images) | MBatch rate: {microbatch_rate:.2f}/sec | Image rate: {image_rate:.2f}/sec | Unique: {unique_microbatches} | Out-of-order: {out_of_order_count} | Weight matrix: {weight_matrix_size}x{my_split_size} | MICRO-BATCHING - FIXED")
                    last_report_time = current_time
                
                # FIXED: Detailed logging for first few micro-batches with input analysis
                if DEBUG and total_processed_microbatches <= 3:
                    input_sum = torch.sum(activation_split).item()
                    input_mean = torch.mean(activation_split).item()
                    input_std = torch.std(activation_split).item()
                    output_sum = torch.sum(weighted_results).item()
                    output_mean = torch.mean(weighted_results).item()
                    output_std = torch.std(weighted_results).item()
                    
                    debug_print(f"FIXED: Processed micro-batch {microbatch_id}:", rank)
                    debug_print(f"  Size: {microbatch_size} images", rank)
                    debug_print(f"  Input shape: {activation_split.shape}", rank)
                    debug_print(f"  Weight matrix shape: {weight_matrix.shape}", rank)
                    debug_print(f"  Output shape: {weighted_results.shape}", rank)
                    debug_print(f"  Processing time: {microbatch_processing_time:.4f}s", rank)
                    debug_print(f"  FIXED: Input stats - sum: {input_sum:.4f}, mean: {input_mean:.6f}, std: {input_std:.6f}", rank)
                    debug_print(f"  FIXED: Output stats - sum: {output_sum:.4f}, mean: {output_mean:.6f}, std: {output_std:.6f}", rank)
                    debug_print(f"  FIXED: Amplification factor: {output_sum/input_sum:.3f}" if input_sum != 0 else "  FIXED: Input sum is zero", rank)
                    debug_print(f"  MICRO-BATCHING: Processed {microbatch_size} images in single operation", rank)
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["connection", "recv", "send", "peer", "socket"]):
                    debug_print("⚠️  Connection lost with activation node, shutting down...", rank)
                    break
                else:
                    if DEBUG:
                        print(f"[WeightedWorker {rank}] ❌ Runtime error: {e}")
                    raise
            except Exception as e:
                if DEBUG:
                    print(f"[WeightedWorker {rank}] ❌ Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()
                break
        
        # Send shutdown signal to output node
        try:
            shutdown_tensor = torch.tensor([-999], dtype=torch.long)
            dist.send(shutdown_tensor, dst=output_rank)
            debug_print("✓ Sent shutdown signal to output node", rank)
        except:
            debug_print("⚠️  Failed to send shutdown signal to output node", rank)
        
        # Final statistics
        final_time = time.time()
        total_elapsed = final_time - start_time
        avg_microbatch_rate = total_processed_microbatches / total_elapsed if total_elapsed > 0 else 0
        avg_image_rate = total_processed_images / total_elapsed if total_elapsed > 0 else 0
        unique_microbatches = len(processed_microbatches)
        
        if DEBUG:
            print(f"\n[WeightedWorker {rank}] 📊 MICRO-BATCHING ENHANCED FINAL STATISTICS (FIXED):")
            print(f"[WeightedWorker {rank}]   Total micro-batches processed: {total_processed_microbatches}")
            print(f"[WeightedWorker {rank}]   Total images processed: {total_processed_images}")
            print(f"[WeightedWorker {rank}]   Unique micro-batches processed: {unique_microbatches}")
            print(f"[WeightedWorker {rank}]   Out-of-order events: {out_of_order_count}")
            print(f"[WeightedWorker {rank}]   Total time: {total_elapsed:.2f}s")
            print(f"[WeightedWorker {rank}]   Average micro-batch rate: {avg_microbatch_rate:.2f} micro-batches/second")
            print(f"[WeightedWorker {rank}]   Average image rate: {avg_image_rate:.2f} images/second")
            print(f"[WeightedWorker {rank}]   Average images per micro-batch: {total_processed_images/total_processed_microbatches:.1f}" if total_processed_microbatches > 0 else "")
            print(f"[WeightedWorker {rank}]   Weight matrix dimensions: {weight_matrix_size} x {my_split_size}")
            print(f"[WeightedWorker {rank}]   FIXED: Weight matrix uniqueness - seed: {base_seed}, bias: {rank_bias:.3f}, scale: {scale_factor:.3f}")
            print(f"[WeightedWorker {rank}]   FIXED: Final weight stats - mean: {torch.mean(weight_matrix).item():.6f}, std: {torch.std(weight_matrix).item():.6f}")
            print(f"[WeightedWorker {rank}]   MICRO-BATCHING: Communication events reduced significantly")
            print(f"[WeightedWorker {rank}]   MICRO-BATCHING: Matrix operations provide computational efficiency")
            
            # Show some processed micro-batch IDs for verification
            if processed_microbatches:
                sample_microbatch_ids = sorted(list(processed_microbatches))[:10]
                print(f"[WeightedWorker {rank}]   Sample processed micro-batch IDs: {sample_microbatch_ids}")
                if len(processed_microbatches) > 10:
                    print(f"[WeightedWorker {rank}]   ... and {len(processed_microbatches) - 10} more")
        
    except KeyboardInterrupt:
        debug_print("🛑 Interrupted by user", rank)
    except Exception as e:
        if DEBUG:
            print(f"\n[WeightedWorker {rank}] ❌ Failed to start or run: {e}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_distributed()
        debug_print("👋 Enhanced micro-batching weighted worker process terminated - FIXED VERSION", rank)

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
    """Interactive setup for weighted worker configuration"""
    if DEBUG:
        print("=" * 60)
        print("INTERACTIVE MICRO-BATCHING ENHANCED WEIGHTED WORKER SETUP - FIXED VERSION")
        print("=" * 60)
    
    # Get local IP for reference
    local_ip = get_local_ip()
    if DEBUG:
        print(f"Local machine IP: {local_ip}")
        print("MICRO-BATCHING: This worker processes activation splits and applies weights")
        print("FIXED: Each weighted worker has unique weight matrices for different outputs")
    
    # Get configuration from user
    try:
        rank = int(input("Enter weighted worker rank (between activation and output ranks): "))
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
    
    world_size = input(f"Enter world size (total nodes including master) [6]: ").strip()
    if not world_size:
        world_size = 6
    else:
        try:
            world_size = int(world_size)
        except ValueError:
            print("Invalid world size. Using default 6.")
            world_size = 6
    
    # Get activation rank
    activation_rank = input(f"Enter activation node rank [3]: ").strip()
    if not activation_rank:
        activation_rank = 3
    else:
        try:
            activation_rank = int(activation_rank)
        except ValueError:
            print("Invalid activation rank. Using default 3.")
            activation_rank = 3
    
    # Get output rank (usually last rank)
    output_rank = input(f"Enter output node rank [{world_size - 1}]: ").strip()
    if not output_rank:
        output_rank = world_size - 1
    else:
        try:
            output_rank = int(output_rank)
        except ValueError:
            print(f"Invalid output rank. Using default {world_size - 1}.")
            output_rank = world_size - 1
    
    # Get weight matrix parameters
    weight_matrix_size = input(f"Enter weight matrix output size (number of classes) [10]: ").strip()
    if not weight_matrix_size:
        weight_matrix_size = 10
    else:
        try:
            weight_matrix_size = int(weight_matrix_size)
            if weight_matrix_size < 1:
                print("Weight matrix size must be >= 1. Using default 10.")
                weight_matrix_size = 10
        except ValueError:
            print("Invalid weight matrix size. Using default 10.")
            weight_matrix_size = 10
    
    activation_size = input(f"Enter activation layer size (from activation node) [100]: ").strip()
    if not activation_size:
        activation_size = 100
    else:
        try:
            activation_size = int(activation_size)
            if activation_size < 1:
                print("Activation size must be >= 1. Using default 100.")
                activation_size = 100
        except ValueError:
            print("Invalid activation size. Using default 100.")
            activation_size = 100
    
    if DEBUG:
        print(f"\nConfiguration:")
        print(f"  Weighted worker rank: {rank}")
        print(f"  Master: {master_addr}:{master_port}")
        print(f"  World size: {world_size}")
        print(f"  Activation node rank: {activation_rank}")
        print(f"  Output node rank: {output_rank}")
        print(f"  Weight matrix size: {weight_matrix_size}")
        print(f"  Activation size: {activation_size}")
        print(f"  Local IP: {local_ip}")
        print(f"  MICRO-BATCHING: Matrix-based operations enabled")
        print(f"  FIXED: Unique weight matrix generation per weighted worker")
    
    confirm = input(f"\nProceed with this configuration? [y/N]: ").strip().lower()
    if confirm in ['y', 'yes']:
        return rank, world_size, activation_rank, output_rank, weight_matrix_size, activation_size, master_addr, master_port
    else:
        print("Setup cancelled.")
        return None

def main():
    """Main weighted worker entry point with multiple launch modes"""
    global DEBUG
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Micro-Batching Enhanced Distributed Weighted Worker Node - FIXED VERSION')
    parser.add_argument('--rank', type=int, help='Weighted worker rank (between activation and output)')
    parser.add_argument('--world-size', type=int, default=6, help='Total world size including all nodes')
    parser.add_argument('--activation-rank', type=int, default=3, help='Activation node rank')
    parser.add_argument('--output-rank', type=int, help='Output node rank (usually last)')
    parser.add_argument('--weight-matrix-size', type=int, default=10, help='Size of weight matrix output (number of classes)')
    parser.add_argument('--activation-size', type=int, default=100, help='Size of activation layer input')
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
        rank, world_size, activation_rank, output_rank, weight_matrix_size, activation_size, master_addr, master_port = setup
    else:
        # Command line mode
        rank = args.rank
        world_size = args.world_size
        activation_rank = args.activation_rank
        output_rank = args.output_rank if args.output_rank is not None else world_size - 1
        weight_matrix_size = args.weight_matrix_size
        activation_size = args.activation_size
        master_addr = args.master_addr
        master_port = args.master_port
    
    # Validate configuration
    if rank <= activation_rank or rank >= output_rank:
        print(f"Error: Weighted worker rank {rank} must be between activation rank {activation_rank} and output rank {output_rank}")
        return
    
    if rank < 1:
        print(f"Error: Weighted worker rank must be >= 1 (rank 0 is reserved for master)")
        return
    
    if weight_matrix_size < 1 or activation_size < 1:
        print(f"Error: Weight matrix size and activation size must be >= 1")
        return
    
    if DEBUG:
        print(f"\n🚀 Starting micro-batching enhanced weighted worker with configuration - FIXED VERSION:")
        print(f"   Rank: {rank}")
        print(f"   World size: {world_size}")
        print(f"   Master: {master_addr}:{master_port}")
        print(f"   Activation node rank: {activation_rank}")
        print(f"   Output node rank: {output_rank}")
        print(f"   Weight matrix size: {weight_matrix_size}")
        print(f"   Activation size: {activation_size}")
        print(f"   Debug mode: {DEBUG}")
        print(f"   MICRO-BATCHING: Matrix operations for computational efficiency")
        print(f"   FIXED: Unique weight matrices per weighted worker ensure different outputs")
    
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
        # Run the micro-batching enhanced weighted worker
        weighted_worker_process(rank, world_size, activation_rank, output_rank, weight_matrix_size, activation_size, master_addr, master_port)
    except KeyboardInterrupt:
        debug_print("🛑 Weighted worker interrupted by user")
    except Exception as e:
        if DEBUG:
            print(f"\n❌ Weighted worker failed: {e}")

if __name__ == "__main__":
    # Handle different launch methods
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        if DEBUG:
            print("No arguments provided. Starting interactive setup...")
        sys.argv.append('--interactive')
    
    main()