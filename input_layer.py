# ENHANCED INPUT LAYER NODE WITH IMAGE INDEXING

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

# Global debug flag
DEBUG = True

def debug_print(message, rank=None):
    """Print debug messages only if DEBUG is True"""
    if DEBUG:
        if rank is not None:
            print(f"[InputWorker {rank}] {message}")
        else:
            print(f"[InputWorker] {message}")

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
        print(f"[InputWorker {rank}] Setting up distributed training:")
        print(f"[InputWorker {rank}]   Rank: {rank}")
        print(f"[InputWorker {rank}]   World size: {world_size}")
        print(f"[InputWorker {rank}]   Master addr: {master_addr}")
        print(f"[InputWorker {rank}]   Master port: {master_port}")
    
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
            print(f"\n[InputWorker {rank}] ❌ FAILED TO INITIALIZE DISTRIBUTED TRAINING:")
            print(f"[InputWorker {rank}] Error: {e}")
            print_troubleshooting_tips(rank, master_addr, master_port)
        raise

def print_troubleshooting_tips(rank, master_addr, master_port):
    """Print comprehensive troubleshooting information"""
    if DEBUG:
        print(f"\n[InputWorker {rank}] TROUBLESHOOTING CHECKLIST:")
        print(f"[InputWorker {rank}] 1. Is the master node running?")
        print(f"[InputWorker {rank}] 2. Can you ping {master_addr}?")
        print(f"[InputWorker {rank}] 3. Is port {master_port} open in firewall?")
        print(f"[InputWorker {rank}] 4. Try: telnet {master_addr} {master_port}")
        print(f"[InputWorker {rank}] 5. Are you on the same network as master?")
        print(f"[InputWorker {rank}] 6. Check if master changed to a different port")
        print(f"[InputWorker {rank}] 7. Ensure master started before workers")

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            debug_print("✓ Distributed cleanup completed")
        except:
            pass

def worker_process(rank, world_size, activation_rank, master_addr, master_port):
    """ENHANCED input worker process - receives image index + vector split"""
    if DEBUG:
        print(f"=" * 60)
        print(f"STARTING ENHANCED INPUT WORKER RANK {rank}")
        print(f"=" * 60)
    
    try:
        debug_print("Initializing...", rank)
        
        # Setup distributed environment
        setup_distributed(rank, world_size, master_addr, master_port)
        
        debug_print("✓ Ready for processing!", rank)
        debug_print(f"Will send processed data to activation node (rank {activation_rank})", rank)
        debug_print("ENHANCEMENT: Expecting image index + vector split for proper synchronization", rank)
        debug_print("Entering main processing loop...", rank)
        
        if DEBUG:
            print(f"-" * 60)
        
        # Processing statistics
        total_processed = 0
        start_time = time.time()
        last_report_time = start_time
        
        # ENHANCED: Track processed images for debugging
        processed_images = set()
        last_image_index = 0
        out_of_order_count = 0
        
        # Generate random weight matrix for this input worker (based on rank for reproducibility)
        torch.manual_seed(rank * 42)  # Different seed for each worker
        
        # Calculate expected split size (784 / number of input workers)
        num_input_workers = world_size - 2  # Exclude master and activation node
        expected_split_size = 784 // num_input_workers
        
        weight_matrix = torch.randn(100, expected_split_size)  # 100 x expected_split_size
        debug_print(f"Generated weight matrix: {weight_matrix.shape} for expected split size {expected_split_size}", rank)
        
        # Worker processing loop
        while True:
            try:
                # ENHANCED: Receive image index first
                image_index_tensor = torch.zeros(1, dtype=torch.long)
                dist.recv(image_index_tensor, src=0)
                image_index = image_index_tensor.item()
                
                # Check for shutdown signal (special image index -999)
                if image_index == -999:
                    debug_print("🛑 Received shutdown signal", rank)
                    break
                
                # ENHANCED: Receive the vector split
                vector_split = torch.zeros(expected_split_size)  # Fixed size reception
                dist.recv(vector_split, src=0)
                
                # Check for additional shutdown patterns
                if torch.all(vector_split == -999.0):  # Backup shutdown signal pattern
                    debug_print("🛑 Received backup shutdown signal", rank)
                    break
                elif torch.all(vector_split == 0.0):  # Potential heartbeat
                    debug_print("💓 Received potential heartbeat", rank)
                    continue
                
                # ENHANCED: Track image processing order
                if image_index in processed_images:
                    debug_print(f"⚠️  Duplicate image index {image_index} received!", rank)
                else:
                    processed_images.add(image_index)
                
                # Check for out-of-order processing
                if image_index < last_image_index:
                    out_of_order_count += 1
                    if DEBUG and out_of_order_count <= 5:  # Only log first few out-of-order events
                        debug_print(f"📋 Out-of-order processing: received {image_index} after {last_image_index}", rank)
                
                last_image_index = max(last_image_index, image_index)
                
                # Process the vector split with weight matrix
                # Apply weight matrix: (100, split_size) @ (split_size,) = (100,)
                weighted_result = torch.mv(weight_matrix, vector_split)
                
                # ENHANCED: Send image index + weighted result to activation node
                image_index_tensor_out = torch.tensor([image_index], dtype=torch.long)
                dist.send(image_index_tensor_out, dst=activation_rank)
                dist.send(weighted_result, dst=activation_rank)
                
                total_processed += 1
                
                # Report progress every 50 operations or every 10 seconds
                current_time = time.time()
                if DEBUG and ((total_processed % 50 == 0) or (current_time - last_report_time > 10)):
                    elapsed = current_time - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    unique_images = len(processed_images)
                    print(f"[InputWorker {rank}] Processed {total_processed} splits | Unique images: {unique_images} | Rate: {rate:.2f}/sec | Out-of-order: {out_of_order_count} | ENHANCED")
                    last_report_time = current_time
                
                # Detailed logging for first few images
                if DEBUG and total_processed <= 5:
                    debug_print(f"Processed image {image_index}: split_size={expected_split_size}, weighted_sum={torch.sum(weighted_result).item():.4f}", rank)
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["connection", "recv", "send", "peer", "socket"]):
                    debug_print("⚠️  Connection lost with master, shutting down...", rank)
                    break
                else:
                    if DEBUG:
                        print(f"[InputWorker {rank}] ❌ Runtime error: {e}")
                    raise
            except Exception as e:
                if DEBUG:
                    print(f"[InputWorker {rank}] ❌ Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()
                break
        
        # Final statistics
        final_time = time.time()
        total_elapsed = final_time - start_time
        avg_rate = total_processed / total_elapsed if total_elapsed > 0 else 0
        unique_images = len(processed_images)
        
        if DEBUG:
            print(f"\n[InputWorker {rank}] 📊 ENHANCED FINAL STATISTICS:")
            print(f"[InputWorker {rank}]   Total splits processed: {total_processed}")
            print(f"[InputWorker {rank}]   Unique images processed: {unique_images}")
            print(f"[InputWorker {rank}]   Out-of-order events: {out_of_order_count}")
            print(f"[InputWorker {rank}]   Total time: {total_elapsed:.2f}s")
            print(f"[InputWorker {rank}]   Average rate: {avg_rate:.2f} splits/second")
            print(f"[InputWorker {rank}]   ENHANCEMENT: Image indexing ensures proper synchronization")
            
            # Show some processed image indices for verification
            if processed_images:
                sample_indices = sorted(list(processed_images))[:10]
                print(f"[InputWorker {rank}]   Sample processed indices: {sample_indices}")
                if len(processed_images) > 10:
                    print(f"[InputWorker {rank}]   ... and {len(processed_images) - 10} more")
        
    except KeyboardInterrupt:
        debug_print("🛑 Interrupted by user", rank)
    except Exception as e:
        if DEBUG:
            print(f"\n[InputWorker {rank}] ❌ Failed to start or run: {e}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_distributed()
        debug_print("👋 Enhanced input worker process terminated", rank)

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
        print("INTERACTIVE ENHANCED INPUT WORKER SETUP")
        print("=" * 60)
    
    # Get local IP for reference
    local_ip = get_local_ip()
    if DEBUG:
        print(f"Local machine IP: {local_ip}")
        print("ENHANCEMENT: This worker uses image indexing for proper synchronization")
    
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
    
    if DEBUG:
        print(f"\nConfiguration:")
        print(f"  Input worker rank: {rank}")
        print(f"  Master: {master_addr}:{master_port}")
        print(f"  World size: {world_size}")
        print(f"  Activation node rank: {activation_rank}")
        print(f"  Local IP: {local_ip}")
        print(f"  ENHANCEMENT: Image indexing enabled")
    
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
    
    parser = argparse.ArgumentParser(description='Enhanced Distributed Input Worker Node with Image Indexing')
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
    
    if DEBUG:
        print(f"\n🚀 Starting enhanced input worker with configuration:")
        print(f"   Rank: {rank}")
        print(f"   World size: {world_size}")
        print(f"   Master: {master_addr}:{master_port}")
        print(f"   Activation node rank: {activation_rank}")
        print(f"   Debug mode: {DEBUG}")
        print(f"   ENHANCEMENT: Image indexing for proper synchronization")
    
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
        # Run the enhanced input worker
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