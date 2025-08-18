# ENHANCED ACTIVATION LAYER NODE WITH MICRO-BATCHING AND CONFIGURABLE WEIGHT MATRIX SIZE - FIXED VERSION

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
from collections import defaultdict

# Global debug flag
DEBUG = True

def debug_print(message, rank=None):
    """Print debug messages only if DEBUG is True"""
    if DEBUG:
        if rank is not None:
            print(f"[ActivationNode {rank}] {message}")
        else:
            print(f"[ActivationNode] {message}")

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
        print(f"[ActivationNode {rank}] Setting up distributed training:")
        print(f"[ActivationNode {rank}]   Rank: {rank}")
        print(f"[ActivationNode {rank}]   World size: {world_size}")
        print(f"[ActivationNode {rank}]   Master addr: {master_addr}")
        print(f"[ActivationNode {rank}]   Master port: {master_port}")
    
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
            print(f"\n[ActivationNode {rank}] ❌ FAILED TO INITIALIZE DISTRIBUTED TRAINING:")
            print(f"[ActivationNode {rank}] Error: {e}")
            print_troubleshooting_tips(rank, master_addr, master_port)
        raise

def print_troubleshooting_tips(rank, master_addr, master_port):
    """Print comprehensive troubleshooting information"""
    if DEBUG:
        print(f"\n[ActivationNode {rank}] TROUBLESHOOTING CHECKLIST:")
        print(f"[ActivationNode {rank}] 1. Is the master node running?")
        print(f"[ActivationNode {rank}] 2. Can you ping {master_addr}?")
        print(f"[ActivationNode {rank}] 3. Is port {master_port} open in firewall?")
        print(f"[ActivationNode {rank}] 4. Try: telnet {master_addr} {master_port}")
        print(f"[ActivationNode {rank}] 5. Are you on the same network as master?")
        print(f"[ActivationNode {rank}] 6. Check if master changed to a different port")
        print(f"[ActivationNode {rank}] 7. Ensure master started before workers")

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            debug_print("✓ Distributed cleanup completed")
        except:
            pass

def get_activation_function(activation_name):
    """Get activation function by name"""
    activation_functions = {
        'relu': F.relu,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'leaky_relu': lambda x: F.leaky_relu(x, 0.01),
        'elu': F.elu,
        'gelu': F.gelu,
        'swish': lambda x: x * torch.sigmoid(x),
        'linear': lambda x: x  # No activation
    }
    
    if activation_name.lower() not in activation_functions:
        debug_print(f"Warning: Unknown activation '{activation_name}', using ReLU")
        return activation_functions['relu']
    
    return activation_functions[activation_name.lower()]

def activation_process(rank, world_size, num_input_workers, activation_size, activation_function_name, master_addr, master_port):
    """MICRO-BATCHING ENHANCED activation node process with configurable activation size - FIXED VERSION"""
    if DEBUG:
        print(f"=" * 60)
        print(f"STARTING MICRO-BATCHING ENHANCED ACTIVATION NODE RANK {rank} - FIXED VERSION")
        print(f"=" * 60)
    
    try:
        debug_print("Initializing...", rank)
        
        # Setup distributed environment
        setup_distributed(rank, world_size, master_addr, master_port)
        
        # FIXED: Initialize bias vector with different approach for more variety
        # Use combination of rank and time for unique bias generation
        bias_seed = rank * 500 + int(time.time() * 100) % 1000
        torch.manual_seed(bias_seed)
        
        # FIXED: Generate bias with proper scaling and ensure it's not constant
        # Use normal distribution with reasonable variance
        bias = torch.randn(activation_size) * 0.1  # Scale bias to reasonable magnitude
        
        # FIXED: Add small incremental bias to ensure different activations
        increment_bias = torch.arange(activation_size, dtype=torch.float32) * 0.01  # Small incremental pattern
        bias += increment_bias
        
        debug_print(f"FIXED: Generated bias vector: {bias.shape} with seed {bias_seed}", rank)
        debug_print(f"FIXED: Bias stats - mean: {torch.mean(bias).item():.6f}, std: {torch.std(bias).item():.6f}", rank)
        debug_print(f"FIXED: Bias range - min: {torch.min(bias).item():.6f}, max: {torch.max(bias).item():.6f}", rank)
        debug_print(f"Activation size (configurable): {activation_size}", rank)
        
        # Get activation function
        activation_fn = get_activation_function(activation_function_name)
        debug_print(f"Using activation function: {activation_function_name}", rank)
        
        debug_print("✓ Ready for processing!", rank)
        debug_print(f"Expecting inputs from {num_input_workers} input workers (ranks 1-{num_input_workers})", rank)
        debug_print("MICRO-BATCHING: Using micro-batch synchronization for efficient processing", rank)
        debug_print("FIXED: Bias application will create varied activation outputs", rank)
        debug_print("Entering main processing loop...", rank)
        
        if DEBUG:
            print(f"-" * 60)
        
        # Processing statistics
        total_processed_microbatches = 0
        total_processed_images = 0
        start_time = time.time()
        last_report_time = start_time
        
        # MICRO-BATCHING: Micro-batch synchronization data structures
        pending_microbatch_splits = defaultdict(dict)  # {microbatch_id: {worker_rank: (microbatch_size, weighted_results_matrix)}}
        completed_microbatches = []  # List of completed micro-batch IDs in order
        max_pending_microbatches = 50  # Maximum number of micro-batches to keep in memory
        out_of_order_count = 0
        timeout_count = 0
        
        # FIXED: Track activation statistics for debugging
        activation_stats = {
            'total_sum': 0.0,
            'total_mean': 0.0,
            'min_activation': float('inf'),
            'max_activation': float('-inf'),
            'zero_activations': 0,
            'positive_activations': 0,
            'negative_activations': 0
        }
        
        # Activation node processing loop
        while True:
            try:
                # MICRO-BATCHING ENHANCED: Collect weighted results with proper micro-batch synchronization
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
                        # Expected shape after reshape: (microbatch_size, activation_size)
                        total_elements = microbatch_size * activation_size
                        weighted_results_flat = torch.zeros(total_elements)
                        dist.recv(weighted_results_flat, src=input_worker_rank)
                        
                        # Reshape to matrix form: (microbatch_size, activation_size)
                        weighted_results = weighted_results_flat.view(microbatch_size, activation_size)
                        
                        # Check for invalid data
                        if torch.isnan(weighted_results).any() or torch.isinf(weighted_results).any():
                            debug_print(f"🛑 Received invalid data from worker {input_worker_rank}, treating as shutdown", rank)
                            shutdown_received = True
                            break
                        
                        # MICRO-BATCHING: Store the weighted results for this micro-batch and worker
                        pending_microbatch_splits[microbatch_id][input_worker_rank] = (microbatch_size, weighted_results)
                        
                        if DEBUG and total_processed_microbatches < 2:
                            result_sum = torch.sum(weighted_results).item()
                            result_mean = torch.mean(weighted_results).item()
                            result_std = torch.std(weighted_results).item()
                            debug_print(f"FIXED: Received weighted results for micro-batch {microbatch_id} from input worker {input_worker_rank}: size={microbatch_size}, sum={result_sum:.4f}, mean={result_mean:.6f}, std={result_std:.6f}", rank)
                        
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
                
                # MICRO-BATCHING: Process all complete micro-batches (micro-batches that have splits from all workers)
                microbatches_to_process = []
                for microbatch_id in sorted(pending_microbatch_splits.keys()):
                    if len(pending_microbatch_splits[microbatch_id]) == num_input_workers:
                        # All splits received for this micro-batch
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
                        # MICRO-BATCHING: Sum all weighted results for this micro-batch
                        # Each element in weighted_results_list is (microbatch_size, activation_size)
                        # Stack and sum across input workers: (num_input_workers, microbatch_size, activation_size) -> (microbatch_size, activation_size)
                        z_matrix = torch.stack(weighted_results_list).sum(dim=0)  # Sum across input workers
                        
                        if DEBUG and total_processed_microbatches < 2:
                            z_sum = torch.sum(z_matrix).item()
                            z_mean = torch.mean(z_matrix).item()
                            z_std = torch.std(z_matrix).item()
                            debug_print(f"FIXED: Combined {len(weighted_results_list)} weighted results for micro-batch {microbatch_id}: size={microbatch_size}, sum={z_sum:.4f}, mean={z_mean:.6f}, std={z_std:.6f}", rank)
                        
                        # MICRO-BATCHING: Add bias to all images in micro-batch
                        # z_matrix: (microbatch_size, activation_size), bias: (activation_size,)
                        # Broadcasting: z_matrix + bias -> (microbatch_size, activation_size)
                        z_matrix_with_bias = z_matrix + bias  # Broadcasting automatically handles this
                        
                        if DEBUG and total_processed_microbatches < 2:
                            bias_sum = torch.sum(bias).item()
                            z_bias_sum = torch.sum(z_matrix_with_bias).item()
                            z_bias_mean = torch.mean(z_matrix_with_bias).item()
                            z_bias_std = torch.std(z_matrix_with_bias).item()
                            debug_print(f"FIXED: After bias addition - bias_sum={bias_sum:.4f}, z_with_bias: sum={z_bias_sum:.4f}, mean={z_bias_mean:.6f}, std={z_bias_std:.6f}", rank)
                        
                        # MICRO-BATCHING: Apply activation function to entire micro-batch
                        activation_output_matrix = activation_fn(z_matrix_with_bias)  # (microbatch_size, activation_size)
                        
                        # FIXED: Update activation statistics
                        with torch.no_grad():
                            current_sum = torch.sum(activation_output_matrix).item()
                            current_mean = torch.mean(activation_output_matrix).item()
                            current_min = torch.min(activation_output_matrix).item()
                            current_max = torch.max(activation_output_matrix).item()
                            
                            activation_stats['total_sum'] += current_sum
                            activation_stats['total_mean'] += current_mean
                            activation_stats['min_activation'] = min(activation_stats['min_activation'], current_min)
                            activation_stats['max_activation'] = max(activation_stats['max_activation'], current_max)
                            
                            # Count activation types
                            zero_count = (activation_output_matrix == 0).sum().item()
                            positive_count = (activation_output_matrix > 0).sum().item()
                            negative_count = (activation_output_matrix < 0).sum().item()
                            
                            activation_stats['zero_activations'] += zero_count
                            activation_stats['positive_activations'] += positive_count
                            activation_stats['negative_activations'] += negative_count
                        
                        # MICRO-BATCHING ENHANCED: Send micro-batch size + micro-batch ID + activation results back to master
                        try:
                            microbatch_size_tensor_out = torch.tensor([microbatch_size], dtype=torch.long)
                            microbatch_id_tensor_out = torch.tensor([microbatch_id], dtype=torch.long)
                            
                            dist.send(microbatch_size_tensor_out, dst=0)
                            dist.send(microbatch_id_tensor_out, dst=0)
                            dist.send(activation_output_matrix.flatten(), dst=0)  # Flatten for transmission
                            
                            if DEBUG and total_processed_microbatches < 2:
                                act_sum = torch.sum(activation_output_matrix).item()
                                act_mean = torch.mean(activation_output_matrix).item()
                                act_std = torch.std(activation_output_matrix).item()
                                debug_print(f"FIXED: Sent activation results for micro-batch {microbatch_id} to master: size={microbatch_size}, sum={act_sum:.4f}, mean={act_mean:.6f}, std={act_std:.6f}", rank)
                                
                                # FIXED: Show per-image activation analysis
                                for img_idx in range(min(2, microbatch_size)):
                                    img_activation_sum = torch.sum(activation_output_matrix[img_idx]).item()
                                    img_activation_mean = torch.mean(activation_output_matrix[img_idx]).item()
                                    debug_print(f"FIXED:   Image {img_idx} in micro-batch {microbatch_id}: activation_sum={img_activation_sum:.4f}, activation_mean={img_activation_mean:.6f}", rank)
                                
                        except Exception as send_error:
                            debug_print(f"✗ Error sending result for micro-batch {microbatch_id} to master: {send_error}", rank)
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
                            debug_print(f"FIXED: Micro-batch {microbatch_id} processing complete:", rank)
                            debug_print(f"  Received from {len(weighted_results_list)} input workers", rank)
                            debug_print(f"  Micro-batch size: {microbatch_size} images", rank)
                            debug_print(f"  Z-matrix (weighted sum) shape: {z_matrix.shape}, sum: {torch.sum(z_matrix).item():.4f}", rank)
                            debug_print(f"  Bias sum: {torch.sum(bias).item():.4f}", rank)
                            debug_print(f"  Z-matrix (after bias) sum: {torch.sum(z_matrix_with_bias).item():.4f}", rank)
                            debug_print(f"  Activation output shape: {activation_output_matrix.shape}, sum: {torch.sum(activation_output_matrix).item():.4f}", rank)
                            debug_print(f"  Activation output mean: {torch.mean(activation_output_matrix).item():.6f}", rank)
                            debug_print(f"  Activation output max: {torch.max(activation_output_matrix).item():.4f}", rank)
                            debug_print(f"  Activation output min: {torch.min(activation_output_matrix).item():.4f}", rank)
                            debug_print(f"  FIXED: Activation function: {activation_function_name}", rank)
                            debug_print(f"  MICRO-BATCHING: Processed {microbatch_size} images in single operation", rank)
                    
                    # Clean up processed micro-batch
                    del pending_microbatch_splits[microbatch_id]
                
                # Clean up old pending micro-batches to prevent memory issues
                if len(pending_microbatch_splits) > max_pending_microbatches:
                    # Remove oldest incomplete micro-batches
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
                    avg_activation_sum = activation_stats['total_sum'] / total_processed_microbatches if total_processed_microbatches > 0 else 0
                    print(f"[ActivationNode {rank}] Processed {total_processed_microbatches} micro-batches ({total_processed_images} images) | MBatch rate: {microbatch_rate:.2f}/sec | Image rate: {image_rate:.2f}/sec | Pending: {pending_count} | Out-of-order: {out_of_order_count} | Timeouts: {timeout_count} | Avg Act Sum: {avg_activation_sum:.4f} | Activation size: {activation_size} | MICRO-BATCHING - FIXED")
                    last_report_time = current_time
                
                # If no micro-batches were processed in this iteration and we have no pending splits, 
                # we might be done or there might be a communication issue
                if not microbatches_to_process and not pending_microbatch_splits:
                    # Wait a bit for more data or check if we should continue
                    if total_processed_microbatches > 0:  # Only wait if we've processed at least one micro-batch
                        time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["connection", "recv", "send", "peer", "socket"]):
                    debug_print("⚠️  Connection lost with master or input workers, shutting down...", rank)
                    break
                else:
                    if DEBUG:
                        print(f"[ActivationNode {rank}] ❌ Runtime error: {e}")
                    raise
            except Exception as e:
                if DEBUG:
                    print(f"[ActivationNode {rank}] ❌ Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()
                break
        
        # Final statistics
        final_time = time.time()
        total_elapsed = final_time - start_time
        avg_microbatch_rate = total_processed_microbatches / total_elapsed if total_elapsed > 0 else 0
        avg_image_rate = total_processed_images / total_elapsed if total_elapsed > 0 else 0
        
        if DEBUG:
            print(f"\n[ActivationNode {rank}] 📊 MICRO-BATCHING ENHANCED FINAL STATISTICS (FIXED):")
            print(f"[ActivationNode {rank}]   Total micro-batches processed: {total_processed_microbatches}")
            print(f"[ActivationNode {rank}]   Total images processed: {total_processed_images}")
            print(f"[ActivationNode {rank}]   Out-of-order completions: {out_of_order_count}")
            print(f"[ActivationNode {rank}]   Timed-out micro-batches: {timeout_count}")
            print(f"[ActivationNode {rank}]   Remaining pending micro-batches: {len(pending_microbatch_splits)}")
            print(f"[ActivationNode {rank}]   Total time: {total_elapsed:.2f}s")
            print(f"[ActivationNode {rank}]   Average micro-batch rate: {avg_microbatch_rate:.2f} micro-batches/second")
            print(f"[ActivationNode {rank}]   Average image rate: {avg_image_rate:.2f} images/second")
            print(f"[ActivationNode {rank}]   Average images per micro-batch: {total_processed_images/total_processed_microbatches:.1f}" if total_processed_microbatches > 0 else "")
            print(f"[ActivationNode {rank}]   Activation function used: {activation_function_name}")
            print(f"[ActivationNode {rank}]   Activation size (configurable): {activation_size}")
            
            # FIXED: Show comprehensive activation statistics
            if total_processed_microbatches > 0:
                avg_total_sum = activation_stats['total_sum'] / total_processed_microbatches
                avg_total_mean = activation_stats['total_mean'] / total_processed_microbatches
                activation_elements = total_processed_images * activation_size
                zero_percentage = (activation_stats['zero_activations'] / activation_elements) * 100 if activation_elements > 0 else 0
                positive_percentage = (activation_stats['positive_activations'] / activation_elements) * 100 if activation_elements > 0 else 0
                negative_percentage = (activation_stats['negative_activations'] / activation_elements) * 100 if activation_elements > 0 else 0
                
                print(f"[ActivationNode {rank}]   FIXED: Activation Statistics Summary:")
                print(f"[ActivationNode {rank}]     Average activation sum per micro-batch: {avg_total_sum:.6f}")
                print(f"[ActivationNode {rank}]     Average activation mean per micro-batch: {avg_total_mean:.6f}")
                print(f"[ActivationNode {rank}]     Global activation range: {activation_stats['min_activation']:.6f} to {activation_stats['max_activation']:.6f}")
                print(f"[ActivationNode {rank}]     Zero activations: {activation_stats['zero_activations']} ({zero_percentage:.2f}%)")
                print(f"[ActivationNode {rank}]     Positive activations: {activation_stats['positive_activations']} ({positive_percentage:.2f}%)")
                print(f"[ActivationNode {rank}]     Negative activations: {activation_stats['negative_activations']} ({negative_percentage:.2f}%)")
                print(f"[ActivationNode {rank}]     FIXED: Bias vector mean: {torch.mean(bias).item():.6f}, std: {torch.std(bias).item():.6f}")
            
            print(f"[ActivationNode {rank}]   MICRO-BATCHING: Matrix operations provide computational efficiency")
            print(f"[ActivationNode {rank}]   MICRO-BATCHING: Communication events reduced significantly")
            
            # Show sample completed micro-batch IDs
            if completed_microbatches:
                sample_completed = completed_microbatches[:10]
                print(f"[ActivationNode {rank}]   Sample completed micro-batch IDs: {sample_completed}")
                if len(completed_microbatches) > 10:
                    print(f"[ActivationNode {rank}]   ... and {len(completed_microbatches) - 10} more")
            
            # Show any remaining pending micro-batches
            if pending_microbatch_splits:
                pending_list = list(pending_microbatch_splits.keys())[:5]
                print(f"[ActivationNode {rank}]   Pending micro-batches at shutdown: {pending_list}")
                if len(pending_microbatch_splits) > 5:
                    print(f"[ActivationNode {rank}]   ... and {len(pending_microbatch_splits) - 5} more")
        
    except KeyboardInterrupt:
        debug_print("🛑 Interrupted by user", rank)
    except Exception as e:
        if DEBUG:
            print(f"\n[ActivationNode {rank}] ❌ Failed to start or run: {e}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_distributed()
        debug_print("👋 Enhanced micro-batching activation node process terminated - FIXED VERSION", rank)

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
    """Interactive setup for activation node configuration"""
    if DEBUG:
        print("=" * 60)
        print("INTERACTIVE MICRO-BATCHING ENHANCED ACTIVATION NODE SETUP - FIXED VERSION")
        print("=" * 60)
    
    # Get local IP for reference
    local_ip = get_local_ip()
    if DEBUG:
        print(f"Local machine IP: {local_ip}")
        print("MICRO-BATCHING: This node processes multiple images per communication event")
        print("FIXED: Bias generation now ensures varied activation outputs")
    
    # Get configuration from user
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
    
    # Calculate configuration
    activation_rank = world_size - 1  # Last rank is activation node
    num_input_workers = world_size - 2  # All ranks except master (0) and activation node
    
    # Get activation parameters (configurable activation size)
    activation_size = input(f"Enter activation layer size (weight matrix output size) [100]: ").strip()
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
    
    print(f"\nAvailable activation functions: relu, sigmoid, tanh, leaky_relu, elu, gelu, swish, linear")
    activation_function = input(f"Enter activation function [relu]: ").strip()
    if not activation_function:
        activation_function = "relu"
    
    if DEBUG:
        print(f"\nConfiguration:")
        print(f"  Activation node rank: {activation_rank}")
        print(f"  Master: {master_addr}:{master_port}")
        print(f"  World size: {world_size}")
        print(f"  Number of input workers: {num_input_workers}")
        print(f"  Activation size (configurable): {activation_size}")
        print(f"  Activation function: {activation_function}")
        print(f"  Local IP: {local_ip}")
        print(f"  MICRO-BATCHING: Matrix operations for computational efficiency")
        print(f"  FIXED: Bias generation ensures activation variety")
        print(f"  NOTE: Input workers must use same activation size as weight matrix output size")
    
    confirm = input(f"\nProceed with this configuration? [y/N]: ").strip().lower()
    if confirm in ['y', 'yes']:
        return activation_rank, world_size, num_input_workers, activation_size, activation_function, master_addr, master_port
    else:
        print("Setup cancelled.")
        return None

def main():
    """Main activation node entry point with multiple launch modes"""
    global DEBUG
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Micro-Batching Enhanced Distributed Activation Node - FIXED VERSION')
    parser.add_argument('--rank', type=int, default=None, help='Activation node rank (usually last rank)')
    parser.add_argument('--world-size', type=int, default=4, help='Total world size including master and activation node')
    parser.add_argument('--activation-size', type=int, default=100, help='Size of activation layer (weight matrix output size, default: 100)')
    parser.add_argument('--activation-function', default='relu', help='Activation function (relu, sigmoid, tanh, etc.)')
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
        rank, world_size, num_input_workers, activation_size, activation_function, master_addr, master_port = setup
    else:
        # Command line mode
        rank = args.rank if args.rank is not None else args.world_size - 1
        world_size = args.world_size
        num_input_workers = world_size - 2  # All ranks except master (0) and activation node
        activation_size = args.activation_size
        activation_function = args.activation_function
        master_addr = args.master_addr
        master_port = args.master_port
    
    # Validate configuration
    if rank != world_size - 1:
        print(f"Warning: Activation node rank {rank} should typically be the last rank ({world_size - 1})")
    
    if rank <= 0:
        print(f"Error: Activation node rank must be > 0 (rank 0 is reserved for master)")
        return
    
    if num_input_workers < 1:
        print(f"Error: Must have at least 1 input worker (current world_size: {world_size})")
        return
    
    if activation_size < 1:
        print(f"Error: Activation size must be >= 1")
        return
    
    if DEBUG:
        print(f"\n🚀 Starting micro-batching enhanced activation node with configuration - FIXED VERSION:")
        print(f"   Rank: {rank}")
        print(f"   World size: {world_size}")
        print(f"   Master: {master_addr}:{master_port}")
        print(f"   Number of input workers: {num_input_workers}")
        print(f"   Activation size (configurable): {activation_size}")
        print(f"   Activation function: {activation_function}")
        print(f"   Debug mode: {DEBUG}")
        print(f"   MICRO-BATCHING: Matrix operations for computational efficiency")
        print(f"   FIXED: Bias ensures activation variety across images")
        print(f"   NOTE: Ensure input workers use --weight-matrix-size {activation_size}")
    
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
        # Run the micro-batching enhanced activation node
        activation_process(rank, world_size, num_input_workers, activation_size, activation_function, master_addr, master_port)
    except KeyboardInterrupt:
        debug_print("🛑 Activation node interrupted by user")
    except Exception as e:
        if DEBUG:
            print(f"\n❌ Activation node failed: {e}")

if __name__ == "__main__":
    # Handle different launch methods
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        if DEBUG:
            print("No arguments provided. Starting interactive setup...")
        sys.argv.append('--interactive')
    
    main()