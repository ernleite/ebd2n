# EPOCH / MASTER NODE 

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

def setup_distributed(rank, world_size, master_addr="192.168.1.191", master_port="12355"):
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
        # Initialize the process group with longer timeout
        dist.init_process_group(
            backend="gloo",  # More reliable for CPU-based communication
            rank=rank, 
            init_method=f"tcp://{master_addr}:{master_port}", 
            world_size=world_size,
            timeout=timedelta(minutes=3)  # Longer timeout
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

def send_shutdown_signals(world_size, activation_rank):
    """Send shutdown signals to all workers including activation node"""
    debug_print("Sending shutdown signals to workers...")
    
    # Send to input layer workers (ranks 1 to activation_rank-1)
    for worker_rank in range(1, activation_rank):
        try:
            # ENHANCED: Send shutdown signal with special micro-batch size -999
            shutdown_microbatch_size = torch.tensor([-999], dtype=torch.long)
            shutdown_signal = torch.tensor([-999.0] * 10)  # Fixed size shutdown signal
            
            dist.send(shutdown_microbatch_size, dst=worker_rank)
            dist.send(shutdown_signal, dst=worker_rank)
            debug_print(f"✓ Shutdown signal sent to input worker {worker_rank}")
        except Exception as e:
            debug_print(f"Warning: Could not send shutdown to input worker {worker_rank}: {e}")

def create_master_engine(world_size, splits, batch_size, micro_batch_size, activation_rank):
    """Create the master node engine with MICRO-BATCHING enhancement"""
    
    # Store dataset globally to avoid reloading
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    data_iter = iter(data_loader)
    
    # Pre-calculate split size for efficiency
    vector_size = 784  # 28x28 MNIST images
    split_size = vector_size // splits
    
    # Batch tracking variables
    current_batch = []
    batch_number = 1
    
    # MICRO-BATCHING: Track pending micro-batches for proper synchronization
    pending_microbatches = {}  # {microbatch_id: {'start_time': time, 'splits_sent': int, 'images': list}}
    
    # Timing variables
    global_start_time = time.time()
    batch_start_time = time.time()
    last_stats_time = time.time()
    
    def master_step(engine, batch):
        nonlocal current_batch, batch_number, batch_start_time, last_stats_time, pending_microbatches
        
        # Start batch timing if this is the first image of the batch
        if len(current_batch) == 0:
            batch_start_time = time.time()
        
        try:
            # MICRO-BATCHING: Collect micro_batch_size images
            micro_batch_images = []
            micro_batch_labels = []
            micro_batch_indices = []
            
            microbatch_start_time = time.time()
            
            for i in range(micro_batch_size):
                try:
                    # Get next image from iterator
                    image, label = next(data_iter)
                    image = image.squeeze(0)  # Remove batch dimension
                    
                    # Transform 28x28 matrix to 784x1 vector
                    vector = image.view(-1, 1)  # Reshape to (784, 1)
                    
                    # Use engine iteration * micro_batch_size + i as unique image index
                    image_index = (engine.state.iteration - 1) * micro_batch_size + i + 1
                    
                    micro_batch_images.append(vector.squeeze())  # Store as 1D tensor (784,)
                    micro_batch_labels.append(label.item() if hasattr(label, 'item') else label)
                    micro_batch_indices.append(image_index)
                    
                except StopIteration:
                    # End of dataset reached
                    if not micro_batch_images:  # No images collected
                        return {
                            'status': 'complete',
                            'message': f'All images processed. Total batches: {batch_number}',
                            'final_batch_size': len(current_batch) if current_batch else 0,
                            'total_processing_time': time.time() - global_start_time,
                            'total_images': (batch_number - 1) * batch_size + len(current_batch),
                            'processing_rate': ((batch_number - 1) * batch_size + len(current_batch)) / (time.time() - global_start_time) if (time.time() - global_start_time) > 0 else 0,
                            'avg_time_per_image': (time.time() - global_start_time) / ((batch_number - 1) * batch_size + len(current_batch)) if ((batch_number - 1) * batch_size + len(current_batch)) > 0 else 0,
                            'pending_count': len(pending_microbatches)
                        }
                    else:
                        # Process remaining images in incomplete micro-batch
                        break
            
            if not micro_batch_images:
                return {'status': 'no_data'}
            
            # Create micro-batch matrices
            # Stack images into matrix: (micro_batch_size, 784)
            micro_batch_matrix = torch.stack(micro_batch_images)  # Shape: (micro_batch_size, 784)
            
            # Generate unique micro-batch ID
            microbatch_id = engine.state.iteration
            
            if DEBUG and engine.state.iteration <= 3:  # Only show details for first few micro-batches
                print(f"[Master] Processing micro-batch {microbatch_id}:")
                print(f"[Master]   Micro-batch size: {len(micro_batch_images)}")
                print(f"[Master]   Matrix shape: {micro_batch_matrix.shape}")
                print(f"[Master]   Image indices: {micro_batch_indices}")
                print(f"[Master]   Vector size: {vector_size}")
                print(f"[Master]   Split size: {split_size}")
                print(f"[Master]   Number of splits: {splits}")
            
            # MICRO-BATCHING: Track this micro-batch as pending
            pending_microbatches[microbatch_id] = {
                'start_time': microbatch_start_time,
                'splits_sent': 0,
                'images': micro_batch_images,
                'labels': micro_batch_labels,
                'indices': micro_batch_indices,
                'micro_batch_size': len(micro_batch_images)
            }
            
            # MICRO-BATCHING: Split the micro-batch matrix and send each split to corresponding input worker
            for i in range(splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < splits - 1 else vector_size
                
                # Extract split from all images in the micro-batch
                # micro_batch_split shape: (micro_batch_size, split_size)
                micro_batch_split = micro_batch_matrix[:, start_idx:end_idx]
                
                # Pad split to fixed size for consistency if needed
                if micro_batch_split.size(1) < split_size:
                    padding_size = split_size - micro_batch_split.size(1)
                    padding = torch.zeros(micro_batch_split.size(0), padding_size)
                    micro_batch_split = torch.cat([micro_batch_split, padding], dim=1)
                
                # Send to corresponding input worker (rank = i + 1)
                worker_rank = i + 1
                
                try:
                    # MICRO-BATCHING: Send micro-batch size, micro-batch ID, and micro-batch split
                    microbatch_size_tensor = torch.tensor([len(micro_batch_images)], dtype=torch.long)
                    microbatch_id_tensor = torch.tensor([microbatch_id], dtype=torch.long)
                    
                    dist.send(microbatch_size_tensor, dst=worker_rank)
                    dist.send(microbatch_id_tensor, dst=worker_rank)
                    dist.send(micro_batch_split.flatten(), dst=worker_rank)  # Flatten for transmission
                    
                    pending_microbatches[microbatch_id]['splits_sent'] += 1
                    
                    if DEBUG and engine.state.iteration <= 2:  # Show details for first few micro-batches
                        print(f"[Master]   Split {i} -> Input Worker {worker_rank}: microbatch_id={microbatch_id}, size={len(micro_batch_images)}, shape={micro_batch_split.shape}")
                    
                except Exception as send_error:
                    debug_print(f"✗ Communication error with input worker rank {worker_rank} for micro-batch {microbatch_id}: {send_error}")
                    # Remove from pending if failed to send
                    if microbatch_id in pending_microbatches:
                        del pending_microbatches[microbatch_id]
                    raise send_error
            
            # Now receive the final activation result from the activation node
            try:
                # MICRO-BATCHING: Receive micro-batch size, micro-batch ID, then activation results
                received_microbatch_size = torch.zeros(1, dtype=torch.long)
                dist.recv(received_microbatch_size, src=activation_rank)
                received_size = received_microbatch_size.item()
                
                received_microbatch_id = torch.zeros(1, dtype=torch.long)
                dist.recv(received_microbatch_id, src=activation_rank)
                received_id = received_microbatch_id.item()
                
                # Receive flattened activation results and reshape
                total_activation_elements = received_size * 100  # activation_size = 100
                activation_results_flat = torch.zeros(total_activation_elements)
                dist.recv(activation_results_flat, src=activation_rank)
                
                # Reshape to (micro_batch_size, 100)
                activation_results = activation_results_flat.view(received_size, 100)
                
                if DEBUG and engine.state.iteration <= 2:
                    print(f"[Master]   Received activation results for micro-batch {received_id} from rank {activation_rank}: shape={activation_results.shape}, sum={torch.sum(activation_results).item():.4f}")
                
                # MICRO-BATCHING: Verify this is the micro-batch we expect
                if received_id != microbatch_id:
                    debug_print(f"⚠️  Micro-batch ID mismatch! Sent: {microbatch_id}, Received: {received_id}")
                    # Handle out-of-order processing - this is actually normal in async processing
                
                # Clean up pending tracking and get original data
                if received_id in pending_microbatches:
                    original_images = pending_microbatches[received_id]['images']
                    original_labels = pending_microbatches[received_id]['labels']
                    original_indices = pending_microbatches[received_id]['indices']
                    del pending_microbatches[received_id]
                else:
                    # Fallback - use current micro-batch if tracking failed
                    original_images = micro_batch_images
                    original_labels = micro_batch_labels
                    original_indices = micro_batch_indices
                    debug_print(f"⚠️  No pending record for micro-batch {received_id}, using current micro-batch")
                
            except Exception as recv_error:
                debug_print(f"✗ Communication error with activation node rank {activation_rank} for micro-batch {microbatch_id}: {recv_error}")
                # Clean up pending tracking
                if microbatch_id in pending_microbatches:
                    del pending_microbatches[microbatch_id]
                raise recv_error
            
            microbatch_processing_time = time.time() - microbatch_start_time
            
            # MICRO-BATCHING: Store results for all images in the micro-batch
            for idx, (original_image, original_label, original_index, activation_result) in enumerate(
                zip(original_images, original_labels, original_indices, activation_results)):
                
                image_result = {
                    'image_index': original_index,
                    'microbatch_id': received_id,
                    'microbatch_position': idx,
                    'original_image': original_image,
                    'label': original_label,
                    'vector_shape': original_image.shape,
                    'activation_result': activation_result.clone(),  # Store the activation vector
                    'activation_sum': torch.sum(activation_result).item(),
                    'activation_mean': torch.mean(activation_result).item(),
                    'activation_max': torch.max(activation_result).item(),
                    'total_splits': splits,
                    'split_size': split_size,
                    'processing_time': microbatch_processing_time / len(original_images),  # Approximate per-image time
                    'microbatch_match': received_id == microbatch_id,
                    'microbatch_size': len(original_images)
                }
                
                current_batch.append(image_result)
            
            # Show progress indicator every 10 micro-batches or every 5 seconds (only in debug mode)
            current_time = time.time()
            if DEBUG and (engine.state.iteration % 10 == 0 or (current_time - last_stats_time) >= 5.0):
                global_elapsed = current_time - global_start_time
                total_images_processed = len(current_batch) + (batch_number - 1) * batch_size
                rate = total_images_processed / global_elapsed if global_elapsed > 0 else 0
                pending_count = len(pending_microbatches)
                debug_print(f"Processed {engine.state.iteration} micro-batches ({total_images_processed} images) | Rate: {rate:.2f} img/s | Elapsed: {global_elapsed:.1f}s | Pending: {pending_count}")
                last_stats_time = current_time
            
            # Check if batch is complete
            if len(current_batch) >= batch_size:
                # Calculate batch timing
                batch_end_time = time.time()
                batch_processing_time = batch_end_time - batch_start_time
                
                # Calculate global elapsed time
                global_elapsed_time = time.time() - global_start_time
                
                # MICRO-BATCHING: Calculate micro-batch match statistics
                microbatch_matches = sum(1 for img in current_batch if img['microbatch_match'])
                match_rate = (microbatch_matches / len(current_batch)) * 100
                
                # Always show batch completion time (concise format)
                stats_print(f"BATCH {batch_number} COMPLETED: {batch_size} images | Time: {batch_processing_time:.2f}s | Micro-batch matches: {microbatch_matches}/{batch_size} ({match_rate:.1f}%)")
                
                # Calculate batch statistics for debug mode
                if DEBUG:
                    activation_sums = [img['activation_sum'] for img in current_batch]
                    activation_means = [img['activation_mean'] for img in current_batch]
                    batch_total = sum(activation_sums)
                    batch_avg = batch_total / len(activation_sums)
                    total_images_so_far = batch_number * batch_size
                    overall_rate = total_images_so_far / global_elapsed_time if global_elapsed_time > 0 else 0
                
                # Display detailed batch results and statistics only in debug mode
                if DEBUG:
                    print(f"\n{'='*80}")
                    print(f"DETAILED BATCH {batch_number} RESULTS (MICRO-BATCHING ENHANCED)")
                    print(f"{'='*80}")
                    
                    for i, img_result in enumerate(current_batch):
                        match_indicator = "✓" if img_result['microbatch_match'] else "✗"
                        print(f"Image {img_result['image_index']:3d} {match_indicator} | MBatch: {img_result['microbatch_id']} | Pos: {img_result['microbatch_position']} | Label: {img_result['label']} | Act_Sum: {img_result['activation_sum']:.4f} | Act_Mean: {img_result['activation_mean']:.4f} | Act_Max: {img_result['activation_max']:.4f}")
                    
                    print(f"{'-'*80}")
                    print(f"BATCH {batch_number} MICRO-BATCHING SUMMARY:")
                    print(f"  Total activation sum: {batch_total:.4f}")
                    print(f"  Average activation sum per image: {batch_avg:.4f}")
                    print(f"  Min activation sum: {min(activation_sums):.4f} | Max: {max(activation_sums):.4f}")
                    print(f"  Average activation mean: {sum(activation_means)/len(activation_means):.4f}")
                    print(f"  Micro-batch matches: {microbatch_matches}/{batch_size} ({match_rate:.1f}%)")
                    print(f"  Batch processing time: {batch_processing_time:.3f}s")
                    print(f"  Average time per image: {batch_processing_time/batch_size:.3f}s")
                    print(f"  Images processed so far: {total_images_so_far}")
                    print(f"  Global elapsed time: {global_elapsed_time:.3f}s")
                    print(f"  Overall rate: {overall_rate:.2f} images/second")
                    print(f"  Pending micro-batches: {len(pending_microbatches)}")
                    print(f"  MICRO-BATCHING: Reduced communication events by factor of {micro_batch_size}")
                    print(f"{'='*80}\n")
                
                # Prepare return data for this completed batch
                batch_data = {
                    'batch_number': batch_number,
                    'batch_size': len(current_batch),
                    'batch_results': current_batch.copy(),
                    'batch_activation_total': sum([img['activation_sum'] for img in current_batch]),
                    'batch_activation_average': sum([img['activation_sum'] for img in current_batch]) / len(current_batch),
                    'batch_processing_time': batch_processing_time,
                    'global_elapsed_time': global_elapsed_time,
                    'microbatch_match_rate': match_rate,
                    'pending_count': len(pending_microbatches),
                    'status': 'batch_complete'
                }
                
                # Reset for next batch
                current_batch = []
                batch_number += 1
                
                return batch_data
            else:
                # Batch not complete yet, continue silently
                return {
                    'batch_number': batch_number,
                    'images_in_batch': len(current_batch),
                    'remaining': batch_size - len(current_batch),
                    'pending_count': len(pending_microbatches),
                    'status': 'batch_in_progress'
                }
                    
        except StopIteration:
            # Calculate final statistics
            global_elapsed_time = time.time() - global_start_time
            total_images_processed = (batch_number - 1) * batch_size + len(current_batch)
            
            # Process remaining images in incomplete batch
            if current_batch:
                # Calculate final batch timing
                batch_end_time = time.time()
                batch_processing_time = batch_end_time - batch_start_time
                
                # MICRO-BATCHING: Calculate final micro-batch match statistics
                final_microbatch_matches = sum(1 for img in current_batch if img['microbatch_match'])
                final_match_rate = (final_microbatch_matches / len(current_batch)) * 100
                
                # Always show final batch completion time (concise format)
                stats_print(f"FINAL BATCH {batch_number} COMPLETED: {len(current_batch)} images | Time: {batch_processing_time:.2f}s | Micro-batch matches: {final_microbatch_matches}/{len(current_batch)} ({final_match_rate:.1f}%)")
                
                # Calculate final batch statistics for debug mode
                if DEBUG:
                    activation_sums = [img['activation_sum'] for img in current_batch]
                    batch_total = sum(activation_sums)
                    batch_avg = batch_total / len(activation_sums)
                
                if DEBUG:
                    print(f"\n{'='*80}")
                    print(f"FINAL BATCH {batch_number} MICRO-BATCHING RESULTS")
                    print(f"{'='*80}")
                    
                    for i, img_result in enumerate(current_batch):
                        match_indicator = "✓" if img_result['microbatch_match'] else "✗"
                        print(f"Image {img_result['image_index']:3d} {match_indicator} | MBatch: {img_result['microbatch_id']} | Pos: {img_result['microbatch_position']} | Label: {img_result['label']} | Act_Sum: {img_result['activation_sum']:.4f} | Act_Mean: {img_result['activation_mean']:.4f} | Act_Max: {img_result['activation_max']:.4f}")
                    
                    print(f"{'-'*80}")
                    print(f"FINAL MICRO-BATCHING PROCESSING SUMMARY:")
                    print(f"  Total images processed: {total_images_processed}")
                    print(f"  Total processing time: {global_elapsed_time:.3f}s")
                    print(f"  Overall average time per image: {global_elapsed_time/total_images_processed:.3f}s")
                    print(f"  Processing rate: {total_images_processed/global_elapsed_time:.2f} images/second")
                    print(f"  Total batches: {batch_number}")
                    print(f"  Final micro-batch match rate: {final_match_rate:.1f}%")
                    print(f"  Remaining pending micro-batches: {len(pending_microbatches)}")
                    print(f"  MICRO-BATCHING: Communication efficiency gain: {micro_batch_size}x")
                    print(f"{'='*80}\n")
            
            return {
                'status': 'complete',
                'message': f'All images processed. Total batches: {batch_number}',
                'final_batch_size': len(current_batch) if current_batch else 0,
                'total_processing_time': global_elapsed_time,
                'total_images': total_images_processed,
                'processing_rate': total_images_processed / global_elapsed_time if global_elapsed_time > 0 else 0,
                'avg_time_per_image': global_elapsed_time / total_images_processed if total_images_processed > 0 else 0,
                'pending_count': len(pending_microbatches)
            }
        except Exception as e:
            error_msg = f"Error in master step: {e}"
            stats_print(error_msg)
            if DEBUG:
                import traceback
                traceback.print_exc()
            return {
                'status': 'error',
                'error': str(e),
                'pending_count': len(pending_microbatches)
            }
    
    return Engine(master_step)

def run_master(splits=2, batch_size=50, micro_batch_size=5, activation_size=100, master_addr="192.168.1.191", master_port="12355"):
    """Run the master node with MICRO-BATCHING enhancement"""
    stats_print("STARTING ENHANCED DISTRIBUTED MASTER NODE WITH MICRO-BATCHING")
    
    # Configuration
    activation_rank = splits + 1  # Activation node rank comes after input workers
    world_size = splits + 2  # Master + input workers + activation node
    
    # Configuration - show only in debug mode
    if DEBUG:
        stats_print(f"Configuration: {splits} input splits | Activation size: {activation_size} | World size: {world_size} | Batch size: {batch_size} | Micro-batch size: {micro_batch_size} | Address: {master_addr}:{master_port}")
        print(f"[Master] Expected input worker ranks: {list(range(1, splits + 1))}")
        print(f"[Master] Expected activation node rank: {activation_rank}")
        print(f"[Master] MICRO-BATCHING: Sending {micro_batch_size} images per communication event")
        print(f"[Master] MICRO-BATCHING: Communication reduction factor: {micro_batch_size}x")
    else:
        stats_print(f"Starting with {splits} input workers + 1 activation node, batch size {batch_size}, micro-batch size {micro_batch_size}")
    
    actual_port = None
    
    try:
        # Setup distributed environment 
        actual_port = setup_distributed(rank=0, world_size=world_size, master_addr=master_addr, master_port=master_port)
        
        stats_print("🚀 Distributed setup completed successfully!")
        if DEBUG and actual_port != master_port:
            print(f"[Master] Note: Using port {actual_port} instead of {master_port}")
        
        # Give workers a moment to fully initialize
        debug_print("Waiting 2 seconds for workers to stabilize...")
        time.sleep(2)
        
        # Create engine
        engine = create_master_engine(world_size, splits, batch_size, micro_batch_size, activation_rank)
        
        # Add event handlers
        @engine.on(Events.ITERATION_COMPLETED)
        def log_results(engine):
            result = engine.state.output
            if result.get('status') == 'batch_complete':
                # Batch completed - results already displayed in master_step
                pass
            elif result.get('status') == 'batch_in_progress':
                # Progress already shown in master_step
                pass
            elif result.get('status') == 'complete':
                # Always show final results
                stats_print(f"🎉 ALL PROCESSING COMPLETE!")
                stats_print(f"Final message: {result.get('message', 'Done')}")
                
                # Display final performance summary - always show regardless of debug mode
                total_time = result.get('total_processing_time', 0)
                total_images = result.get('total_images', 0)
                processing_rate = result.get('processing_rate', 0)
                avg_time = result.get('avg_time_per_image', 0)
                pending_count = result.get('pending_count', 0)
                
                if total_time > 0 and total_images > 0:
                    stats_print(f"📊 FINAL PERFORMANCE SUMMARY:")
                    stats_print(f"  Total execution time: {total_time:.3f}s ({total_time/60:.2f} minutes)")
                    stats_print(f"  Total images processed: {total_images}")
                    stats_print(f"  Average processing time per image: {avg_time:.3f}s")
                    stats_print(f"  Processing rate: {processing_rate:.2f} images/second")
                    stats_print(f"  Processing rate: {processing_rate*60:.1f} images/minute")
                    stats_print(f"  MICRO-BATCHING: Communication events reduced by factor of {micro_batch_size}")
                    if pending_count > 0:
                        stats_print(f"  ⚠️  {pending_count} micro-batches remained pending at completion")
                else:
                    stats_print("⚠️  No performance data available")
                
                # Send shutdown signals before terminating
                send_shutdown_signals(world_size, activation_rank)
                engine.terminate()
            else:
                error_msg = result.get('error', 'Unknown error')
                pending_count = result.get('pending_count', 0)
                stats_print(f"Iteration {engine.state.iteration} failed: {error_msg}")
                if pending_count > 0:
                    stats_print(f"  {pending_count} micro-batches were pending at failure")
                send_shutdown_signals(world_size, activation_rank)
                engine.terminate()
        
        @engine.on(Events.COMPLETED)
        def on_complete(engine):
            debug_print("Engine processing completed!")
            cleanup_distributed()
        
        @engine.on(Events.EXCEPTION_RAISED)
        def handle_exception(engine, e):
            stats_print(f"Exception raised in engine: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            send_shutdown_signals(world_size, activation_rank)
            cleanup_distributed()
        
        # Create dummy data for the engine (MNIST has ~10k test images)
        # Adjust for micro-batching: each iteration processes micro_batch_size images
        max_iterations = 10000 // micro_batch_size  # Maximum micro-batches we can process
        dummy_data = [None] * max_iterations
        
        # Run the engine
        if DEBUG:
            stats_print(f"🏁 Starting MICRO-BATCHED MNIST processing with {world_size} nodes...")
            print(f"[Master] Input workers (ranks 1-{splits}) and activation node (rank {activation_rank}) should be running!")
            print(f"[Master] MICRO-BATCHING: Processing {micro_batch_size} images per iteration")
            print(f"[Master] Press Ctrl+C to stop gracefully")
            stats_print("-" * 60)
        else:
            stats_print("🏁 Starting MICRO-BATCHED MNIST processing...")
        
        engine.run(dummy_data, max_epochs=1)
        
    except KeyboardInterrupt:
        stats_print("🛑 Received interrupt signal. Shutting down gracefully...")
        if world_size:
            send_shutdown_signals(world_size, activation_rank)
        cleanup_distributed()
    except Exception as e:
        stats_print(f"❌ Error running master: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        if world_size:
            send_shutdown_signals(world_size, activation_rank)
        cleanup_distributed()
    
    stats_print("👋 Master node shutdown complete")

def main():
    """Main entry point with argument parsing"""
    global DEBUG
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Enhanced Distributed Master Node with Micro-Batching')
    parser.add_argument('--splits', type=int, default=2, help='Number of input splits/workers')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--micro-batch-size', type=int, default=5, help='Micro-batch size (images per communication event)')
    parser.add_argument('--activation-size', type=int, default=100, help='Size of activation layer')
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
    
    # Validate micro-batch size
    if args.micro_batch_size <= 0:
        print("Error: Micro-batch size must be > 0")
        return
    
    if args.micro_batch_size > args.batch_size:
        print("Warning: Micro-batch size larger than batch size, adjusting...")
        args.micro_batch_size = args.batch_size
    
    run_master(
        splits=args.splits,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        activation_size=args.activation_size,
        master_addr=args.master_addr,
        master_port=args.master_port
    )

if __name__ == "__main__":
    main()