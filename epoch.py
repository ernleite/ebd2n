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
            # ENHANCED: Send shutdown signal with special image index -999
            shutdown_image_index = torch.tensor([-999], dtype=torch.long)
            shutdown_signal = torch.tensor([-999.0] * 10)  # Fixed size shutdown signal
            
            dist.send(shutdown_image_index, dst=worker_rank)
            dist.send(shutdown_signal, dst=worker_rank)
            debug_print(f"✓ Shutdown signal sent to input worker {worker_rank}")
        except Exception as e:
            debug_print(f"Warning: Could not send shutdown to input worker {worker_rank}: {e}")

def create_master_engine(world_size, splits, batch_size, activation_rank):
    """Create the master node engine with enhanced image indexing"""
    
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
    
    # ENHANCED: Track pending images for proper synchronization
    pending_images = {}  # {image_index: {'start_time': time, 'splits_sent': int}}
    
    # Timing variables
    global_start_time = time.time()
    batch_start_time = time.time()
    last_stats_time = time.time()
    
    def master_step(engine, batch):
        nonlocal current_batch, batch_number, batch_start_time, last_stats_time, pending_images
        
        # Start batch timing if this is the first image of the batch
        if len(current_batch) == 0:
            batch_start_time = time.time()
        
        try:
            # Get next image from iterator
            image, label = next(data_iter)
            image = image.squeeze(0)  # Remove batch dimension
            
            # Transform 28x28 matrix to 784x1 vector
            vector = image.view(-1, 1)  # Reshape to (784, 1)
            
            # ENHANCED: Use engine iteration as image index (1-based)
            image_index = engine.state.iteration
            
            if DEBUG and engine.state.iteration <= 5:  # Only show details for first few images
                print(f"[Master] Processing image {image_index}:")
                print(f"[Master]   Vector size: {vector_size}")
                print(f"[Master]   Split size: {split_size}")
                print(f"[Master]   Number of splits: {splits}")
            
            # Time individual image processing
            image_start_time = time.time()
            
            # ENHANCED: Track this image as pending
            pending_images[image_index] = {
                'start_time': image_start_time,
                'splits_sent': 0,
                'original_image': image,
                'label': label
            }
            
            # OPTIMIZATION: Split the vector and send each split with image index to corresponding input worker
            for i in range(splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < splits - 1 else vector_size
                vector_split = vector[start_idx:end_idx]
                
                # Pad split to fixed size for consistency (optimization)
                if vector_split.size(0) < split_size:
                    padding = torch.zeros(split_size - vector_split.size(0), 1)
                    vector_split = torch.cat([vector_split, padding], dim=0)
                
                # Send vector split to corresponding input worker (rank = i + 1)
                worker_rank = i + 1
                
                try:
                    # ENHANCED: Send image index first, then vector split
                    image_index_tensor = torch.tensor([image_index], dtype=torch.long)
                    dist.send(image_index_tensor, dst=worker_rank)
                    dist.send(vector_split.squeeze(), dst=worker_rank)  # Send as 1D tensor
                    
                    pending_images[image_index]['splits_sent'] += 1
                    
                    if DEBUG and engine.state.iteration <= 3:  # Show details for first few images
                        print(f"[Master]   Split {i} -> Input Worker {worker_rank}: image_idx={image_index}, size={vector_split.size(0)}")
                    
                except Exception as send_error:
                    debug_print(f"✗ Communication error with input worker rank {worker_rank} for image {image_index}: {send_error}")
                    # Remove from pending if failed to send
                    if image_index in pending_images:
                        del pending_images[image_index]
                    raise send_error
            
            # Now receive the final activation result from the activation node
            try:
                # ENHANCED: Receive image index first, then activation result
                received_image_index = torch.zeros(1, dtype=torch.long)
                dist.recv(received_image_index, src=activation_rank)
                received_idx = received_image_index.item()
                
                activation_result = torch.zeros(100)  # Expecting 100-dimensional activation vector
                dist.recv(activation_result, src=activation_rank)
                
                if DEBUG and engine.state.iteration <= 3:
                    print(f"[Master]   Received activation result for image {received_idx} from rank {activation_rank}: shape={activation_result.shape}, sum={torch.sum(activation_result).item():.4f}")
                
                # ENHANCED: Verify this is the image we expect
                if received_idx != image_index:
                    debug_print(f"⚠️  Image index mismatch! Sent: {image_index}, Received: {received_idx}")
                    # Handle out-of-order processing - this is actually normal in async processing
                    # We'll process it anyway but log the discrepancy
                
                # Clean up pending tracking
                if received_idx in pending_images:
                    original_image = pending_images[received_idx]['original_image']
                    original_label = pending_images[received_idx]['label']
                    del pending_images[received_idx]
                else:
                    # Fallback - use current image if tracking failed
                    original_image = image
                    original_label = label
                    debug_print(f"⚠️  No pending record for image {received_idx}, using current image")
                
            except Exception as recv_error:
                debug_print(f"✗ Communication error with activation node rank {activation_rank} for image {image_index}: {recv_error}")
                # Clean up pending tracking
                if image_index in pending_images:
                    del pending_images[image_index]
                raise recv_error
            
            image_processing_time = time.time() - image_start_time
            
            # Store image results in current batch
            image_result = {
                'image_index': received_idx,  # Use the received index for consistency
                'sent_index': image_index,    # Track what we sent for debugging
                'original_image': original_image,
                'label': original_label.item() if hasattr(original_label, 'item') else original_label,
                'vector_shape': vector.shape,
                'activation_result': activation_result.clone(),  # Store the activation vector
                'activation_sum': torch.sum(activation_result).item(),
                'activation_mean': torch.mean(activation_result).item(),
                'activation_max': torch.max(activation_result).item(),
                'total_splits': splits,
                'split_size': split_size,
                'processing_time': image_processing_time,
                'index_match': received_idx == image_index
            }
            
            current_batch.append(image_result)
            
            # Show progress indicator every 10 images or every 5 seconds (only in debug mode)
            current_time = time.time()
            if DEBUG and (engine.state.iteration % 10 == 0 or (current_time - last_stats_time) >= 5.0):
                global_elapsed = current_time - global_start_time
                rate = engine.state.iteration / global_elapsed if global_elapsed > 0 else 0
                pending_count = len(pending_images)
                debug_print(f"Processed {engine.state.iteration} images | Rate: {rate:.2f} img/s | Elapsed: {global_elapsed:.1f}s | Pending: {pending_count}")
                last_stats_time = current_time
            
            # Check if batch is complete
            if len(current_batch) >= batch_size:
                # Calculate batch timing
                batch_end_time = time.time()
                batch_processing_time = batch_end_time - batch_start_time
                
                # Calculate global elapsed time
                global_elapsed_time = time.time() - global_start_time
                
                # ENHANCED: Calculate index match statistics
                index_matches = sum(1 for img in current_batch if img['index_match'])
                match_rate = (index_matches / len(current_batch)) * 100
                
                # Always show batch completion time (concise format)
                stats_print(f"BATCH {batch_number} COMPLETED: {batch_size} images | Time: {batch_processing_time:.2f}s | Index matches: {index_matches}/{batch_size} ({match_rate:.1f}%)")
                
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
                    print(f"DETAILED BATCH {batch_number} RESULTS (ENHANCED WITH IMAGE INDEXING)")
                    print(f"{'='*80}")
                    
                    for i, img_result in enumerate(current_batch):
                        match_indicator = "✓" if img_result['index_match'] else "✗"
                        print(f"Image {img_result['image_index']:3d} {match_indicator} | Label: {img_result['label']} | Act_Sum: {img_result['activation_sum']:.4f} | Act_Mean: {img_result['activation_mean']:.4f} | Act_Max: {img_result['activation_max']:.4f} | Time: {img_result['processing_time']:.3f}s")
                    
                    print(f"{'-'*80}")
                    print(f"BATCH {batch_number} ENHANCED SUMMARY:")
                    print(f"  Total activation sum: {batch_total:.4f}")
                    print(f"  Average activation sum per image: {batch_avg:.4f}")
                    print(f"  Min activation sum: {min(activation_sums):.4f} | Max: {max(activation_sums):.4f}")
                    print(f"  Average activation mean: {sum(activation_means)/len(activation_means):.4f}")
                    print(f"  Image index matches: {index_matches}/{batch_size} ({match_rate:.1f}%)")
                    print(f"  Batch processing time: {batch_processing_time:.3f}s")
                    print(f"  Average time per image: {batch_processing_time/batch_size:.3f}s")
                    print(f"  Images processed so far: {total_images_so_far}")
                    print(f"  Global elapsed time: {global_elapsed_time:.3f}s")
                    print(f"  Overall rate: {overall_rate:.2f} images/second")
                    print(f"  Pending images: {len(pending_images)}")
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
                    'index_match_rate': match_rate,
                    'pending_count': len(pending_images),
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
                    'pending_count': len(pending_images),
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
                
                # ENHANCED: Calculate final index match statistics
                final_index_matches = sum(1 for img in current_batch if img['index_match'])
                final_match_rate = (final_index_matches / len(current_batch)) * 100
                
                # Always show final batch completion time (concise format)
                stats_print(f"FINAL BATCH {batch_number} COMPLETED: {len(current_batch)} images | Time: {batch_processing_time:.2f}s | Index matches: {final_index_matches}/{len(current_batch)} ({final_match_rate:.1f}%)")
                
                # Calculate final batch statistics for debug mode
                if DEBUG:
                    activation_sums = [img['activation_sum'] for img in current_batch]
                    batch_total = sum(activation_sums)
                    batch_avg = batch_total / len(activation_sums)
                
                if DEBUG:
                    print(f"\n{'='*80}")
                    print(f"FINAL BATCH {batch_number} ENHANCED RESULTS")
                    print(f"{'='*80}")
                    
                    for i, img_result in enumerate(current_batch):
                        match_indicator = "✓" if img_result['index_match'] else "✗"
                        print(f"Image {img_result['image_index']:3d} {match_indicator} | Label: {img_result['label']} | Act_Sum: {img_result['activation_sum']:.4f} | Act_Mean: {img_result['activation_mean']:.4f} | Act_Max: {img_result['activation_max']:.4f} | Time: {img_result['processing_time']:.3f}s")
                    
                    print(f"{'-'*80}")
                    print(f"FINAL PROCESSING SUMMARY:")
                    print(f"  Total images processed: {total_images_processed}")
                    print(f"  Total processing time: {global_elapsed_time:.3f}s")
                    print(f"  Overall average time per image: {global_elapsed_time/total_images_processed:.3f}s")
                    print(f"  Processing rate: {total_images_processed/global_elapsed_time:.2f} images/second")
                    print(f"  Total batches: {batch_number}")
                    print(f"  Final index match rate: {final_match_rate:.1f}%")
                    print(f"  Remaining pending images: {len(pending_images)}")
                    print(f"{'='*80}\n")
            
            return {
                'status': 'complete',
                'message': f'All images processed. Total batches: {batch_number}',
                'final_batch_size': len(current_batch) if current_batch else 0,
                'total_processing_time': global_elapsed_time,
                'total_images': total_images_processed,
                'processing_rate': total_images_processed / global_elapsed_time if global_elapsed_time > 0 else 0,
                'avg_time_per_image': global_elapsed_time / total_images_processed if total_images_processed > 0 else 0,
                'pending_count': len(pending_images)
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
                'pending_count': len(pending_images)
            }
    
    return Engine(master_step)

def run_master(splits=2, batch_size=50, activation_size=100, master_addr="192.168.1.191", master_port="12355"):
    """Run the master node with enhanced image indexing"""
    stats_print("STARTING ENHANCED DISTRIBUTED MASTER NODE WITH IMAGE INDEXING")
    
    # Configuration
    activation_rank = splits + 1  # Activation node rank comes after input workers
    world_size = splits + 2  # Master + input workers + activation node
    
    # Configuration - show only in debug mode
    if DEBUG:
        stats_print(f"Configuration: {splits} input splits | Activation size: {activation_size} | World size: {world_size} | Batch size: {batch_size} | Address: {master_addr}:{master_port}")
        print(f"[Master] Expected input worker ranks: {list(range(1, splits + 1))}")
        print(f"[Master] Expected activation node rank: {activation_rank}")
        print(f"[Master] ENHANCEMENT: Image indexing for proper synchronization")
    else:
        stats_print(f"Starting with {splits} input workers + 1 activation node, batch size {batch_size}")
    
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
        engine = create_master_engine(world_size, splits, batch_size, activation_rank)
        
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
                    stats_print(f"  ENHANCEMENT: Image indexing ensures correct synchronization")
                    if pending_count > 0:
                        stats_print(f"  ⚠️  {pending_count} images remained pending at completion")
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
                    stats_print(f"  {pending_count} images were pending at failure")
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
        dummy_data = [None] * 10000  # Enough iterations for full dataset
        
        # Run the engine
        if DEBUG:
            stats_print(f"🏁 Starting ENHANCED MNIST processing with {world_size} nodes...")
            print(f"[Master] Input workers (ranks 1-{splits}) and activation node (rank {activation_rank}) should be running!")
            print(f"[Master] Press Ctrl+C to stop gracefully")
            stats_print("-" * 60)
        else:
            stats_print("🏁 Starting ENHANCED MNIST processing...")
        
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
    
    parser = argparse.ArgumentParser(description='Enhanced Distributed Master Node with Image Indexing')
    parser.add_argument('--splits', type=int, default=2, help='Number of input splits/workers')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
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
    
    run_master(
        splits=args.splits,
        batch_size=args.batch_size,
        activation_size=args.activation_size,
        master_addr=args.master_addr,
        master_port=args.master_port
    )

if __name__ == "__main__":
    main()