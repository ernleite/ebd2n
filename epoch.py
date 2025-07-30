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
        print(f"[Master] Port {master_port} is busy, finding alternative...")
        master_port = find_available_port()
        print(f"[Master] Using port {master_port}")
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Optional: Force specific network interface if needed
    # os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
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
        print(f"[Master] ✓ Successfully initialized process group")
        
        # Wait for all processes to join
        print(f"[Master] Waiting for all {world_size} processes to join...")
        dist.barrier()  # This ensures all processes are ready
        print(f"[Master] ✓ All processes have joined successfully!")
        
        # Enhanced connection test with better error handling
        print("[Master] Testing worker connections...")
        for worker_rank in range(1, world_size):
            try:
                # Send connection test signal
                test_tensor = torch.tensor([float(worker_rank)])
                print(f"[Master] Sending connection test to worker rank {worker_rank}...")
                dist.send(test_tensor, dst=worker_rank)
                print(f"[Master] ✓ Connection test successful with worker rank {worker_rank}")
            except Exception as e:
                print(f"[Master] ✗ Connection test failed with worker rank {worker_rank}: {e}")
                raise ConnectionError(f"Cannot establish connection with worker {worker_rank}")
        
        print(f"[Master] ✓ All worker connections verified!")
        return master_port  # Return the actual port being used
        
    except Exception as e:
        print(f"\n[Master] ❌ FAILED TO INITIALIZE DISTRIBUTED TRAINING:")
        print(f"[Master] Error: {e}")
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
            print("[Master] ✓ Distributed cleanup completed")
        except:
            pass

def send_shutdown_signals(world_size):
    """Send shutdown signals to all workers"""
    print("[Master] Sending shutdown signals to workers...")
    for worker_rank in range(1, world_size):
        try:
            # Send shutdown signal (negative size indicates shutdown)
            shutdown_signal = torch.tensor([-1], dtype=torch.long)
            dist.send(shutdown_signal, dst=worker_rank)
            print(f"[Master] ✓ Shutdown signal sent to worker {worker_rank}")
        except Exception as e:
            print(f"[Master] Warning: Could not send shutdown to worker {worker_rank}: {e}")

def create_master_engine(world_size, splits, batch_size):
    """Create the master node engine with enhanced functionality"""
    
    # Store dataset globally to avoid reloading
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    data_iter = iter(data_loader)
    
    # Batch tracking variables
    current_batch = []
    batch_number = 1
    
    # Timing variables
    global_start_time = time.time()
    batch_start_time = time.time()
    
    def master_step(engine, batch):
        nonlocal current_batch, batch_number, batch_start_time
        
        # Start batch timing if this is the first image of the batch
        if len(current_batch) == 0:
            batch_start_time = time.time()
        
        try:
            # Get next image from iterator
            image, label = next(data_iter)
            image = image.squeeze(0)  # Remove batch dimension
            
            # Transform 28x28 matrix to 784x1 vector
            vector = image.view(-1, 1)  # Reshape to (784, 1)
            
            # Calculate split size
            vector_size = vector.size(0)  # 784
            split_size = vector_size // splits  # 784 / splits
            
            if engine.state.iteration <= 5:  # Only show details for first few images
                print(f"[Master] Processing image {engine.state.iteration}:")
                print(f"[Master]   Vector size: {vector_size}")
                print(f"[Master]   Split size: {split_size}")
                print(f"[Master]   Number of splits: {splits}")
            
            # Time individual image processing
            image_start_time = time.time()
            
            # Split the vector and send each split to its corresponding worker
            split_results = []
            for i in range(splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < splits - 1 else vector_size
                vector_split = vector[start_idx:end_idx]
                
                # Send vector split to corresponding worker node (rank = i + 1)
                worker_rank = i + 1
                
                try:
                    # Send the size of the vector split first
                    size_tensor = torch.tensor([vector_split.size(0)], dtype=torch.long)
                    dist.send(size_tensor, dst=worker_rank)
                    
                    # Send the actual vector split
                    dist.send(vector_split, dst=worker_rank)
                    
                    # Receive result from the same worker node
                    result = torch.zeros(1)
                    dist.recv(result, src=worker_rank)
                    split_results.append(result.item())
                    
                    if engine.state.iteration <= 3:  # Show details for first few images
                        print(f"[Master]   Split {i} -> Worker {worker_rank}: size={vector_split.size(0)}, result={result.item():.4f}")
                    
                except Exception as send_recv_error:
                    print(f"[Master] ✗ Communication error with worker rank {worker_rank}: {send_recv_error}")
                    raise send_recv_error
            
            image_processing_time = time.time() - image_start_time
            
            # Store image results in current batch
            image_result = {
                'image_index': engine.state.iteration,
                'original_image': image,
                'vector_shape': vector.shape,
                'split_results': split_results,
                'total_splits': splits,
                'split_size': split_size,
                'sum_of_splits': sum(split_results),
                'processing_time': image_processing_time
            }
            
            current_batch.append(image_result)
            
            # Show progress indicator
            if engine.state.iteration % 10 == 0:
                print(f"[Master] Processed {engine.state.iteration} images...", end='\r')
            
            # Check if batch is complete
            if len(current_batch) >= batch_size:
                # Calculate batch timing
                batch_end_time = time.time()
                batch_processing_time = batch_end_time - batch_start_time
                
                # Calculate global elapsed time
                global_elapsed_time = time.time() - global_start_time
                
                # Display batch results
                print(f"\n{'='*80}")
                print(f"BATCH {batch_number} COMPLETED ({batch_size} images)")
                print(f"{'='*80}")
                
                for i, img_result in enumerate(current_batch):
                    splits_with_ranks = [f"W{j+1}:{img_result['split_results'][j]:.4f}" for j in range(len(img_result['split_results']))]
                    print(f"Image {img_result['image_index']:3d} | {' '.join(splits_with_ranks)} | Sum: {img_result['sum_of_splits']:.4f} | Time: {img_result['processing_time']:.3f}s")
                
                # Calculate batch statistics
                batch_sums = [img['sum_of_splits'] for img in current_batch]
                batch_times = [img['processing_time'] for img in current_batch]
                batch_total = sum(batch_sums)
                batch_avg = batch_total / len(batch_sums)
                
                print(f"{'-'*80}")
                print(f"BATCH {batch_number} SUMMARY:")
                print(f"  Total sum: {batch_total:.4f}")
                print(f"  Average per image: {batch_avg:.4f}")
                print(f"  Min: {min(batch_sums):.4f} | Max: {max(batch_sums):.4f}")
                print(f"  Batch processing time: {batch_processing_time:.3f}s")
                print(f"  Average time per image: {batch_processing_time/batch_size:.3f}s")
                print(f"  Images processed so far: {batch_number * batch_size}")
                print(f"  Global elapsed time: {global_elapsed_time:.3f}s")
                print(f"  Overall rate: {(batch_number * batch_size)/global_elapsed_time:.2f} images/second")
                print(f"{'='*80}\n")
                
                # Prepare return data for this completed batch
                batch_data = {
                    'batch_number': batch_number,
                    'batch_size': len(current_batch),
                    'batch_results': current_batch.copy(),
                    'batch_total': batch_total,
                    'batch_average': batch_avg,
                    'batch_processing_time': batch_processing_time,
                    'global_elapsed_time': global_elapsed_time,
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
                    'status': 'batch_in_progress'
                }
                    
        except StopIteration:
            # Process remaining images in incomplete batch
            if current_batch:
                # Calculate final batch timing
                batch_end_time = time.time()
                batch_processing_time = batch_end_time - batch_start_time
                global_elapsed_time = time.time() - global_start_time
                
                print(f"\n{'='*80}")
                print(f"FINAL BATCH {batch_number} COMPLETED ({len(current_batch)} images)")
                print(f"{'='*80}")
                
                for i, img_result in enumerate(current_batch):
                    splits_with_ranks = [f"W{j+1}:{img_result['split_results'][j]:.4f}" for j in range(len(img_result['split_results']))]
                    print(f"Image {img_result['image_index']:3d} | {' '.join(splits_with_ranks)} | Sum: {img_result['sum_of_splits']:.4f} | Time: {img_result['processing_time']:.3f}s")
                
                batch_sums = [img['sum_of_splits'] for img in current_batch]
                batch_times = [img['processing_time'] for img in current_batch]
                batch_total = sum(batch_sums)
                batch_avg = batch_total / len(batch_sums)
                
                total_images_processed = (batch_number - 1) * batch_size + len(current_batch)
                
                print(f"{'-'*80}")
                print(f"FINAL PROCESSING SUMMARY:")
                print(f"  Total images processed: {total_images_processed}")
                print(f"  Total processing time: {global_elapsed_time:.3f}s")
                print(f"  Overall average time per image: {global_elapsed_time/total_images_processed:.3f}s")
                print(f"  Processing rate: {total_images_processed/global_elapsed_time:.2f} images/second")
                print(f"  Total batches: {batch_number}")
                print(f"{'='*80}\n")
            
            return {
                'status': 'complete',
                'message': f'All images processed. Total batches: {batch_number}',
                'final_batch_size': len(current_batch) if current_batch else 0,
                'total_processing_time': time.time() - global_start_time,
                'total_images': (batch_number - 1) * batch_size + (len(current_batch) if current_batch else 0)
            }
        except Exception as e:
            print(f"[Master] Error in master step: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'error': str(e)
            }
    
    return Engine(master_step)

def run_master():
    """Run the master node with enhanced error handling"""
    print("=" * 80)
    print("STARTING ENHANCED MASTER NODE")
    print("=" * 80)
    
    # Configuration
    splits = 4
    batch_size = 25
    world_size = splits + 1  # Master + workers
    master_addr = "192.168.1.191"
    master_port = "12355"
    
    print(f"Configuration:")
    print(f"  Splits: {splits}")
    print(f"  World size: {world_size}")
    print(f"  Expected worker ranks: {list(range(1, world_size))}")
    print(f"  Batch size: {batch_size}")
    print(f"  Master address: {master_addr}:{master_port}")
    
    actual_port = None
    
    try:
        # Setup distributed environment 
        actual_port = setup_distributed(rank=0, world_size=world_size, master_addr=master_addr, master_port=master_port)
        
        print(f"\n[Master] 🚀 Distributed setup completed successfully!")
        if actual_port != master_port:
            print(f"[Master] Note: Using port {actual_port} instead of {master_port}")
        
        # Give workers a moment to fully initialize
        print(f"[Master] Waiting 2 seconds for workers to stabilize...")
        time.sleep(2)
        
        # Create engine
        engine = create_master_engine(world_size, splits, batch_size)
        
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
                print(f"\n🎉 ALL PROCESSING COMPLETE!")
                print(f"Final message: {result.get('message', 'Done')}")
                
                # Display final performance summary
                total_time = result.get('total_processing_time', 0)
                total_images = result.get('total_images', 0)
                if total_time > 0 and total_images > 0:
                    print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
                    print(f"  Total execution time: {total_time:.3f}s ({total_time/60:.2f} minutes)")
                    print(f"  Total images processed: {total_images}")
                    print(f"  Average processing time per image: {total_time/total_images:.3f}s")
                    print(f"  Processing rate: {total_images/total_time:.2f} images/second")
                    print(f"  Processing rate: {(total_images/total_time)*60:.1f} images/minute")
                
                # Send shutdown signals before terminating
                send_shutdown_signals(world_size)
                engine.terminate()
            else:
                print(f"[Master] Iteration {engine.state.iteration} failed: {result.get('error', 'Unknown error')}")
                send_shutdown_signals(world_size)
                engine.terminate()
        
        @engine.on(Events.COMPLETED)
        def on_complete(engine):
            print("[Master] Engine processing completed!")
            cleanup_distributed()
        
        @engine.on(Events.EXCEPTION_RAISED)
        def handle_exception(engine, e):
            print(f"[Master] Exception raised in engine: {e}")
            import traceback
            traceback.print_exc()
            send_shutdown_signals(world_size)
            cleanup_distributed()
        
        # Create dummy data for the engine (MNIST has ~10k test images)
        dummy_data = [None] * 10000  # Enough iterations for full dataset
        
        # Run the engine
        print(f"\n[Master] 🏁 Starting MNIST processing with {world_size} nodes...")
        print(f"[Master] Workers should be running on expected machines!")
        print(f"[Master] Press Ctrl+C to stop gracefully")
        print("-" * 80)
        
        engine.run(dummy_data, max_epochs=1)
        
    except KeyboardInterrupt:
        print(f"\n[Master] 🛑 Received interrupt signal. Shutting down gracefully...")
        if world_size:
            send_shutdown_signals(world_size)
        cleanup_distributed()
    except Exception as e:
        print(f"\n[Master] ❌ Error running master: {e}")
        import traceback
        traceback.print_exc()
        if world_size:
            send_shutdown_signals(world_size)
        cleanup_distributed()
    
    print(f"[Master] 👋 Master node shutdown complete")

if __name__ == "__main__":
    run_master()