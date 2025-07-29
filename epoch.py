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

def setup_distributed(rank, world_size, master_addr="192.168.1.191", master_port="12355"):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    print(f"Setting up distributed training:")
    print(f"  Rank: {rank}")
    print(f"  World size: {world_size}")
    print(f"  Master addr: {master_addr}")
    print(f"  Master port: {master_port}")
    
    if rank == 0:
        if not check_port_available(master_port):
            print(f"Warning: Port {master_port} may be in use")
    
    try:
        # Initialize the process group with timeout
        dist.init_process_group(
            backend="gloo", 
            rank=rank, 
            init_method=f"tcp://{master_addr}:{master_port}", 
            world_size=world_size,
            timeout=timedelta(minutes=0.3)
        )
        print(f"Successfully initialized process group for rank {rank}")
        
        # Test connection
        if rank == 0:
            print("Master node waiting for worker connections...")
            # Send test tensors to all workers
            for worker_rank in range(1, world_size):
                test_tensor = torch.tensor([1.0])
                dist.send(test_tensor, dst=worker_rank)
                print(f"Test connection successful - worker {worker_rank} connected!")
        else:
            # Receive test tensor
            test_tensor = torch.zeros(1)
            dist.recv(test_tensor, src=0)
            print(f"Test connection successful - worker rank {rank} connected to master!")
            
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        print("Make sure both nodes are running and network is accessible")
        raise

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def create_master_engine():
    """Create the master node engine"""
    
    # Configuration
    split = 2  # Number of splits for the vector
    batch_size = 25  # Number of images to process before displaying results
    world_size = split + 1  # Master (rank 0) + 4 workers (ranks 1-4)
    
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
    total_processing_time = 0.0
    
    def master_step(engine, batch):
        nonlocal current_batch, batch_number, batch_start_time, total_processing_time
        
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
            split_size = vector_size // split  # 784 / 4 = 196
            
            # Time individual image processing
            image_start_time = time.time()
            
            # Split the vector and send each split to its corresponding worker
            split_results = []
            for i in range(split):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < split - 1 else vector_size
                vector_split = vector[start_idx:end_idx]
                
                # Send vector split to corresponding worker node (rank = i + 1)
                worker_rank = i + 1  # Split 0 -> Worker rank 1, Split 1 -> Worker rank 2, etc.
                dist.send(vector_split, dst=worker_rank)
                
                # Receive result from the same worker node
                result = torch.zeros(1)
                dist.recv(result, src=worker_rank)
                
                split_results.append(result.item())
            
            image_processing_time = time.time() - image_start_time
            
            # Store image results in current batch
            image_result = {
                'image_index': engine.state.iteration,
                'original_image': image,
                'vector_shape': vector.shape,
                'split_results': split_results,
                'total_splits': split,
                'split_size': split_size,
                'sum_of_splits': sum(split_results),
                'processing_time': image_processing_time
            }
            
            current_batch.append(image_result)
            
            # Check if batch is complete
            if len(current_batch) >= batch_size:
                # Calculate batch timing
                batch_end_time = time.time()
                batch_processing_time = batch_end_time - batch_start_time
                total_processing_time += batch_processing_time
                
                # Calculate global elapsed time
                global_elapsed_time = time.time() - global_start_time
                
                # Display batch results
                print(f"\n{'='*80}")
                print(f"BATCH {batch_number} COMPLETED ({batch_size} images)")
                print(f"{'='*80}")
                
                for i, img_result in enumerate(current_batch):
                    splits_with_ranks = [f"R{j+1}:{img_result['split_results'][j]:.4f}" for j in range(len(img_result['split_results']))]
                    print(f"Image {img_result['image_index']:3d} | Splits: {splits_with_ranks} | Sum: {img_result['sum_of_splits']:.4f} | Time: {img_result['processing_time']:.3f}s")
                
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
                print(f"  Min image time: {min(batch_times):.3f}s | Max image time: {max(batch_times):.3f}s")
                print(f"  Images processed so far: {batch_number * batch_size}")
                print(f"  Global elapsed time: {global_elapsed_time:.3f}s")
                print(f"  Overall average time per image: {global_elapsed_time/(batch_number * batch_size):.3f}s")
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
                total_processing_time += batch_processing_time
                global_elapsed_time = time.time() - global_start_time
                
                print(f"\n{'='*80}")
                print(f"FINAL BATCH {batch_number} COMPLETED ({len(current_batch)} images)")
                print(f"{'='*80}")
                
                for i, img_result in enumerate(current_batch):
                    splits_with_ranks = [f"R{j+1}:{img_result['split_results'][j]:.4f}" for j in range(len(img_result['split_results']))]
                    print(f"Image {img_result['image_index']:3d} | Splits: {splits_with_ranks} | Sum: {img_result['sum_of_splits']:.4f} | Time: {img_result['processing_time']:.3f}s")
                
                batch_sums = [img['sum_of_splits'] for img in current_batch]
                batch_times = [img['processing_time'] for img in current_batch]
                batch_total = sum(batch_sums)
                batch_avg = batch_total / len(batch_sums)
                
                total_images_processed = (batch_number - 1) * batch_size + len(current_batch)
                
                print(f"{'-'*80}")
                print(f"FINAL BATCH {batch_number} SUMMARY:")
                print(f"  Total sum: {batch_total:.4f}")
                print(f"  Average per image: {batch_avg:.4f}")
                print(f"  Final batch processing time: {batch_processing_time:.3f}s")
                print(f"  Average time per image: {batch_processing_time/len(current_batch):.3f}s")
                print(f"GLOBAL PROCESSING SUMMARY:")
                print(f"  Total images processed: {total_images_processed}")
                print(f"  Total processing time: {global_elapsed_time:.3f}s")
                print(f"  Overall average time per image: {global_elapsed_time/total_images_processed:.3f}s")
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
            print(f"Error in master step: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    return Engine(master_step)

def run_master():
    """Run the master node"""
    print("Starting Master Node (192.168.1.191)")
    
    # Setup distributed environment 
    setup_distributed(rank=0, world_size=3)
    
    # Create engine
    engine = create_master_engine()
    
    # Add event handlers
    @engine.on(Events.ITERATION_COMPLETED)
    def log_results(engine):
        result = engine.state.output
        if result.get('status') == 'batch_complete':
            # Batch completed - results already displayed in master_step
            pass
        elif result.get('status') == 'batch_in_progress':
            # Show progress indicator for current batch
            print(f"Batch {result['batch_number']}: {result['images_in_batch']}/{result['images_in_batch'] + result['remaining']} images processed", end='\r')
        elif result.get('status') == 'complete':
            print(f"\n🎉 ALL PROCESSING COMPLETE!")
            print(f"Final message: {result.get('message', 'Done')}")
            if result.get('final_batch_size', 0) > 0:
                print(f"Final incomplete batch had {result['final_batch_size']} images")
            
            # Display final timing summary
            total_time = result.get('total_processing_time', 0)
            total_images = result.get('total_images', 0)
            if total_time > 0 and total_images > 0:
                print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
                print(f"  Total execution time: {total_time:.3f}s ({total_time/60:.2f} minutes)")
                print(f"  Total images processed: {total_images}")
                print(f"  Average processing time per image: {total_time/total_images:.3f}s")
                print(f"  Processing rate: {total_images/total_time:.2f} images/second")
                print(f"  Processing rate: {(total_images/total_time)*60:.1f} images/minute")
            
            engine.terminate()  # Stop the engine
        else:
            print(f"Iteration {engine.state.iteration} failed: {result.get('error', 'Unknown error')}")
        
        # Only print separator for errors or completion
        if result.get('status') in ['error', 'complete']:
            print("=" * 60)
    
    @engine.on(Events.COMPLETED)
    def on_complete(engine):
        print("Master node processing completed!")
        cleanup_distributed()
    
    # Create dummy data for the engine (MNIST has ~10k test images)
    dummy_data = [None] * 10000  # Enough iterations for full dataset
    
    # Run the engine
    print("Master node ready. Starting processing...")
    engine.run(dummy_data, max_epochs=1)

if __name__ == "__main__":
    run_master()