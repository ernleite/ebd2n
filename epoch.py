# CORRECTED EPOCH / MASTER NODE FOR FULL EBD2N NETWORK TOPOLOGY
# Supports: Input → Activation → Weighted → Output → Epoch

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
            debug_print(f"✓ Shutdown signal sent to input worker {worker_rank}")
        except Exception as e:
            debug_print(f"Warning: Could not send shutdown to input worker {worker_rank}: {e}")
    
    # Send to activation layer
    try:
        shutdown_microbatch_size = torch.tensor([-999], dtype=torch.long)
        shutdown_microbatch_id = torch.tensor([-999], dtype=torch.long)
        shutdown_signal = torch.tensor([-999.0] * 10)
        
        dist.send(shutdown_microbatch_size, dst=activation_rank)
        dist.send(shutdown_microbatch_id, dst=activation_rank)
        dist.send(shutdown_signal, dst=activation_rank)
        debug_print(f"✓ Shutdown signal sent to activation layer {activation_rank}")
    except Exception as e:
        debug_print(f"Warning: Could not send shutdown to activation layer {activation_rank}: {e}")
    
    # Send to weighted layer workers
    for worker_rank in weighted_worker_ranks:
        try:
            shutdown_microbatch_size = torch.tensor([-999], dtype=torch.long)
            shutdown_microbatch_id = torch.tensor([-999], dtype=torch.long)
            shutdown_signal = torch.tensor([-999.0] * 10)
            
            dist.send(shutdown_microbatch_size, dst=worker_rank)
            dist.send(shutdown_microbatch_id, dst=worker_rank)
            dist.send(shutdown_signal, dst=worker_rank)
            debug_print(f"✓ Shutdown signal sent to weighted worker {worker_rank}")
        except Exception as e:
            debug_print(f"Warning: Could not send shutdown to weighted worker {worker_rank}: {e}")
    
    # Send to output layer
    try:
        shutdown_microbatch_size = torch.tensor([-999], dtype=torch.long)
        shutdown_microbatch_id = torch.tensor([-999], dtype=torch.long)
        shutdown_signal = torch.tensor([-999.0] * 10)
        
        dist.send(shutdown_microbatch_size, dst=output_rank)
        dist.send(shutdown_microbatch_id, dst=output_rank)
        dist.send(shutdown_signal, dst=output_rank)
        debug_print(f"✓ Shutdown signal sent to output layer {output_rank}")
    except Exception as e:
        debug_print(f"Warning: Could not send shutdown to output layer {output_rank}: {e}")

def create_master_engine(world_size, splits, batch_size, micro_batch_size, activation_size, output_dimension):
    """Create the master node engine for full EBD2N network topology"""
    
    # Store dataset globally to avoid reloading
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    data_iter = iter(data_loader)
    
    # Pre-calculate split size for efficiency
    vector_size = 784  # 28x28 MNIST images
    split_size = vector_size // splits
    
    # Calculate network topology ranks
    input_worker_ranks = list(range(1, splits + 1))  # ranks 1, 2
    activation_rank = splits + 1                     # rank 3
    num_weighted_workers = 2  # Fixed for this topology
    weighted_worker_ranks = list(range(activation_rank + 1, activation_rank + 1 + num_weighted_workers))  # ranks 4, 5
    output_rank = world_size - 1                     # rank 6
    
    if DEBUG:
        debug_print(f"EBD2N Network Topology:")
        debug_print(f"  Input workers: {input_worker_ranks}")
        debug_print(f"  Activation layer: {activation_rank}")
        debug_print(f"  Weighted workers: {weighted_worker_ranks}")
        debug_print(f"  Output layer: {output_rank}")
    
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
                
                # Send to corresponding input worker
                worker_rank = input_worker_ranks[i]
                
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
            
            # CORRECTED: Now receive the final SOFTMAX results from the OUTPUT LAYER
            try:
                # Receive micro-batch size, micro-batch ID, then softmax predictions from OUTPUT LAYER
                received_microbatch_size = torch.zeros(1, dtype=torch.long)
                dist.recv(received_microbatch_size, src=output_rank)
                received_size = received_microbatch_size.item()
                
                received_microbatch_id = torch.zeros(1, dtype=torch.long)
                dist.recv(received_microbatch_id, src=output_rank)
                received_id = received_microbatch_id.item()
                
                # Receive flattened softmax predictions and reshape
                total_prediction_elements = received_size * output_dimension  # (microbatch_size * num_classes)
                softmax_predictions_flat = torch.zeros(total_prediction_elements)
                dist.recv(softmax_predictions_flat, src=output_rank)
                
                # Reshape to (micro_batch_size, output_dimension)
                softmax_predictions = softmax_predictions_flat.view(received_size, output_dimension)
                
                if DEBUG and engine.state.iteration <= 2:
                    pred_sum = torch.sum(softmax_predictions).item()
                    pred_max = torch.max(softmax_predictions).item()
                    # Show predicted classes
                    _, predicted_classes = torch.max(softmax_predictions, dim=-1)
                    print(f"[Master]   Received SOFTMAX predictions for micro-batch {received_id} from OUTPUT LAYER rank {output_rank}:")
                    print(f"[Master]     Predictions shape: {softmax_predictions.shape}, sum: {pred_sum:.4f}, max: {pred_max:.4f}")
                    print(f"[Master]     Sample predicted classes: {predicted_classes[:min(5, len(predicted_classes))].tolist()}")
                
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
                debug_print(f"✗ Communication error with output layer rank {output_rank} for micro-batch {microbatch_id}: {recv_error}")
                # Clean up pending tracking
                if microbatch_id in pending_microbatches:
                    del pending_microbatches[microbatch_id]
                raise recv_error
            
            microbatch_processing_time = time.time() - microbatch_start_time
            
            # CORRECTED: Store SOFTMAX PREDICTIONS for all images in the micro-batch
            for idx, (original_image, original_label, original_index, softmax_prediction) in enumerate(
                zip(original_images, original_labels, original_indices, softmax_predictions)):
                
                # Get predicted class and confidence
                confidence, predicted_class = torch.max(softmax_prediction, dim=0)
                is_correct = predicted_class.item() == original_label
                
                image_result = {
                    'image_index': original_index,
                    'microbatch_id': received_id,
                    'microbatch_position': idx,
                    'original_image': original_image,
                    'true_label': original_label,
                    'predicted_class': predicted_class.item(),
                    'prediction_confidence': confidence.item(),
                    'is_correct': is_correct,
                    'softmax_predictions': softmax_prediction.clone(),  # Store the full softmax vector
                    'vector_shape': original_image.shape,
                    'total_splits': splits,
                    'split_size': split_size,
                    'processing_time': microbatch_processing_time / len(original_images),  # Approximate per-image time
                    'microbatch_match': received_id == microbatch_id,
                    'microbatch_size': len(original_images),
                    'output_dimension': output_dimension
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
                
                # CORRECTED: Calculate accuracy and prediction statistics
                correct_predictions = sum(1 for img in current_batch if img['is_correct'])
                accuracy = (correct_predictions / len(current_batch)) * 100
                microbatch_matches = sum(1 for img in current_batch if img['microbatch_match'])
                match_rate = (microbatch_matches / len(current_batch)) * 100
                
                # Always show batch completion time with accuracy
                stats_print(f"BATCH {batch_number} COMPLETED: {batch_size} images | Time: {batch_processing_time:.2f}s | Accuracy: {accuracy:.1f}% ({correct_predictions}/{batch_size}) | Micro-batch matches: {microbatch_matches}/{batch_size} ({match_rate:.1f}%)")
                
                # Display detailed batch results only in debug mode
                if DEBUG:
                    print(f"\n{'='*100}")
                    print(f"DETAILED BATCH {batch_number} RESULTS (FULL EBD2N NETWORK)")
                    print(f"{'='*100}")
                    
                    for i, img_result in enumerate(current_batch):
                        match_indicator = "✓" if img_result['microbatch_match'] else "✗"
                        correct_indicator = "✓" if img_result['is_correct'] else "✗"
                        print(f"Image {img_result['image_index']:3d} {match_indicator} {correct_indicator} | MBatch: {img_result['microbatch_id']} | True: {img_result['true_label']} | Pred: {img_result['predicted_class']} | Conf: {img_result['prediction_confidence']:.4f}")
                    
                    print(f"{'-'*100}")
                    print(f"BATCH {batch_number} EBD2N NETWORK SUMMARY:")
                    print(f"  Accuracy: {accuracy:.2f}% ({correct_predictions}/{batch_size} correct)")
                    print(f"  Average confidence: {sum(img['prediction_confidence'] for img in current_batch)/len(current_batch):.4f}")
                    print(f"  Micro-batch matches: {microbatch_matches}/{batch_size} ({match_rate:.1f}%)")
                    print(f"  Batch processing time: {batch_processing_time:.3f}s")
                    print(f"  Average time per image: {batch_processing_time/batch_size:.3f}s")
                    print(f"  Pending micro-batches: {len(pending_microbatches)}")
                    print(f"  EBD2N: Full network pipeline (Input → Activation → Weighted → Output → Softmax)")
                    print(f"  MICRO-BATCHING: Communication efficiency gain: {micro_batch_size}x")
                    print(f"{'='*100}\n")
                
                # Prepare return data for this completed batch
                batch_data = {
                    'batch_number': batch_number,
                    'batch_size': len(current_batch),
                    'batch_results': current_batch.copy(),
                    'batch_accuracy': accuracy,
                    'correct_predictions': correct_predictions,
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
                # Calculate final batch timing and accuracy
                batch_end_time = time.time()
                batch_processing_time = batch_end_time - batch_start_time
                final_correct = sum(1 for img in current_batch if img['is_correct'])
                final_accuracy = (final_correct / len(current_batch)) * 100
                final_microbatch_matches = sum(1 for img in current_batch if img['microbatch_match'])
                final_match_rate = (final_microbatch_matches / len(current_batch)) * 100
                
                # Always show final batch completion
                stats_print(f"FINAL BATCH {batch_number} COMPLETED: {len(current_batch)} images | Time: {batch_processing_time:.2f}s | Accuracy: {final_accuracy:.1f}% ({final_correct}/{len(current_batch)}) | Micro-batch matches: {final_microbatch_matches}/{len(current_batch)} ({final_match_rate:.1f}%)")
            
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

def run_master(splits=2, batch_size=50, micro_batch_size=5, activation_size=100, world_size=7, output_dimension=10, master_addr="192.168.1.191", master_port="12355"):
    """Run the master node for full EBD2N network topology"""
    stats_print("STARTING CORRECTED EBD2N EPOCH NODE FOR FULL NETWORK TOPOLOGY")
    
    # Calculate network topology ranks
    input_worker_ranks = list(range(1, splits + 1))  # ranks 1, 2
    activation_rank = splits + 1                     # rank 3
    num_weighted_workers = 2  # Fixed for this topology
    weighted_worker_ranks = list(range(activation_rank + 1, activation_rank + 1 + num_weighted_workers))  # ranks 4, 5
    output_rank = world_size - 1                     # rank 6
    
    # Configuration - show only in debug mode
    if DEBUG:
        stats_print(f"EBD2N Network Configuration:")
        stats_print(f"  World size: {world_size}")
        stats_print(f"  Input workers: {input_worker_ranks} (splits: {splits})")
        stats_print(f"  Activation layer: {activation_rank} (size: {activation_size})")
        stats_print(f"  Weighted workers: {weighted_worker_ranks}")
        stats_print(f"  Output layer: {output_rank} (classes: {output_dimension})")
        stats_print(f"  Batch size: {batch_size} | Micro-batch size: {micro_batch_size}")
        stats_print(f"  Address: {master_addr}:{master_port}")
        print(f"[Master] MICRO-BATCHING: Sending {micro_batch_size} images per communication event")
        print(f"[Master] MICRO-BATCHING: Communication reduction factor: {micro_batch_size}x")
        print(f"[Master] EBD2N: Full pipeline Input→Activation→Weighted→Output→Softmax")
    else:
        stats_print(f"Starting EBD2N network: {splits} input + 1 activation + {num_weighted_workers} weighted + 1 output | batch size {batch_size}")
    
    actual_port = None
    
    try:
        # Setup distributed environment 
        actual_port = setup_distributed(rank=0, world_size=world_size, master_addr=master_addr, master_port=master_port)
        
        stats_print("🚀 EBD2N distributed setup completed successfully!")
        if DEBUG and actual_port != master_port:
            print(f"[Master] Note: Using port {actual_port} instead of {master_port}")
        
        # Give workers a moment to fully initialize
        debug_print("Waiting 2 seconds for all EBD2N workers to stabilize...")
        time.sleep(2)
        
        # Create engine with full network topology
        engine = create_master_engine(world_size, splits, batch_size, micro_batch_size, activation_size, output_dimension)
        
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
                stats_print(f"🎉 EBD2N PROCESSING COMPLETE!")
                stats_print(f"Final message: {result.get('message', 'Done')}")
                
                # Display final performance summary
                total_time = result.get('total_processing_time', 0)
                total_images = result.get('total_images', 0)
                processing_rate = result.get('processing_rate', 0)
                avg_time = result.get('avg_time_per_image', 0)
                pending_count = result.get('pending_count', 0)
                
                if total_time > 0 and total_images > 0:
                    stats_print(f"📊 FINAL EBD2N PERFORMANCE SUMMARY:")
                    stats_print(f"  Total execution time: {total_time:.3f}s ({total_time/60:.2f} minutes)")
                    stats_print(f"  Total images processed: {total_images}")
                    stats_print(f"  Average processing time per image: {avg_time:.3f}s")
                    stats_print(f"  Processing rate: {processing_rate:.2f} images/second")
                    stats_print(f"  Processing rate: {processing_rate*60:.1f} images/minute")
                    stats_print(f"  EBD2N: Full network pipeline completed successfully")
                    stats_print(f"  MICRO-BATCHING: Communication events reduced by factor of {micro_batch_size}")
                    if pending_count > 0:
                        stats_print(f"  ⚠️  {pending_count} micro-batches remained pending at completion")
                else:
                    stats_print("⚠️  No performance data available")
                
                # Send shutdown signals to all workers before terminating
                send_shutdown_signals(world_size, input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank)
                engine.terminate()
            else:
                error_msg = result.get('error', 'Unknown error')
                pending_count = result.get('pending_count', 0)
                stats_print(f"Iteration {engine.state.iteration} failed: {error_msg}")
                if pending_count > 0:
                    stats_print(f"  {pending_count} micro-batches were pending at failure")
                send_shutdown_signals(world_size, input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank)
                engine.terminate()
        
        @engine.on(Events.COMPLETED)
        def on_complete(engine):
            debug_print("EBD2N engine processing completed!")
            cleanup_distributed()
        
        @engine.on(Events.EXCEPTION_RAISED)
        def handle_exception(engine, e):
            stats_print(f"Exception raised in EBD2N engine: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            send_shutdown_signals(world_size, input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank)
            cleanup_distributed()
        
        # Create dummy data for the engine
        max_iterations = 10000 // micro_batch_size  # Maximum micro-batches we can process
        dummy_data = [None] * max_iterations
        
        # Run the engine
        if DEBUG:
            stats_print(f"🏁 Starting EBD2N MNIST processing with {world_size} nodes...")
            print(f"[Master] EBD2N Network topology: Input({input_worker_ranks}) → Activation({activation_rank}) → Weighted({weighted_worker_ranks}) → Output({output_rank})")
            print(f"[Master] MICRO-BATCHING: Processing {micro_batch_size} images per iteration")
            print(f"[Master] Press Ctrl+C to stop gracefully")
            stats_print("-" * 80)
        else:
            stats_print("🏁 Starting EBD2N MNIST processing...")
        
        engine.run(dummy_data, max_epochs=1)
        
    except KeyboardInterrupt:
        stats_print("🛑 Received interrupt signal. Shutting down EBD2N network gracefully...")
        if world_size:
            send_shutdown_signals(world_size, input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank)
        cleanup_distributed()
    except Exception as e:
        stats_print(f"❌ Error running EBD2N master: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        if world_size:
            send_shutdown_signals(world_size, input_worker_ranks, activation_rank, weighted_worker_ranks, output_rank)
        cleanup_distributed()
    
    stats_print("👋 EBD2N master node shutdown complete")

def main():
    """Main entry point with argument parsing"""
    global DEBUG
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Corrected EBD2N Epoch Node for Full Network Topology')
    parser.add_argument('--world-size', type=int, default=7, help='Total world size including all nodes')
    parser.add_argument('--splits', type=int, default=2, help='Number of input splits/workers')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--micro-batch-size', type=int, default=5, help='Micro-batch size (images per communication event)')
    parser.add_argument('--activation-size', type=int, default=100, help='Size of activation layer')
    parser.add_argument('--output-dimension', type=int, default=10, help='Output dimension (number of classes)')
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
    
    # Validate arguments
    if args.micro_batch_size <= 0:
        print("Error: Micro-batch size must be > 0")
        return
    
    if args.micro_batch_size > args.batch_size:
        print("Warning: Micro-batch size larger than batch size, adjusting...")
        args.micro_batch_size = args.batch_size
    
    if args.world_size < 7:
        print(f"Error: World size must be at least 7 for full EBD2N topology (got {args.world_size})")
        return
    
    run_master(
        splits=args.splits,
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