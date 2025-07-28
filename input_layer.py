import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ignite.engine import Engine, Events
import os
import time
import socket
from datetime import timedelta

def setup_distributed(rank, world_size, master_addr="192.168.1.191", master_port="12355"):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    print(f"Setting up distributed training:")
    print(f"  Rank: {rank}")
    print(f"  World size: {world_size}")
    print(f"  Master addr: {master_addr}")
    print(f"  Master port: {master_port}")
    
    try:
        # Initialize the process group with timeout
        dist.init_process_group(
            backend="gloo", 
            rank=rank, 
            world_size=world_size,
            init_method=f"tcp://{master_addr}:{master_port}", 
            timeout=timedelta(minutes=0.3)
        )
        print(f"Successfully initialized process group for rank {rank}")
        
        # Test connection
        if rank == 0:
            print("Master node waiting for worker connection...")
            # Send a test tensor
            test_tensor = torch.tensor([1.0])
            dist.send(test_tensor, dst=1)
            print("Test connection successful - worker connected!")
        else:
            # Receive test tensor
            test_tensor = torch.zeros(1)
            dist.recv(test_tensor, src=0)
            print("Test connection successful - connected to master!")
            
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        print("Make sure both nodes are running and network is accessible")
        raise

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def create_worker_engine():
    """Create the worker node engine"""
    
    def worker_step(engine, batch):
        try:
            # Prepare tensor to receive matrix from master
            received_matrix = torch.zeros(1000000, 1)
            
            print("Worker waiting for matrix from master...")
            # Receive matrix from master node (rank 0)
            dist.recv(received_matrix, src=0)
            print("Matrix received successfully!")
            
            print(f"Worker received matrix with shape: {received_matrix.shape}")
            print(f"Received matrix content (first 5 elements): {received_matrix[:5].flatten()}")
            
            # Compute the sum of the matrix
            matrix_sum = torch.sum(received_matrix)
            print(f"Worker computed sum: {matrix_sum.item()}")
            
            # Send result back to master node
            print("Sending sum result back to master...")
            dist.send(matrix_sum, dst=0)
            print("Result sent successfully!")
            
            return {
                'received_matrix': received_matrix,
                'computed_sum': matrix_sum.item(),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error in worker step: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    return Engine(worker_step)

def run_worker():
    """Run the worker node"""
    print("Starting Worker Node (192.168.1.233)")
    
    # Setup distributed environment
    setup_distributed(rank=1, world_size=2)
    
    # Create engine
    engine = create_worker_engine()
    
    # Add event handlers
    @engine.on(Events.ITERATION_COMPLETED)
    def log_results(engine):
        result = engine.state.output
        if result.get('status') == 'success':
            print(f"Worker iteration {engine.state.iteration} completed successfully")
            print(f"Computed sum: {result['computed_sum']}")
        else:
            print(f"Worker iteration {engine.state.iteration} failed: {result.get('error', 'Unknown error')}")
        print("-" * 50)
    
    @engine.on(Events.COMPLETED)
    def on_complete(engine):
        print("Worker node processing completed!")
        cleanup_distributed()
    
    # Create dummy data for the engine (we just need something to iterate over)
    dummy_data = [None] * 3  # Run 3 iterations to match master
    
    # Run the engine
    print("Worker node ready. Waiting for tasks...")
    engine.run(dummy_data, max_epochs=1)

if __name__ == "__main__":
    run_worker()

