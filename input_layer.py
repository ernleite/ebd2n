import torch
import torch.distributed as dist
import os
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

def run_worker():
    """Run the worker node"""
    print("Starting Worker Node")
    
    # Setup distributed environment
    setup_distributed(rank=1, world_size=2)
    
    while True:
        try:
            # Receive image matrix from master node
            image = torch.zeros((28, 28))  # MNIST image size
            print("Waiting for image from master node...")
            dist.recv(image, src=0)
            print("Image received successfully!")
            
            # Calculate weighted sum (example: sum of all pixels)
            weighted_sum = torch.sum(image)
            print(f"Calculated weighted sum: {weighted_sum.item()}")
            
            # Send weighted sum back to master node
            print("Sending weighted sum to master node...")
            dist.send(weighted_sum, dst=0)
            print("Weighted sum sent successfully!")
            
        except Exception as e:
            print(f"Error in worker: {e}")
            break
    
    cleanup_distributed()

if __name__ == "__main__":
    run_worker()