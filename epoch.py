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

def create_master_engine():
    """Create the master node engine"""
    
    def master_step(engine, batch):
        try:
            # Load MNIST dataset
            transform = transforms.ToTensor()
            dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            
            for i, (image, label) in enumerate(data_loader):
                if i >= engine.state.iteration:
                    image = image.squeeze(0)  # Remove batch dimension
                    print(f"Master processing image {i} with shape: {image.shape}")
                    
                    # Send image matrix to worker node (rank 1)
                    print("Sending image to worker node...")
                    dist.send(image, dst=1)
                    print("Image sent successfully!")
                    
                    # Receive weighted sum from worker node
                    result = torch.zeros(1)  # Prepare tensor to receive the sum
                    print("Waiting for weighted sum from worker node...")
                    
                    # Add timeout for receiving
                    dist.recv(result, src=1)
                    print("Weighted sum received successfully!")
                    
                    print(f"Received weighted sum from worker: {result.item()}")
                    
                    yield {
                        'original_image': image,
                        'weighted_sum': result.item(),
                        'status': 'success'
                    }
                    
                    engine.state.iteration += 1
                    
        except Exception as e:
            print(f"Error in master step: {e}")
            yield {
                'status': 'error',
                'error': str(e)
            }
    
    return Engine(master_step)

def run_master():
    """Run the master node"""
    print("Starting Master Node (192.168.1.191)")
    
    # Setup distributed environment
    setup_distributed(rank=0, world_size=2)
    
    # Create engine
    engine = create_master_engine()
    
    # Add event handlers
    @engine.on(Events.ITERATION_COMPLETED)
    def log_results(engine):
        result = engine.state.output
        if result.get('status') == 'success':
            print(f"Iteration {engine.state.iteration} completed successfully")
            print(f"Weighted sum: {result['weighted_sum']}")
        else:
            print(f"Iteration {engine.state.iteration} failed: {result.get('error', 'Unknown error')}")
        print("-" * 50)
    
    @engine.on(Events.COMPLETED)
    def on_complete(engine):
        print("Master node processing completed!")
        cleanup_distributed()
    
    # Create dummy data for the engine (we just need something to iterate over)
    dummy_data = [None]  # Run until MNIST dataset is exhausted
    
    # Run the engine
    print("Master node ready. Starting processing...")
    engine.run(dummy_data, max_epochs=1)

if __name__ == "__main__":
    run_master()