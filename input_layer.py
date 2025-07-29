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
    
    # Configuration
    nextLayerSize = 100
    seed = 123
    
    # Setup distributed environment
    setup_distributed(rank=1, world_size=2)
    
    # Set random seed for reproducible weights
    torch.manual_seed(seed)
    print(f"Random seed set to {seed} for reproducible weights")
    
    weights_cache = {}  # Cache weights for different input sizes
    
    while True:
        try:
            # First, we need to determine the size of incoming vector
            # We'll receive vectors of size 196 (784/4) based on master configuration
            vector_size = 196  # This should match split_size from master
            
            # Receive vector from master node
            received_vector = torch.zeros(vector_size, 1)  # Shape: [196, 1]
            print(f"Waiting for vector from master node (expected size: {vector_size})...")
            dist.recv(received_vector, src=0)
            print(f"Vector received successfully! Shape: {received_vector.shape}")
            
            # Get actual input size from received vector
            input_size = received_vector.shape[0]  # Should be 196
            
            # Generate or retrieve weights matrix
            if input_size not in weights_cache:
                # Generate weights matrix of size [nextLayerSize, input_size]
                # Reset seed to ensure consistent weights for same input size
                torch.manual_seed(seed)
                weights = torch.randn(nextLayerSize, input_size)  # Shape: [100, 196]
                weights_cache[input_size] = weights
                print(f"Generated new weights matrix of shape: {weights.shape}")
            else:
                weights = weights_cache[input_size]
                print(f"Using cached weights matrix of shape: {weights.shape}")
            
            # Matrix multiplication: weights @ received_vector
            # [100, 196] @ [196, 1] = [100, 1]
            result_vector = torch.matmul(weights, received_vector)
            print(f"Matrix multiplication completed. Result shape: {result_vector.shape}")
            
            # Ensure result is the correct shape [nextLayerSize, 1]
            assert result_vector.shape == (nextLayerSize, 1), f"Result shape {result_vector.shape} != expected {(nextLayerSize, 1)}"
            
            # For sending back to master, we need to send a scalar (sum of the result vector)
            # or we can send the full vector - let's send the sum as a scalar to match master expectations
            result_sum = torch.sum(result_vector)
            print(f"Calculated result sum: {result_sum.item()}")
            
            # Send result back to master node
            print("Sending result to master node...")
            dist.send(result_sum, dst=0)
            print("Result sent successfully!")
            print("-" * 50)
            
        except RuntimeError as e:
            if "Connection closed by peer" in str(e) or "recv" in str(e):
                print("Master node disconnected. Shutting down worker...")
                break
            else:
                print(f"Runtime error in worker: {e}")
                break
        except Exception as e:
            print(f"Error in worker: {e}")
            break
    
    print("Worker shutting down...")
    cleanup_distributed()

if __name__ == "__main__":
    run_worker()