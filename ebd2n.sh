#!/bin/bash
#SBATCH --job-name=ebd2n_training
#SBATCH --partition=gpu
#SBATCH --nodes=4                    # Number of compute nodes
#SBATCH --gres=gpu:a100:2            # 2 GPUs per node (adjust as needed)
#SBATCH --ntasks-per-node=1          # 1 task per node initially
#SBATCH --cpus-per-task=32           # CPUs per task
#SBATCH --time=24:00:00
#SBATCH --output=ebd2n_training_%j.out
#SBATCH --error=ebd2n_training_%j.err

# Load PyTorch module
module purge
module load aidl/pytorch/2.4.1-cuda12.1

echo "================================================"
echo "EBD2N Multi-Node Distributed Training"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "================================================"

# =============================================================================
# NETWORK TOPOLOGY CONFIGURATION
# =============================================================================
# Customize these parameters for your EBD2N network

export NUM_INPUT_WORKERS=2          # Number of input layer partitions
export NUM_WEIGHTED_WORKERS=2       # Number of weighted layer workers
export BATCH_SIZE=64                # Total batch size
export MICRO_BATCH_SIZE=8           # Micro-batch size for pipeline
export ACTIVATION_SIZE=128          # Activation layer dimension (d^[1])
export OUTPUT_DIMENSION=10          # Output dimension (e.g., 10 for MNIST)

# Calculate world size: 1 master + NUM_INPUT_WORKERS + 1 activation + NUM_WEIGHTED_WORKERS + 1 output
export WORLD_SIZE=$((1 + $NUM_INPUT_WORKERS + 1 + $NUM_WEIGHTED_WORKERS + 1))

echo "Network Topology:"
echo "  World Size: $WORLD_SIZE"
echo "  Input Workers: $NUM_INPUT_WORKERS (ranks 1-$NUM_INPUT_WORKERS)"
echo "  Activation Layer: rank $((NUM_INPUT_WORKERS + 1))"
echo "  Weighted Workers: $NUM_WEIGHTED_WORKERS (ranks $((NUM_INPUT_WORKERS + 2))-$((NUM_INPUT_WORKERS + 1 + NUM_WEIGHTED_WORKERS)))"
echo "  Output Layer: rank $((WORLD_SIZE - 1))"
echo "================================================"

# =============================================================================
# DISTRIBUTED TRAINING SETUP
# =============================================================================

# Get master node hostname (first node in the allocation)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355

# Enable debug mode
export DEBUG=true

echo "Master Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "================================================"

# =============================================================================
# VALIDATION
# =============================================================================

# Validate input partitioning (784 = 28*28 for MNIST)
INPUT_SIZE=784
if [ $((INPUT_SIZE % NUM_INPUT_WORKERS)) -ne 0 ]; then
    echo "ERROR: Input size ($INPUT_SIZE) must be divisible by NUM_INPUT_WORKERS ($NUM_INPUT_WORKERS)"
    exit 1
fi

# Validate output partitioning
if [ $((OUTPUT_DIMENSION % NUM_WEIGHTED_WORKERS)) -ne 0 ]; then
    echo "ERROR: OUTPUT_DIMENSION ($OUTPUT_DIMENSION) must be divisible by NUM_WEIGHTED_WORKERS ($NUM_WEIGHTED_WORKERS)"
    exit 1
fi

echo "✓ Validation passed"
echo "  Input partition size: $((INPUT_SIZE / NUM_INPUT_WORKERS))"
echo "  Output partition size: $((OUTPUT_DIMENSION / NUM_WEIGHTED_WORKERS))"
echo "================================================"

# =============================================================================
# LAUNCH NETWORK COMPONENTS
# =============================================================================

# Create log directory
mkdir -p logs_${SLURM_JOB_ID}

echo "Launching EBD2N network components..."
echo ""

# Launch Master (Epoch) Node - Rank 0
echo "→ Launching MASTER (rank 0) on $MASTER_ADDR"
srun --nodes=1 --ntasks=1 --exclusive \
    python3 epoch.py \
    --world-size=$WORLD_SIZE \
    --num-input-workers=$NUM_INPUT_WORKERS \
    --batch-size=$BATCH_SIZE \
    --micro-batch-size=$MICRO_BATCH_SIZE \
    --activation-size=$ACTIVATION_SIZE \
    --output-dimension=$OUTPUT_DIMENSION \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    > logs_${SLURM_JOB_ID}/master_rank0.log 2>&1 &

MASTER_PID=$!
echo "  PID: $MASTER_PID"
sleep 5  # Wait for master to initialize

# Launch Input Workers - Ranks 1 to NUM_INPUT_WORKERS
for i in $(seq 1 $NUM_INPUT_WORKERS); do
    RANK=$i
    echo "→ Launching INPUT WORKER (rank $RANK)"
    srun --nodes=1 --ntasks=1 --exclusive \
        python3 input_layer.py \
        --rank=$RANK \
        --world-size=$WORLD_SIZE \
        --num-input-workers=$NUM_INPUT_WORKERS \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        > logs_${SLURM_JOB_ID}/input_rank${RANK}.log 2>&1 &
    
    echo "  PID: $!"
    sleep 1
done

# Launch Activation Layer - Rank NUM_INPUT_WORKERS+1
ACTIVATION_RANK=$((NUM_INPUT_WORKERS + 1))
echo "→ Launching ACTIVATION LAYER (rank $ACTIVATION_RANK)"
srun --nodes=1 --ntasks=1 --exclusive \
    python3 activation_layer.py \
    --rank=$ACTIVATION_RANK \
    --world-size=$WORLD_SIZE \
    --num-input-workers=$NUM_INPUT_WORKERS \
    --activation-size=$ACTIVATION_SIZE \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    > logs_${SLURM_JOB_ID}/activation_rank${ACTIVATION_RANK}.log 2>&1 &

echo "  PID: $!"
sleep 1

# Launch Weighted Workers - Ranks (NUM_INPUT_WORKERS+2) to (WORLD_SIZE-2)
WEIGHTED_START=$((NUM_INPUT_WORKERS + 2))
WEIGHTED_END=$((WORLD_SIZE - 2))

for RANK in $(seq $WEIGHTED_START $WEIGHTED_END); do
    WORKER_ID=$((RANK - WEIGHTED_START))
    echo "→ Launching WEIGHTED WORKER (rank $RANK, worker_id $WORKER_ID)"
    srun --nodes=1 --ntasks=1 --exclusive \
        python3 weighted_layer.py \
        --rank=$RANK \
        --world-size=$WORLD_SIZE \
        --worker-id=$WORKER_ID \
        --num-weighted-workers=$NUM_WEIGHTED_WORKERS \
        --activation-size=$ACTIVATION_SIZE \
        --output-dimension=$OUTPUT_DIMENSION \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        > logs_${SLURM_JOB_ID}/weighted_rank${RANK}.log 2>&1 &
    
    echo "  PID: $!"
    sleep 1
done

# Launch Output Layer - Last Rank
OUTPUT_RANK=$((WORLD_SIZE - 1))
echo "→ Launching OUTPUT LAYER (rank $OUTPUT_RANK)"
srun --nodes=1 --ntasks=1 --exclusive \
    python3 output_layer.py \
    --rank=$OUTPUT_RANK \
    --world-size=$WORLD_SIZE \
    --num-weighted-workers=$NUM_WEIGHTED_WORKERS \
    --output-dimension=$OUTPUT_DIMENSION \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    > logs_${SLURM_JOB_ID}/output_rank${OUTPUT_RANK}.log 2>&1 &

echo "  PID: $!"

echo ""
echo "================================================"
echo "All components launched!"
echo "================================================"
echo "Monitoring master process (PID: $MASTER_PID)..."
echo "Logs available in: logs_${SLURM_JOB_ID}/"
echo ""

# Wait for master to finish
wait $MASTER_PID
EXIT_CODE=$?

echo ""
echo "================================================"
echo "Training Complete!"
echo "Exit code: $EXIT_CODE"
echo "================================================"

# Show final statistics from logs
echo ""
echo "=== Final Statistics ==="
grep -h "Statistics\|Epoch\|Loss\|Accuracy" logs_${SLURM_JOB_ID}/*.log | tail -20

exit $EXIT_CODE