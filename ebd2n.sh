#!/bin/bash
#SBATCH --job-name=ebd2n_training
#SBATCH --partition=gpu
#SBATCH --nodes=1                    # Start with 1 node for testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=ebd2n_%j.out
#SBATCH --error=ebd2n_%j.err

# Load PyTorch
module purge
module load aidl/pytorch/2.4.1-cuda12.1

echo "================================================"
echo "EBD2N Network Training"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "================================================"

# =============================================================================
# CONFIGURATION
# =============================================================================

export NUM_INPUT_WORKERS=2
export NUM_WEIGHTED_WORKERS=2
export BATCH_SIZE=64
export MICRO_BATCH_SIZE=8
export ACTIVATION_SIZE=128
export OUTPUT_DIMENSION=10
export WORLD_SIZE=7  # 1 + 2 + 1 + 2 + 1

# Get node hostname
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=12355
export DEBUG=true

echo "Configuration:"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "================================================"

# =============================================================================
# CREATE LOG DIRECTORY
# =============================================================================

LOG_DIR="logs_${SLURM_JOB_ID}"
mkdir -p $LOG_DIR
echo "Logs will be saved to: $LOG_DIR"
echo "================================================"

# =============================================================================
# LAUNCH COMPONENTS AS BACKGROUND PROCESSES
# =============================================================================

echo "Launching network components..."
echo ""

# 1. Master (Rank 0)
echo "→ Launching MASTER (rank 0)"
python3 epoch.py \
    --world-size=$WORLD_SIZE \
    --num-input-workers=$NUM_INPUT_WORKERS \
    --batch-size=$BATCH_SIZE \
    --micro-batch-size=$MICRO_BATCH_SIZE \
    --activation-size=$ACTIVATION_SIZE \
    --output-dimension=$OUTPUT_DIMENSION \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    > $LOG_DIR/master_rank0.log 2>&1 &

MASTER_PID=$!
echo "  PID: $MASTER_PID"
sleep 5  # Wait for master to initialize

# 2. Input Workers (Ranks 1-2)
for i in 1 2; do
    echo "→ Launching INPUT WORKER (rank $i)"
    python3 input_layer.py \
        --rank=$i \
        --world-size=$WORLD_SIZE \
        --num-input-workers=$NUM_INPUT_WORKERS \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        > $LOG_DIR/input_rank${i}.log 2>&1 &
    echo "  PID: $!"
    sleep 1
done

# 3. Activation Layer (Rank 3)
echo "→ Launching ACTIVATION LAYER (rank 3)"
python3 activation_layer.py \
    --rank=3 \
    --world-size=$WORLD_SIZE \
    --num-input-workers=$NUM_INPUT_WORKERS \
    --activation-size=$ACTIVATION_SIZE \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    > $LOG_DIR/activation_rank3.log 2>&1 &
echo "  PID: $!"
sleep 1

# 4. Weighted Workers (Ranks 4-5)
for rank in 4 5; do
    worker_id=$((rank - 4))
    echo "→ Launching WEIGHTED WORKER (rank $rank, worker_id $worker_id)"
    python3 weighted_layer.py \
        --rank=$rank \
        --world-size=$WORLD_SIZE \
        --worker-id=$worker_id \
        --num-weighted-workers=$NUM_WEIGHTED_WORKERS \
        --activation-size=$ACTIVATION_SIZE \
        --output-dimension=$OUTPUT_DIMENSION \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        > $LOG_DIR/weighted_rank${rank}.log 2>&1 &
    echo "  PID: $!"
    sleep 1
done

# 5. Output Layer (Rank 6)
echo "→ Launching OUTPUT LAYER (rank 6)"
python3 output_layer.py \
    --rank=6 \
    --world-size=$WORLD_SIZE \
    --num-weighted-workers=$NUM_WEIGHTED_WORKERS \
    --output-dimension=$OUTPUT_DIMENSION \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    > $LOG_DIR/output_rank6.log 2>&1 &
echo "  PID: $!"

echo ""
echo "================================================"
echo "All components launched!"
echo "================================================"
echo "Master PID: $MASTER_PID"
echo "Logs: $LOG_DIR/"
echo ""

# =============================================================================
# MONITOR AND WAIT
# =============================================================================

echo "Monitoring master process..."
echo "To view logs in real-time:"
echo "  tail -f $LOG_DIR/master_rank0.log"
echo ""

# Wait for master to complete
wait $MASTER_PID
EXIT_CODE=$?

echo ""
echo "================================================"
echo "Training Complete!"
echo "Exit code: $EXIT_CODE"
echo "================================================"

# Show summary from logs
echo ""
echo "=== Master Log Summary ==="
tail -30 $LOG_DIR/master_rank0.log

# Clean up any remaining processes
pkill -P $$ python3 2>/dev/null

exit $EXIT_CODE