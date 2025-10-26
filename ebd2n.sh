#!/bin/bash
#SBATCH --job-name=ebd2n_training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=ebd2n_%j.out
#SBATCH --error=ebd2n_%j.err

# =============================================================================
# EBD2N DISTRIBUTED NETWORK TRAINING LAUNCHER
# Corrected version with proper initialization and arguments
# =============================================================================

# Load latest PyTorch module with Python 3.13
module purge
module load aidl/pytorch/2.6.0-cuda12.6

echo "================================================"
echo "EBD2N DISTRIBUTED NETWORK TRAINING"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Verify environment
echo "Environment Verification:"
python3 -c "import sys; print(f'  Python: {sys.version.split()[0]}')"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import ignite; print(f'  Ignite: {ignite.__version__}')"
echo "================================================"

# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================

# Network topology parameters
export NUM_INPUT_WORKERS=2
export NUM_WEIGHTED_WORKERS=2
export BATCH_SIZE=64
export MICRO_BATCH_SIZE=8
export ACTIVATION_SIZE=128
export OUTPUT_DIMENSION=10

# Calculate world size: master + input_workers + activation + weighted_workers + output
export WORLD_SIZE=$((1 + NUM_INPUT_WORKERS + 1 + NUM_WEIGHTED_WORKERS + 1))

# Distributed training configuration
# CRITICAL: Use actual node IP, not front node IP
export NODE_IP=$(ip addr show hsn0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
export MASTER_ADDR="${NODE_IP}"
export MASTER_PORT=29500

# Debug mode
export DEBUG=true

echo ""
echo "Network Configuration:"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NUM_INPUT_WORKERS: $NUM_INPUT_WORKERS"
echo "  NUM_WEIGHTED_WORKERS: $NUM_WEIGHTED_WORKERS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "  ACTIVATION_SIZE: $ACTIVATION_SIZE"
echo "  OUTPUT_DIMENSION: $OUTPUT_DIMENSION"
echo ""
echo "Distributed Setup:"
echo "  MASTER_ADDR: $MASTER_ADDR (HSN network)"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  Node hostname: $(hostname)"
echo "================================================"

# =============================================================================
# VALIDATION
# =============================================================================

echo ""
echo "Configuration Validation:"

# Validate input partitioning
INPUT_SIZE=784  # MNIST 28x28
if [ $((INPUT_SIZE % NUM_INPUT_WORKERS)) -ne 0 ]; then
    echo "  ✗ ERROR: Input size ($INPUT_SIZE) must be divisible by NUM_INPUT_WORKERS ($NUM_INPUT_WORKERS)"
    exit 1
fi
echo "  ✓ Input partitioning valid: $INPUT_SIZE / $NUM_INPUT_WORKERS = $((INPUT_SIZE / NUM_INPUT_WORKERS))"

# Validate output partitioning
if [ $((OUTPUT_DIMENSION % NUM_WEIGHTED_WORKERS)) -ne 0 ]; then
    echo "  ✗ ERROR: OUTPUT_DIMENSION ($OUTPUT_DIMENSION) must be divisible by NUM_WEIGHTED_WORKERS ($NUM_WEIGHTED_WORKERS)"
    exit 1
fi
echo "  ✓ Output partitioning valid: $OUTPUT_DIMENSION / $NUM_WEIGHTED_WORKERS = $((OUTPUT_DIMENSION / NUM_WEIGHTED_WORKERS))"

# Validate batch sizes
if [ $MICRO_BATCH_SIZE -gt $BATCH_SIZE ]; then
    echo "  ✗ ERROR: MICRO_BATCH_SIZE ($MICRO_BATCH_SIZE) must be <= BATCH_SIZE ($BATCH_SIZE)"
    exit 1
fi
echo "  ✓ Batch sizes valid: MICRO_BATCH_SIZE ($MICRO_BATCH_SIZE) <= BATCH_SIZE ($BATCH_SIZE)"

# Check if scripts exist
MISSING_SCRIPTS=0
for script in epoch.py input_layer.py activation_layer.py weighted_layer.py output_layer.py; do
    if [ ! -f "$script" ]; then
        echo "  ✗ ERROR: Missing script: $script"
        MISSING_SCRIPTS=1
    fi
done

if [ $MISSING_SCRIPTS -eq 1 ]; then
    echo "  Please ensure all Python scripts are in the current directory"
    exit 1
fi
echo "  ✓ All required scripts found"

echo "================================================"

# =============================================================================
# CREATE LOG DIRECTORY
# =============================================================================

LOG_DIR="logs_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
echo ""
echo "Log Directory: $LOG_DIR"
echo "================================================"

# =============================================================================
# CLEANUP FUNCTION
# =============================================================================

cleanup() {
    echo ""
    echo "================================================"
    echo "Cleaning up processes..."
    echo "================================================"
    
    # Kill all background python processes started by this script
    jobs -p | while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Killing process $pid"
            kill -TERM $pid 2>/dev/null || true
        fi
    done
    
    # Wait a moment
    sleep 2
    
    # Force kill any remaining
    jobs -p | while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Force killing process $pid"
            kill -KILL $pid 2>/dev/null || true
        fi
    done
    
    echo "  Cleanup complete"
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# =============================================================================
# LAUNCH NETWORK COMPONENTS
# =============================================================================

echo ""
echo "================================================"
echo "LAUNCHING EBD2N NETWORK COMPONENTS"
echo "================================================"
echo ""

# Calculate ranks
ACTIVATION_RANK=$((NUM_INPUT_WORKERS + 1))
OUTPUT_RANK=$((WORLD_SIZE - 1))

echo "Rank Assignment:"
echo "  Rank 0: Master/Epoch"
echo "  Ranks 1-$NUM_INPUT_WORKERS: Input Workers"
echo "  Rank $ACTIVATION_RANK: Activation Layer"
echo "  Ranks $((ACTIVATION_RANK + 1))-$((OUTPUT_RANK - 1)): Weighted Workers"
echo "  Rank $OUTPUT_RANK: Output Layer"
echo ""

# -----------------------------------------------------------------------------
# 1. LAUNCH MASTER (RANK 0)
# -----------------------------------------------------------------------------

echo "→ [1/7] Launching MASTER (rank 0)"
python3 epoch.py \
    --world-size=$WORLD_SIZE \
    --num-input-workers=$NUM_INPUT_WORKERS \
    --batch-size=$BATCH_SIZE \
    --micro-batch-size=$MICRO_BATCH_SIZE \
    --activation-size=$ACTIVATION_SIZE \
    --output-dimension=$OUTPUT_DIMENSION \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    > "$LOG_DIR/master_rank0.log" 2>&1 &

MASTER_PID=$!
echo "  PID: $MASTER_PID"
echo "  Log: $LOG_DIR/master_rank0.log"

# Wait for master to initialize and open TCP port
echo "  Waiting for master to initialize (10 seconds)..."
sleep 10

# Check if master is still running
if ! kill -0 $MASTER_PID 2>/dev/null; then
    echo ""
    echo "  ✗ ERROR: Master process died during initialization!"
    echo "  Check log file:"
    echo ""
    tail -30 "$LOG_DIR/master_rank0.log"
    exit 1
fi

echo "  ✓ Master is running"
echo ""

# -----------------------------------------------------------------------------
# 2. LAUNCH INPUT WORKERS (RANKS 1 to NUM_INPUT_WORKERS)
# -----------------------------------------------------------------------------

for i in $(seq 1 $NUM_INPUT_WORKERS); do
    RANK=$i
    echo "→ [$((RANK+1))/7] Launching INPUT WORKER (rank $RANK)"
    
    python3 input_layer.py \
        --rank=$RANK \
        --world-size=$WORLD_SIZE \
        --num-input-workers=$NUM_INPUT_WORKERS \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        > "$LOG_DIR/input_rank${RANK}.log" 2>&1 &
    
    echo "  PID: $!"
    echo "  Log: $LOG_DIR/input_rank${RANK}.log"
    sleep 3
    echo ""
done

# -----------------------------------------------------------------------------
# 3. LAUNCH ACTIVATION LAYER (RANK ACTIVATION_RANK)
# -----------------------------------------------------------------------------

echo "→ [4/7] Launching ACTIVATION LAYER (rank $ACTIVATION_RANK)"

python3 activation_layer.py \
    --rank=$ACTIVATION_RANK \
    --world-size=$WORLD_SIZE \
    --layer-id=1 \
    --activation-size=$ACTIVATION_SIZE \
    --activation-function=relu \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    > "$LOG_DIR/activation_rank${ACTIVATION_RANK}.log" 2>&1 &

echo "  PID: $!"
echo "  Log: $LOG_DIR/activation_rank${ACTIVATION_RANK}.log"
sleep 3
echo ""

# -----------------------------------------------------------------------------
# 4. LAUNCH WEIGHTED WORKERS (RANKS ACTIVATION_RANK+1 to OUTPUT_RANK-1)
# -----------------------------------------------------------------------------

WEIGHTED_START=$((ACTIVATION_RANK + 1))
WEIGHTED_END=$((OUTPUT_RANK - 1))

for RANK in $(seq $WEIGHTED_START $WEIGHTED_END); do
    SHARD_ID=$((RANK - WEIGHTED_START))
    echo "→ [$((RANK+1))/7] Launching WEIGHTED WORKER (rank $RANK, shard $SHARD_ID)"
    
    python3 weighted_layer.py \
        --rank=$RANK \
        --world-size=$WORLD_SIZE \
        --layer-id=2 \
        --input-dimension=$ACTIVATION_SIZE \
        --output-dimension=$OUTPUT_DIMENSION \
        --source-activation-rank=$ACTIVATION_RANK \
        --target-output-rank=$OUTPUT_RANK \
        --num-weighted-shards=$NUM_WEIGHTED_WORKERS \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        > "$LOG_DIR/weighted_rank${RANK}.log" 2>&1 &
    
    echo "  PID: $!"
    echo "  Log: $LOG_DIR/weighted_rank${RANK}.log"
    sleep 3
    echo ""
done

# -----------------------------------------------------------------------------
# 5. LAUNCH OUTPUT LAYER (RANK OUTPUT_RANK)
# -----------------------------------------------------------------------------

echo "→ [7/7] Launching OUTPUT LAYER (rank $OUTPUT_RANK)"

python3 output_layer.py \
    --rank=$OUTPUT_RANK \
    --world-size=$WORLD_SIZE \
    --num-weighted-workers=$NUM_WEIGHTED_WORKERS \
    --output-dimension=$OUTPUT_DIMENSION \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    > "$LOG_DIR/output_rank${OUTPUT_RANK}.log" 2>&1 &

echo "  PID: $!"
echo "  Log: $LOG_DIR/output_rank${OUTPUT_RANK}.log"
echo ""

# =============================================================================
# MONITORING AND WAITING
# =============================================================================

echo "================================================"
echo "ALL COMPONENTS LAUNCHED"
echo "================================================"
echo ""
echo "Master PID: $MASTER_PID"
echo "Total processes: $WORLD_SIZE"
echo ""
echo "To monitor in real-time (open another terminal):"
echo "  tail -f $LOG_DIR/master_rank0.log"
echo "  tail -f $LOG_DIR/input_rank1.log"
echo "  tail -f $LOG_DIR/activation_rank${ACTIVATION_RANK}.log"
echo ""
echo "================================================"
echo "WAITING FOR TRAINING TO COMPLETE"
echo "================================================"
echo ""

# Monitor component status periodically
MONITOR_INTERVAL=30
ELAPSED=0

while kill -0 $MASTER_PID 2>/dev/null; do
    if [ $ELAPSED -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] Training running... (${ELAPSED}s elapsed)"
        
        # Show last line from master log
        if [ -f "$LOG_DIR/master_rank0.log" ]; then
            LAST_LINE=$(tail -1 "$LOG_DIR/master_rank0.log" 2>/dev/null)
            if [ -n "$LAST_LINE" ]; then
                echo "  Last update: ${LAST_LINE:0:80}"
            fi
        fi
        echo ""
    fi
    
    sleep $MONITOR_INTERVAL
    ELAPSED=$((ELAPSED + MONITOR_INTERVAL))
done

# Get master exit code
wait $MASTER_PID
EXIT_CODE=$?

echo ""
echo "================================================"
echo "TRAINING COMPLETE"
echo "================================================"
echo "Exit code: $EXIT_CODE"
echo "Duration: ${ELAPSED} seconds"
echo ""

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

echo "================================================"
echo "MASTER LOG SUMMARY (last 50 lines)"
echo "================================================"
tail -50 "$LOG_DIR/master_rank0.log"
echo ""

echo "================================================"
echo "COMPONENT STATUS CHECK"
echo "================================================"

# Check each component's final status
for rank in 0 1 2 $ACTIVATION_RANK $WEIGHTED_START $((WEIGHTED_START + 1)) $OUTPUT_RANK; do
    LOG_FILE=$(ls "$LOG_DIR"/*rank${rank}.log 2>/dev/null | head -1)
    if [ -f "$LOG_FILE" ]; then
        LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)
        echo "Rank $rank: ${LAST_LINE:0:70}"
    fi
done

echo ""
echo "================================================"
echo "LOGS AVAILABLE IN: $LOG_DIR"
echo "================================================"
echo ""
echo "View individual logs:"
echo "  cat $LOG_DIR/master_rank0.log"
echo "  cat $LOG_DIR/input_rank1.log"
echo "  cat $LOG_DIR/activation_rank${ACTIVATION_RANK}.log"
echo "  cat $LOG_DIR/weighted_rank${WEIGHTED_START}.log"
echo "  cat $LOG_DIR/output_rank${OUTPUT_RANK}.log"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check master log: cat $LOG_DIR/master_rank0.log"
    echo "  2. Check worker logs for errors"
    echo "  3. Verify network connectivity and configuration"
fi

echo ""
echo "================================================"

exit $EXIT_CODE