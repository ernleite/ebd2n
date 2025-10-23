#!/bin/bash
################################################################################
# EBD2N Worker Launcher
#
# This script starts all worker nodes in the correct sequence.
# IMPORTANT: Start the MASTER first, then run this script!
#
# Usage:
#   1. On master node: python epoch.py
#   2. Wait for "Waiting for all processes to join"
#   3. Run this script: ./start_workers.sh
################################################################################

set -e  # Exit on error

# Load configuration
if [ -f "init_ebd2n.sh" ]; then
    source init_ebd2n.sh
    echo "✓ Loaded configuration from init_ebd2n.sh"
else
    echo "❌ ERROR: init_ebd2n.sh not found!"
    echo "Please create init_ebd2n.sh or set environment variables manually"
    exit 1
fi

# Verify configuration
echo ""
echo "Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NUM_INPUT_WORKERS: $NUM_INPUT_WORKERS"
echo ""

# Check if master is reachable
echo "Checking master connectivity..."
if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$MASTER_ADDR/$MASTER_PORT" 2>/dev/null; then
    echo "✓ Master is reachable at $MASTER_ADDR:$MASTER_PORT"
else
    echo "❌ ERROR: Cannot reach master at $MASTER_ADDR:$MASTER_PORT"
    echo ""
    echo "Please ensure:"
    echo "  1. Master node (epoch.py) is running"
    echo "  2. Master is showing 'Waiting for all processes to join'"
    echo "  3. MASTER_ADDR is correct"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "STARTING EBD2N WORKERS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "⚠️  IMPORTANT: Master (epoch.py) must already be running!"
echo "   Master should show: 'Waiting for all 7 processes to join'"
echo ""
read -p "Is the master running and waiting? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please start master first: python epoch.py"
    exit 1
fi

echo ""
echo "Starting workers in sequence..."
echo "──────────────────────────────────────────────────────────────"

# Calculate topology
NUM_INPUT=$NUM_INPUT_WORKERS
ACTIVATION_RANK=$((NUM_INPUT + 1))
NUM_WEIGHTED=$((WORLD_SIZE - NUM_INPUT - 3))
OUTPUT_RANK=$((WORLD_SIZE - 1))

# Build weighted ranks list
WEIGHTED_START=$((ACTIVATION_RANK + 1))
WEIGHTED_END=$((WORLD_SIZE - 2))
WEIGHTED_RANKS="${WEIGHTED_START}"
for ((i=WEIGHTED_START+1; i<=WEIGHTED_END; i++)); do
    WEIGHTED_RANKS="${WEIGHTED_RANKS},${i}"
done

echo ""
echo "Topology:"
echo "  Input workers: ranks 1-${NUM_INPUT}"
echo "  Activation: rank ${ACTIVATION_RANK}"
echo "  Weighted workers: ranks ${WEIGHTED_START}-${WEIGHTED_END} (${NUM_WEIGHTED} workers)"
echo "  Output: rank ${OUTPUT_RANK}"
echo ""

# Start input workers
echo "▶ Starting input workers (ranks 1-${NUM_INPUT})..."
for ((rank=1; rank<=NUM_INPUT; rank++)); do
    echo "  Starting input worker rank ${rank}..."
    export RANK=$rank
    python input_layer.py > "input_worker_${rank}.log" 2>&1 &
    PID=$!
    echo "    PID: $PID (log: input_worker_${rank}.log)"
    sleep 2
done

# Start activation layer
echo "▶ Starting activation layer (rank ${ACTIVATION_RANK})..."
python activation_layer.py \
    --rank ${ACTIVATION_RANK} \
    --world-size ${WORLD_SIZE} \
    --activation-size 100 \
    --target-weighted-ranks "${WEIGHTED_RANKS}" \
    > "activation_layer.log" 2>&1 &
PID=$!
echo "  PID: $PID (log: activation_layer.log)"
sleep 2

# Start weighted workers
echo "▶ Starting weighted workers (ranks ${WEIGHTED_START}-${WEIGHTED_END})..."
for ((rank=WEIGHTED_START; rank<=WEIGHTED_END; rank++)); do
    shard_id=$((rank - WEIGHTED_START))
    echo "  Starting weighted worker rank ${rank} (shard ${shard_id})..."
    python weighted_layer.py \
        --rank ${rank} \
        --world-size ${WORLD_SIZE} \
        --layer-id 1 \
        --input-dimension 100 \
        --output-dimension 10 \
        --source-activation-rank ${ACTIVATION_RANK} \
        --target-output-rank ${OUTPUT_RANK} \
        --num-weighted-shards ${NUM_WEIGHTED} \
        > "weighted_worker_${rank}.log" 2>&1 &
    PID=$!
    echo "    PID: $PID (log: weighted_worker_${rank}.log)"
    sleep 2
done

# Start output layer
echo "▶ Starting output layer (rank ${OUTPUT_RANK})..."
python output_layer.py \
    --rank ${OUTPUT_RANK} \
    --world-size ${WORLD_SIZE} \
    --output-dimension 10 \
    --num-source-partitions ${NUM_WEIGHTED} \
    --source-weighted-ranks "${WEIGHTED_RANKS}" \
    > "output_layer.log" 2>&1 &
PID=$!
echo "  PID: $PID (log: output_layer.log)"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ All workers started!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Check master terminal for: '✓ All 7 processes have joined'"
echo "  2. Monitor logs:"
echo "       tail -f input_worker_*.log"
echo "       tail -f activation_layer.log"
echo "       tail -f weighted_worker_*.log"
echo "       tail -f output_layer.log"
echo ""
echo "To stop all workers:"
echo "  pkill -f 'input_layer.py|activation_layer.py|weighted_layer.py|output_layer.py'"
echo ""

# Show running processes
echo "Running EBD2N processes:"
ps aux | grep -E "(epoch|input_layer|activation_layer|weighted_layer|output_layer).py" | grep -v grep || echo "  (Use 'ps aux | grep python' to see all)"