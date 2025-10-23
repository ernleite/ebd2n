#!/bin/bash
################################################################################
# EBD2N Dynamic Configuration Init Script
# 
# This script sets up environment variables that will be automatically read
# by the Python scripts. You can override these with command-line arguments.
#
# Priority: Command-line args > Environment variables > Hardcoded defaults
################################################################################

# Network Interface (for Gloo backend)
export GLOO_SOCKET_IFNAME="eno1"

# Master Node Configuration
# IMPORTANT: Set this to the actual IP address of your master node
export MASTER_ADDR="192.168.1.191"  # ← Change this to your master's IP (e.g., 10.150.0.17)
export MASTER_PORT="12355"

# Distributed Training Configuration
export WORLD_SIZE=7                 # Total number of nodes (master + workers)
export NUM_INPUT_WORKERS=2          # ← CRITICAL: Number of input layer workers (MUST be set!)
export RANK=1                       # This node's rank (set appropriately per node)

# IMPORTANT: NUM_INPUT_WORKERS determines network topology:
#   - Input workers: ranks 1 to NUM_INPUT_WORKERS
#   - Activation: rank NUM_INPUT_WORKERS + 1
#   - Weighted workers: ranks NUM_INPUT_WORKERS + 2 to WORLD_SIZE - 2
#   - Output: rank WORLD_SIZE - 1
#
# For NUM_INPUT_WORKERS=2, WORLD_SIZE=7:
#   Rank 0: Master
#   Ranks 1-2: Input Workers (2)
#   Rank 3: Activation
#   Ranks 4-5: Weighted Workers (2)
#   Rank 6: Output

# Training Parameters
export BATCH_SIZE=50
export MICRO_BATCH_SIZE=5
export ACTIVATION_SIZE=100
export OUTPUT_DIMENSION=10

# Debug Settings
export DEBUG="true"                 # Set to "false" to disable verbose output

# NCCL Settings (if using NCCL backend)
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO              # Set to WARN or ERROR for less output

################################################################################
# Helper Functions
################################################################################

function check_master_reachable() {
    echo "Checking if master node is reachable..."
    if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$MASTER_ADDR/$MASTER_PORT" 2>/dev/null; then
        echo "✓ Master node is reachable at $MASTER_ADDR:$MASTER_PORT"
        return 0
    else
        echo "✗ Cannot reach master node at $MASTER_ADDR:$MASTER_PORT"
        echo "  Please verify:"
        echo "    1. Master node is running"
        echo "    2. MASTER_ADDR is correct"
        echo "    3. MASTER_PORT is correct"
        echo "    4. No firewall blocking the port"
        return 1
    fi
}

function show_config() {
    echo "=================================="
    echo "EBD2N Configuration"
    echo "=================================="
    echo "Master Node:"
    echo "  MASTER_ADDR=$MASTER_ADDR"
    echo "  MASTER_PORT=$MASTER_PORT"
    echo ""
    echo "Network Topology:"
    echo "  WORLD_SIZE=$WORLD_SIZE"
    echo "  NUM_INPUT_WORKERS=$NUM_INPUT_WORKERS"
    echo "  RANK=$RANK"
    echo ""
    echo "Training Parameters:"
    echo "  BATCH_SIZE=$BATCH_SIZE"
    echo "  MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE"
    echo "  ACTIVATION_SIZE=$ACTIVATION_SIZE"
    echo "  OUTPUT_DIMENSION=$OUTPUT_DIMENSION"
    echo ""
    echo "Network Interface:"
    echo "  GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
    echo ""
    echo "Debug:"
    echo "  DEBUG=$DEBUG"
    echo "=================================="
}

################################################################################
# Main
################################################################################

# Show configuration
show_config

# Optional: Check master reachability
if [ "$1" == "--check-master" ]; then
    check_master_reachable
fi

echo ""
echo "Environment variables have been set."
echo "You can now run:"
echo "  - Master: python master_node_dynamic.py"
echo "  - Worker: python input_worker_dynamic.py --rank \$RANK"
echo ""
echo "Or override with command-line args:"
echo "  python input_worker_dynamic.py --rank 1 --master-addr 10.150.0.17"
echo ""