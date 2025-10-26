#!/bin/bash
# Patch script to fix epoch.py initialization method
# Changes from file-based init to env-based init (TCP)

echo "================================================"
echo "EBD2N epoch.py Init Method Patcher"
echo "================================================"
echo ""

EPOCH_FILE="epoch.py"

# Check if epoch.py exists
if [ ! -f "$EPOCH_FILE" ]; then
    echo "ERROR: $EPOCH_FILE not found in current directory"
    exit 1
fi

echo "Found: $EPOCH_FILE"
echo ""

# Create backup
BACKUP_FILE="${EPOCH_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$EPOCH_FILE" "$BACKUP_FILE"
echo "✓ Created backup: $BACKUP_FILE"
echo ""

# Check if already patched
if grep -q 'init_method="env://"' "$EPOCH_FILE"; then
    echo "✓ File already uses env:// init method"
    echo "  No patching needed!"
    exit 0
fi

echo "Applying patch..."
echo ""

# Apply the patch
# Replace file-based init with env-based init
sed -i 's|init_method=f"file://{init_file}"|init_method="env://"|g' "$EPOCH_FILE"
sed -i 's|init_method=f'"'"'file://{init_file}'"'"'|init_method="env://"|g' "$EPOCH_FILE"

# Also ensure MASTER_ADDR and MASTER_PORT are set before init
if ! grep -q "os.environ\['MASTER_ADDR'\] = master_addr" "$EPOCH_FILE"; then
    echo "WARNING: Could not verify MASTER_ADDR is set in environment"
    echo "  You may need to manually add before dist.init_process_group():"
    echo "    os.environ['MASTER_ADDR'] = master_addr"
    echo "    os.environ['MASTER_PORT'] = master_port"
fi

# Verify the change
if grep -q 'init_method="env://"' "$EPOCH_FILE"; then
    echo "✓ Patch applied successfully!"
    echo ""
    echo "Changed:"
    echo '  FROM: init_method=f"file://{init_file}"'
    echo '  TO:   init_method="env://"'
    echo ""
    echo "This allows the master to open a TCP listening port"
    echo "that workers can connect to."
else
    echo "✗ Patch failed!"
    echo "  Please manually edit epoch.py"
    echo "  Find: init_method=f\"file://..."
    echo "  Replace with: init_method=\"env://\""
    exit 1
fi

echo ""
echo "================================================"
echo "Patching Complete"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Review the changes (optional):"
echo "     diff $BACKUP_FILE $EPOCH_FILE"
echo ""
echo "  2. Test with the corrected launch script:"
echo "     sbatch launch_ebd2n_corrected.sh"
echo ""
echo "  If something goes wrong, restore backup:"
echo "     cp $BACKUP_FILE $EPOCH_FILE"
echo ""