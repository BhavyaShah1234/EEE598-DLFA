#!/bin/bash

# Monitor training progress

echo "===================================================="
echo "Training Progress Monitor"
echo "===================================================="
echo ""

# Check if training is running
if pgrep -f "train_new.py" > /dev/null; then
    echo "✓ Training is RUNNING"
else
    echo "✗ Training is NOT running"
fi

echo ""
echo "Latest metrics:"
echo "----------------------------------------------------"
tail -30 training.log | grep -E "(Epoch|Loss|IoU|gIoU|cIoU|best|completed)"

echo ""
echo "===================================================="
echo "To view full log: tail -f training.log"
echo "To check GPU usage: nvidia-smi"
echo "To stop training: pkill -f train_new.py"
echo "===================================================="
