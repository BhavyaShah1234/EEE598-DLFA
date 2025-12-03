#!/bin/bash

# Monitor training progress by showing recent metrics from training log

echo "==================================================================="
echo "LISA Training Progress Monitor"
echo "==================================================================="
echo ""

# Check if training log exists
if [ ! -f "training_log.txt" ]; then
    echo "Error: training_log.txt not found. Is training running?"
    exit 1
fi

# Show latest progress
echo "Latest Training Progress:"
echo "-------------------------------------------------------------------"
tail -n 5 training_log.txt | grep -E "(Epoch|loss|cIoU|gIoU|Results)"
echo ""

# Show epoch completion
echo "Epochs Completed:"
echo "-------------------------------------------------------------------"
grep "Epoch.*Results:" training_log.txt | tail -n 5
echo ""

# Show best model saves
echo "Best Model Checkpoints:"
echo "-------------------------------------------------------------------"
grep "Saved best model" training_log.txt | tail -n 5
echo ""

# Show memory usage if available
echo "Memory Usage:"
echo "-------------------------------------------------------------------"
grep "Peak Memory" training_log.txt | tail -n 1
echo ""

# Show current epoch from progress bar
echo "Current Status:"
echo "-------------------------------------------------------------------"
tail -n 1 training_log.txt
echo ""
echo "==================================================================="
echo "To watch live: tail -f training_log.txt"
echo "To stop training: pkill -f train_new2.py"
echo "==================================================================="
