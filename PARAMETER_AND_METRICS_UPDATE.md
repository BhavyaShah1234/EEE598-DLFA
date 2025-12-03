# Parameter Breakdown and IoU Metric Update

## Date: December 2, 2025

### Updates Applied

#### 1. **Detailed Architecture Parameter Breakdown** ✅

Added comprehensive parameter counting for each component of the Modified LISA architecture.

**Output Format:**
```
================================================================================
Architecture Parameter Breakdown
================================================================================
VLM Components:
  Vision Tower:          152,512,512 (  4.10%)
  Projector:              10,493,952 (  0.28%)
  Language Model:      3,373,547,520 ( 90.72%)
  LM Head:               131,334,144 (  3.53%)

SAM Components:
  Vision Encoder:         47,203,584 (  1.27%)
  Prompt Encoder:              6,220 (  0.00%)
  Mask Decoder:            2,108,132 (  0.06%)

Projection Layers:         1,312,256 (  0.04%)

--------------------------------------------------------------------------------
Total Parameters:      3,718,518,320
================================================================================
```

**Key Insights:**
- **VLM Language Model dominates**: 90.72% of all parameters (3.37B)
- **VLM Vision Tower**: 4.10% (152.5M parameters)
- **VLM LM Head**: 3.53% (131.3M parameters) 
- **SAM Vision Encoder**: 1.27% (47.2M parameters)
- **SAM Mask Decoder**: 0.06% (2.1M parameters)
- **Projection Layers**: 0.04% (1.3M parameters)
- **Total Architecture**: 3.72 billion parameters

**Trainable with LoRA:**
- Trainable parameters: 142,942,252 (3.84%)
- Frozen parameters: 3,575,576,068 (96.16%)

---

#### 2. **Standard IoU Metric Added** ✅

Added standard Intersection over Union (IoU) alongside existing cIoU and gIoU metrics.

**Three IoU Variants Now Tracked:**

1. **IoU (Standard)**: 
   ```
   IoU = Intersection / Union
   ```
   - Range: [0, 1] where 1 = perfect overlap
   - Most common segmentation metric
   
2. **cIoU (Cumulative IoU)**:
   ```
   cIoU = mean(IoU across all samples)
   ```
   - Primary metric used in LISA paper for ReasonSeg
   - Same as standard IoU for single predictions
   
3. **gIoU (Generalized IoU)**:
   ```
   gIoU = IoU - (Area(C) - Area(Union)) / Area(C)
   ```
   - Range: [-1, 1] where 1 = perfect overlap
   - Penalizes predictions where enclosing box is much larger than union
   - More informative than standard IoU when masks don't overlap

**Updated Metrics Output:**
```
Train Metrics:
  - IoU: 0.0000
  - cIoU: 0.0000
  - gIoU: -0.9551
  - Accuracy: 0.9575
  - Precision: 0.0000
  - Recall: 0.0000
  - F1-Score: 0.0000
  - AUROC: 0.0000

Val Metrics:
  - Loss: 15.8197
  - IoU: 0.0434
  - cIoU: 0.0434
  - gIoU: -0.3432
  - Accuracy: 0.4009
  - Precision: 0.5875
  - Recall: 0.0635
  - F1-Score: 0.0818
  - AUROC: 0.0000
```

---

#### 3. **Updated Progress Bars** ✅

Training and validation progress bars now show standard IoU instead of cIoU for cleaner output:

**Before:**
```
Epoch 1: 100% |██████| 2/2 [00:03<00:00, 1.87s/it, loss=0.8447, cIoU=0.0000, acc=0.9935, f1=0.0000]
```

**After:**
```
Epoch 1: 100% |██████| 2/2 [00:03<00:00, 1.83s/it, loss=0.8040, IoU=0.0000, acc=0.9955, f1=0.0000]
```

---

#### 4. **Enhanced Plotting** ✅

Updated the IoU metrics subplot to include all three IoU variants:

**Plot Updates:**
- **Subplot [1, 0]**: Now shows IoU (green), cIoU (blue), and gIoU (red)
- **Title**: Changed from "cIoU and gIoU" to "IoU, cIoU and gIoU"
- **Legend**: Smaller font size (8pt) to fit 6 lines (3 train + 3 val)

**Color Coding:**
- Green: Standard IoU
- Blue: cIoU (Cumulative IoU)
- Red: gIoU (Generalized IoU)
- Solid lines: Training metrics
- Dashed lines: Validation metrics

---

### Complete Metrics Tracked

**Per Epoch JSON Output:**
```json
{
  "epoch": 1,
  "train": {
    "loss": 5.5315,
    "bce": 2.5161,
    "dice": 0.9985,
    "IoU": 0.0,
    "cIoU": 0.0,
    "gIoU": -0.9551,
    "accuracy": 0.9575,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "auroc": 0.0
  },
  "val": {
    "val_loss": 15.8197,
    "val_IoU": 0.0434,
    "val_cIoU": 0.0434,
    "val_gIoU": -0.3432,
    "val_accuracy": 0.4009,
    "val_precision": 0.5875,
    "val_recall": 0.0635,
    "val_f1": 0.0818,
    "val_auroc": 0.0
  }
}
```

---

### Code Changes Summary

1. **Lines 356-402**: Added detailed parameter breakdown printing
   - Counts parameters for each component
   - Displays formatted table with percentages
   - Shows total architecture size

2. **Lines 816-828**: Added IoU tracking in `train_epoch()`
   - Initialize `total_iou` counter
   - Compute `batch_iou` alongside other metrics
   - Accumulate IoU values

3. **Lines 898-900**: Updated progress bar to show IoU

4. **Lines 904-918**: Added IoU to training metrics return dictionary

5. **Lines 941-951**: Added IoU tracking in `validate()`
   - Initialize `total_iou` counter
   - Compute and accumulate IoU

6. **Lines 974-976**: Updated validation progress bar

7. **Lines 980-992**: Added IoU to validation metrics return dictionary

8. **Lines 1094-1105**: Updated IoU subplot in plotting
   - Added green lines for standard IoU
   - Updated title and legend
   - Adjusted legend font size

9. **Lines 1357-1375**: Updated console logging
   - Display IoU alongside cIoU and gIoU
   - Show all three metrics for both train and validation

---

### Testing Results

**Test Run (1 epoch, 2 samples):**
```
Architecture Parameter Breakdown: ✅ Displayed correctly
Total Parameters: ✅ 3,718,518,320 
IoU Metric Tracking: ✅ Working
Progress Bars: ✅ Show IoU
JSON Files: ✅ Include IoU field
Plotting: ✅ Shows all 3 IoU variants
Console Output: ✅ Displays IoU
```

---

### Benefits

1. **Better Understanding**: Parameter breakdown helps identify where model capacity is concentrated
2. **Metric Completeness**: Standard IoU is the most common metric; now tracked alongside variants
3. **Comparison**: Can compare IoU, cIoU, and gIoU to understand different aspects of prediction quality
4. **Debugging**: Parameter counts help verify architecture and identify potential issues
5. **Optimization**: Knowing parameter distribution helps make informed decisions about which components to train/freeze

---

### Usage

The updates require no changes to command-line usage. All improvements are automatic:

```bash
# Standard training - now shows parameter breakdown and IoU
accelerate launch scripts/train_new2.py --use_lora --epochs 30
```

The parameter breakdown will be displayed during model initialization, and IoU metrics will be tracked throughout training automatically.

