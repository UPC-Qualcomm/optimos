# Quick Reference: Run Power Model with Your Files

## Your Files
- Compute log: `output/GPT_40B_16L_test_g2_my_sync_fix_64/FoldedClos_iter2/8_4_1_2_1.seq_2048.batch_512.log`
- Network CSV: `output/GPT_40B_16L_test_g2_my_sync_fix_64/FoldedClos_iter2/8_4_1_2_1.seq_2048.batch_512_link_traffic.csv`
- Power config: `power_model/a100_config.json` (with switch_degrees example)

## Command Examples

### 1. Analyze Mode A only (baseline)
```bash
python power_model/calculate_power.py \
  --compute "output/GPT_40B_16L_test_g2_my_sync_fix_64/FoldedClos_iter2/8_4_1_2_1.seq_2048.batch_512.log" \
  --network "output/GPT_40B_16L_test_g2_my_sync_fix_64/FoldedClos_iter2/8_4_1_2_1.seq_2048.batch_512_link_traffic.csv" \
  --config power_model/a100_config.json \
  --mode A
```

### 2. Compare all 4 modes (A, B, C, D) with default config
```bash
python power_model/calculate_power.py \
  --compute "output/GPT_40B_16L_test_g2_my_sync_fix_64/FoldedClos_iter2/8_4_1_2_1.seq_2048.batch_512.log" \
  --network "output/GPT_40B_16L_test_g2_my_sync_fix_64/FoldedClos_iter2/8_4_1_2_1.seq_2048.batch_512_link_traffic.csv" \
  --config power_model/a100_config.json
```

### 3. Save detailed results to JSON
```bash
python power_model/calculate_power.py \
  --compute "output/GPT_40B_16L_test_g2_my_sync_fix_64/FoldedClos_iter2/8_4_1_2_1.seq_2048.batch_512.log" \
  --network "output/GPT_40B_16L_test_g2_my_sync_fix_64/FoldedClos_iter2/8_4_1_2_1.seq_2048.batch_512_link_traffic.csv" \
  --config power_model/a100_config.json \
  --output power_results.json
```

### 4. With topology-aware switch power (if you have nodemap)
```bash
python power_model/calculate_power.py \
  --compute "output/GPT_40B_16L_test_g2_my_sync_fix_64/FoldedClos_iter2/8_4_1_2_1.seq_2048.batch_512.log" \
  --network "output/GPT_40B_16L_test_g2_my_sync_fix_64/FoldedClos_iter2/8_4_1_2_1.seq_2048.batch_512_link_traffic.csv" \
  --config power_model/a100_config.json \
  --nodemap topology/FoldedClos_iter2_nodemap.json
```

## What the CSV Provides

Your CSV has **per-link bandwidth** (in GB/s):
```
Source,Destination,Total_Bytes,Bandwidth (BG/s)
0,148,70258845696.00,1800.0         ← NVLink: 1800 GB/s
128,130,264355784704.00,400.0       ← Fabric: 400 GB/s
```

The parser **automatically**:
1. Reads each row's bandwidth
2. Converts GB/s → Bytes/s (multiply by 1e9)
3. Uses it to calculate per-link utilization:
   - `util = bytes_transmitted / (bandwidth * total_time)`
4. Accumulates per-switch utilization
5. Calculates switch dynamic power based on actual traffic

## Expected Output

For each mode, you'll see:
```
📊 EXECUTION PROFILE:
  Total Time:          X.XXX s
  Compute Time (mean): X.XXX s  util XX.X% [XX.X% – XX.X%]
  Comm Time (mean):    X.XXX s  util XX.X% [XX.X% – XX.X%]

⚡ POWER CONSUMPTION:
  Total Power:         XXX.XX W
    GPU Power:         XXX.XX W (XX.X%)
    Network Power:     XX.XX W (XX.X%)
      Links:           X.XX W
      Switches:        XX.XX W
        nvswitch       X.XX W  (8 units)
        tor            X.XX W  (16 units)
        [other types...]

🔋 ENERGY CONSUMPTION:
  Total Energy:        XXXXX.XX J
    GPU Energy:        XXXXX.XX J (XX.X%)
    Network Energy:    XXXX.XX J (XX.X%)

🎯 PERFORMANCE METRICS:
  Total Samples:       X,XXX,XXX
  Throughput:          X,XXX.XX samples/s

🏆 EFFICIENCY (MLPerf Power Metric):
  Samples per Joule:   X.XXXX samples/J
```

## Key Points

1. **Per-Link Bandwidth**: The parser reads the `Bandwidth (BG/s)` column from your CSV
2. **Per-Switch Utilization**: Each switch accumulates traffic from all links it touches
3. **Per-Switch Dynamic Power**: Only switches with traffic consume dynamic power
4. **Switch Degrees**: `a100_config.json` specifies 18 ports for NVSwitch, 64 for fabric switches
5. **4 Modes**: A (baseline), B (compute LPM), C (comm LPM), D (full LPM)

For more details, see [POWER_MODEL_USAGE.md](POWER_MODEL_USAGE.md).
