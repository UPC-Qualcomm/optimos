#!/bin/bash
# Example script demonstrating network configuration across different optimizers

echo "=================================================="
echo "Network Configuration Examples for Optimizers"
echo "=================================================="
echo ""

# Example 1: Bayesian Optimization with FoldedClos (default)
echo "Example 1: Bayesian Optimization with FoldedClos (default)"
echo "Command:"
echo "python Optimization/optimizers/bayesian_opt.py \\"
echo "    --model 10 \\"
echo "    --model_name 'GPT_175B_BO_FoldedClos' \\"
echo "    --num_npus 128 \\"
echo "    --budget 30 \\"
echo "    --init_samples 5"
echo ""

# Example 2: Bayesian Optimization with Switch network
echo "Example 2: Bayesian Optimization with Switch network"
echo "Command:"
echo "python Optimization/optimizers/bayesian_opt.py \\"
echo "    --model 10 \\"
echo "    --model_name 'GPT_175B_BO_Switch' \\"
echo "    --num_npus 128 \\"
echo "    --budget 30 \\"
echo "    --init_samples 5 \\"
echo "    --network Switch"
echo ""

# Example 3: Random Search with Ring network
echo "Example 3: Random Search with Ring network"
echo "Command:"
echo "python Optimization/optimizers/random_opt.py \\"
echo "    --model 10 \\"
echo "    --model_name 'GPT_175B_RS_Ring' \\"
echo "    --num_npus 128 \\"
echo "    --iterations 30 \\"
echo "    --network Ring"
echo ""

# Example 4: Random Search with FullyConnected network
echo "Example 4: Random Search with FullyConnected network"
echo "Command:"
echo "python Optimization/optimizers/random_opt.py \\"
echo "    --model 10 \\"
echo "    --model_name 'GPT_175B_RS_FC' \\"
echo "    --num_npus 64 \\"
echo "    --iterations 20 \\"
echo "    --network FullyConnected"
echo ""

# Example 5: Comparing networks using Bayesian Optimization
echo "Example 5: Comparing different networks (sequential)"
echo "Commands:"
for network in FoldedClos Switch Ring FullyConnected; do
    echo "python Optimization/optimizers/bayesian_opt.py \\"
    echo "    --model 10 \\"
    echo "    --model_name 'GPT_40B_${network}' \\"
    echo "    --num_npus 64 \\"
    echo "    --budget 20 \\"
    echo "    --network ${network}"
    echo ""
done

echo "=================================================="
echo "Available Networks:"
echo "  - FoldedClos (default)"
echo "  - Switch"
echo "  - Ring"
echo "  - FullyConnected"
echo "  - 2D_Torus"
echo "  - 3D_Torus"
echo "  - Dragonfly"
echo "  - DGX1"
echo "=================================================="
