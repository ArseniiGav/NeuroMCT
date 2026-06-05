#!/bin/bash
set -e

# Base directories
PACKAGE_DIR="/home/ubuntu/arsenii/JUNO/orsa_neuromct"
PROCESSED_DATA_BASE="/storage/jmct_paper/processed_data/sparsified/tede"
RESULTS_BASE="/storage/jmct_paper/results/tdata_size_check/tede"

# Grid sizes and event statistics to train
GRID_SIZES=(21 11 6)
EVENTS=(10000 5000 2000 1000 500)

cd "$PACKAGE_DIR"

for grid in "${GRID_SIZES[@]}"; do
    for events in "${EVENTS[@]}"; do
        
        # Skip the 6 grid 500 events case since it was already done as a pilot
        if [ "$grid" -eq 6 ] && [ "$events" -eq 500 ]; then
            echo "Skipping TEDE 6^3 grid with 500 events (pilot already trained)"
            continue
        fi

        # Skip the 21 grid 10000 events case since it is the baseline we already have
        if [ "$grid" -eq 21 ] && [ "$events" -eq 10000 ]; then
            echo "Skipping TEDE 21^3 grid with 10000 events (baseline already trained)"
            continue
        fi

        echo "--------------------------------------------------------"
        echo "Starting TEDE training for ${grid}^3 grid with $events events"
        echo "--------------------------------------------------------"
        
        # Execute the training script
        python3 scripts/run_training.py \
            --approach_type tede \
            --processed_data_dir "${PROCESSED_DATA_BASE}/${grid}grid_${events}events" \
            --results_dir "${RESULTS_BASE}/${grid}grid_${events}events" \
            --plot_every 20 \
            
    done
done

echo ""
echo "============================================================"
echo "All TEDE training configurations completed successfully!"
echo "============================================================"
