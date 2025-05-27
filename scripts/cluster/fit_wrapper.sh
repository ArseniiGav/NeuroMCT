#!/bin/bash
# allow globbing in scripts
shopt -s extglob
# bash unofficial strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
# set -euo pipefail
# print every command to stderr prefixed with '+' character
# really useful for debugging to understand what happens where
set -x

file_n=$1
fit_tool=$2
dataset=$3
ml_model=$4

wrapped_script_short=fit.py
wrapped_script=$(realpath $wrapped_script_short)

# get writable directory inside container
BASE=$_CONDOR_SCRATCH_DIR
# seed and power_unc are provided to script by HTCondor based on
# "arguments" directive

cd $BASE

# output folder
export final_output="$BASE/testoutput"
output_loc=$STORAGE_URL/pnfs/jinr.ru/data/juno/users/d/dolzhikov/2025-04-21-neuromct-full-fits-final/$ml_model/$dataset

echo "Creating output folder $final_output"
mkdir -p "$final_output"

# Set a pathes to inspect when substituting data (see "Caveats with loading data into GNA on computing nodes")
# export GNA_DATAPATH=/cvmfs/dayabay.jinr.ru/gna/data/data_juno/master
# Daya Bay thermal powers are not public information so we need to keep them in EOS and copy into each task individually

# Going to GNA home and run the analysis.
# With GNA_DATAPATH set GNA will correctly
# handle pathes like `data/dayabay/some/input.yaml`
# Remove output redirection to /dev/null when debugging!
cd /soft/neuromct
python3.11 -c "import sys; print(sys.executable)"
# source "/cvmfs/sft.cern.ch/lcg/releases/LCG_106/xrootd/5.6.9/x86_64-el9-gcc13-opt/xrootd-env.sh"

python3.11 $wrapped_script --sources Cs137 K40 Co60 AmBe AmC --model $ml_model --fit-tool $fit_tool --dataset $dataset --file-number $file_n -o $final_output -mpath ./ -s $file_n 
# Get output name
cd $final_output
output_files=$(ls)

# Copying outputs to EOS.
# IMPORTANT: mind the trailing '/', it makes xrdcp to copy directories as directories!
xrdcp -rpf $output_files $output_loc/
