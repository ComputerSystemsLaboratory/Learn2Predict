#!/bin/bash  

# Function to display usage  
usage() {  
    echo "Usage: $0 --benchmark BENCHMARKS --log_dir DIR"  
    echo ""  
    echo "Commands:"  
    echo "  -b, --benchmark        Specify the benchmark to train (e.g., HST-L, HST-S, VA)."  
    echo "  -l, --log_dir          Log files directory (use absolute path)" 
    echo "  -t, --target           Target (e.g., speedup, runtime, runtime+speedup)"   
    exit 1
}  

# Initialize variables 
log_directory=""  
benchmark=""  
target=""

# Parse command-line arguments using getopt  
OPTIONS=$(getopt -o b:l:t: --long benchmark:,log_dir:target: -- "$@")  
if [ $? -ne 0 ]; then  
    usage  
fi  

eval set -- "$OPTIONS"  

# Extract options  
while true; do  
    case "$1" in  
        -b|--benchmark)  
            benchmark="$2"  
            shift 2  
            ;;  
        -l|--log_dir)  
            log_directory="$2"  
            shift 2  
            ;;   
        -t|--target)  
            target="$2"  
            shift 2  
            ;;                                      
        --)  
            shift  
            break  
            ;;  
        *)  
            echo "Invalid option: $1"  
            usage  
            ;;  
    esac  
done  

# Check if all required arguments are provided  
if [[ -z "$benchmark" || -z "$log_directory" || -z "$target" ]]; then  
    usage  
fi  

# Validate benchmark  
valid_benchmarks=("HST-L" "HST-S" "VA")  
if [[ ! " ${valid_benchmarks[@]} " =~ " ${benchmark} " ]]; then  
    echo "Error: Invalid benchmark $benchmark. Allowed values are: ${valid_benchmarks[*]}"  
    exit 1 
fi  
 
# Check the value of target  
case "$target" in  
    speedup)  
        exclude_columns="runtime"
        model_name="speedup_model.pth" 
        ;;  
    runtime)  
        exclude_columns="speedup"  
        model_name="runtime_model.pth" 
        ;;  
    runtime,speedup)  
        exclude_columns=""  
        model_name="runtime+speedup_model.pth" 
        ;;  
    *)  
        echo "Invalid value for target. Allowed values: 'speedup', 'runtime', 'runtime,speedup'."  
        exit 1  
        ;;  
esac 

# Create log directory if it doesn't exist  
mkdir -p "$log_directory"  

# Get the absolute path of the script directory  
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  
parent_dir="$(dirname "$script_dir")"  
py_path="$parent_dir/py"
exe_path="$parent_dir/example"
current_dir="$PWD"

cd $py_path

python training.py \
    --batch_size 16 \
    --dataset_dir "$exe_path/datasets" \
    --dropout_rate 0.5 \
    --elapsed_filename "elapsed_time.yml" \
    --exclude_columns "$exclude_columns" \
    --hidden_size 256 \
    --history_filename "history_train.yml" \
    --learning_rate 0.0001 \
    --model_dir "$log_directory" \
    --model_name "best_model.pth" \
    --num_epochs 1000 \
    --num_layers 12 \
    --output_dir "$log_directory" \
    --patience 200 \
    --random_state 42 \
    --target_column "$target" \
    --test_size 0.0 \
    --training_benchmarks "$benchmark" \
    --transformation "log" \
    --use_patience \
    --val_size 0.2

python visualization.py \
    --history "$log_directory/history_train.yml" \
    --output_dir "$log_directory" \
    --plot_history


cd "$current_dir"