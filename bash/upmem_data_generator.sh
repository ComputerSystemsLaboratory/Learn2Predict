#!/bin/bash  

# Function to extract/calculate dpu kernel time
calculate_dpu_kernel_time() {  
    local UPMEM_log=$1  # First parameter is the log file  
    local dpu_kernel_time=0  
    local reduction_time=0  
    local scan_time=0  
    local add_time=0
    local step2_time=0
    local step3_time=0

    while IFS= read -r line; do  
        case "$line" in  
            *"DPU Kernel Reduction Time:"*)  
                reduction_time=$(echo "$line" | grep -oP '[\d.]+')  
                ;;  
            *"DPU Kernel Scan Time:"*)  
                scan_time=$(echo "$line" | grep -oP '[\d.]+')  
                ;;  
            *"DPU Kernel Add Time:"*)  
                add_time=$(echo "$line" | grep -oP '[\d.]+')  
                ;;  
            *"Step Two DPU Kernel Time:"*)  
                step2_time=$(echo "$line" | grep -oP '[\d.]+')  
                ;;  
            *"Step Three DPU Kernel Time:"*)  
                step3_time=$(echo "$line" | grep -oP '[\d.]+')  
                ;;  
            *"DPU Kernel Time:"*)  
                dpu_kernel_time=$(echo "$line" | grep -oP '[\d.]+')  
                ;;           
        esac  
    done < "$UPMEM_log"  

    # If DPU Kernel Time is not directly present, calculate it from components  
    if (( $(echo "$dpu_kernel_time == 0" | bc -l) )); then  
        dpu_kernel_time=$(echo "$reduction_time + $scan_time + $add_time + $step2_time + $step3_time" | bc -l)  
    fi  

    echo "$dpu_kernel_time"  # Return the dpu_kernel_time  
}

# Function to display usage  
usage() {  
    echo "Usage: $0 --package PACKAGE --benchmark BENCHMARKS --num_tasklets TASKLETS --num_dpus DPUS --passes FILENAME --log_dir DIR --upmem_dir DIR --timeout TIME"  
    echo ""  
    echo "Commands:"  
    echo "  -p, --package          Specify the package of benchmarks (prim-benchmarks, prim-cinnammon, pim-ml)."  
    echo "  -b, --benchmark        Specify the benchmarks to run as a comma-separated list (e.g., BS,GEMV,HST-L)."      
    echo "  -t, --num_tasklets     Comma-separated list of tasklets (e.g., 1,2,4,8)"  
    echo "  -d, --num_dpus         Comma-separated list of DPUs (e.g., 1,2,4,8)" 
    echo "  -a, --passes           Filename containing optimization sequences, one per line." 
    echo "  -l, --log_dir          Log files directory (use absolute path)"  
    echo "  -u, --upmem_dir        UPMEM SDK directory (use absolute path)"
    echo "  -i, --timeout          Time limit on how long a command is allowed to run (seconds)"    
    exit 1
}  

# Initialize variables 
log_directory=""  
upmem_directory=""
package=""
benchmark=""  
num_tasklets=""  
num_dpus=""  
passes_file="" 
timeout_value=0

# Parse command-line arguments using getopt  
OPTIONS=$(getopt -o p:b:t:d:a:l:u:i: --long package:,benchmark:,num_tasklets:,num_dpus:,passes:,log_dir:,upmem_dir:,timeout: -- "$@")  
if [ $? -ne 0 ]; then  
    usage  
fi  

eval set -- "$OPTIONS"  

# Extract options  
while true; do  
    case "$1" in  
        -p|--package)  
            package="$2"  
            shift 2  
            ;;  
        -b|--benchmark)  
            benchmark="$2"  
            shift 2  
            ;;              
        -t|--num_tasklets)  
            num_tasklets="$2"  
            shift 2  
            ;;  
        -d|--num_dpus)  
            num_dpus="$2"  
            shift 2  
            ;; 
        -a|--passes)  
            passes_file="$2"  
            shift 2  
            ;;   
        -l|--log_dir)  
            log_directory="$2"  
            shift 2  
            ;;  
        -u|--upmem_dir)  
            upmem_directory="$2"  
            shift 2  
            ;;  
        -i|--timeout)  
            timeout_value="$2"  
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
if [[ -z "$package" || -z "$benchmark" || -z "$num_tasklets" || -z "$num_dpus" || -z "$passes_file" || -z "$log_directory" || -z "$upmem_directory" ]]; then  
    usage  
fi  

# Check if the passes file exists and is not empty  
if [[ ! -f "$passes_file" || ! -s "$passes_file" ]]; then  
    echo "Error: Passes file $passes_file does not exist or is empty."   
    exit 1  
fi   

# Check if the upmem directory exists
if [[ ! -d "$upmem_directory" ]]; then  
    echo "Error: UPMEM directory $upmem_directory does not exist."   
    exit 1  
fi  

if [[ ! -f "$upmem_directory/upmem_env.sh" ]]; then  
    echo "Error: UPMEM env $upmem_directory/upmem_env.sh does not exist."   
    exit 1  
fi

# Validate package
valid_packages=("prim-benchmarks" "prim-cinnammon" "pim-ml")  
if [[ ! " ${valid_packages[@]} " =~ " ${package} " ]]; then  
    echo "Error: Invalid benchmark $package. Allowed values are: ${valid_packages[*]}"  
    exit 1 
fi  

# Validate timeout
if ! [[ "$timeout_value" =~ ^[0-9]+$ ]] || [[ "$timeout_value" -le 0 ]]; then  
    echo "Error: Iinvalid timeout: $timeout_value"  
    exit 1
fi  

# Validate benchmark  
IFS=',' read -r -a benchmark_array <<< "$benchmark"  
valid_benchmarks=("BFS" "BS" "GEMV" "HST-L" "HST-S" "LiRnQ" "LiRQ" "LoRnQ" "LoRQ" "MLP" "NW" "RED" "SCAN-RSS" "SCAN-SSA" "SEL" "SpMV" "TRNS" "TS" "UNI" "VA")  

for bm in "${benchmark_array[@]}"; do  
    if [[ ! " ${valid_benchmarks[@]} " =~ " ${bm} " ]]; then  
        echo "Error: Invalid benchmark $bm. Allowed values are: ${valid_benchmarks[*]}"  
        exit 1 
    fi  
done  

# Create log directory if it doesn't exist  
mkdir -p "$log_directory"  

# Initiate UPMEM environment
source "$upmem_directory/upmem_env.sh"

# Get the absolute path of the script directory  
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  
parent_dir="$(dirname "$script_dir")"  
bench_path="$parent_dir/benchmarks/$package"
current_dir="$PWD"

# Convert comma-separated lists into arrays  
IFS=',' read -r -a tasklet_array <<< "$num_tasklets"  
IFS=',' read -r -a dpu_array <<< "$num_dpus"

# Read passes from the file into an array, skipping comments and blank lines  
mapfile -t passes_array < <(grep -v '^\s*#' "$passes_file" | grep -v '^\s*$')  

# Check if the array is empty  
if [ ${#passes_array[@]} -eq 0 ]; then  
    echo "There are no compiler optimization sequences. Exiting."  
    exit 1  
fi   

# Log the start of the process  
for bm in "${benchmark_array[@]}"; do

    # Remove previous log
    rm -f $bm.out

    # Start
    echo "" | tee -a "$log_directory/$bm.out" &> /dev/null
    echo "================================================================================" | tee -a "$log_directory/$bm.out" &> /dev/null
    echo "Starting benchmark process for $bm at $(date)" | tee -a "$log_directory/$bm.out" &> /dev/null   
    echo "================================================================================" | tee -a "$log_directory/$bm.out" &> /dev/null

    # Check if the parameters file exists and is not empty  
    parameters_file="$bench_path/$bm/param.txt"
    if [[ ! -f "$parameters_file" || ! -s "$parameters_file" ]]; then 
        echo "Error: Parameters file $parameters_file does not exist or is empty."   
        continue  
    fi     

    # Read the parameters from the file into an array, skipping comments and blank lines
    mapfile -t data_param_array < <(grep -v '^\s*#' "$parameters_file" | grep -v '^\s*$')  

    # Iterate over tasklets
    for T in "${tasklet_array[@]}"; do  
        
        # Iterate over DPUS
        for D in "${dpu_array[@]}"; do  
            
            # Check if T < D  
            if [ "$T" -lt "$D" ]; then  
                # Skip this iteration  
                continue  
            fi  

            # Iterate over each pass  
            for P_index in "${!passes_array[@]}"; do  
                P="${passes_array[$P_index]}"  

                # Compile and link if log does not exist or is invalid  
                echo "Running: make NR_TASKLETS=$T NR_DPUS=$D PASSES=$P" | tee -a "$log_directory/$bm.out" &> /dev/null
                cd $bench_path/$bm
                make clean
                make NR_TASKLETS=$T NR_DPUS=$D PASSES="$P"

                # Check if compilation OK
                if [ ! -f "$bench_path/$bm/bin/host_code" ] || [ ! -f "$bench_path/$bm/bin/dpu_code" ]; then   
                    continue
                fi

                for PP_index in "${!data_param_array[@]}"; do 
                    PP="${data_param_array[$PP_index]}"

                    # Check if UPMEM_log exists and is valid before running compile and link 
                    UPMEM_log="${log_directory}/UPMEM_${bm}_tasklets_${T}_dpus_${D}_pass_${P_index}_param_${PP_index}.log"  
                    if [[ -f "$UPMEM_log" ]]; then 
                        dpu_kernel_time=$(calculate_dpu_kernel_time "$UPMEM_log")  
                        # Validate dpu_kernel_time  
                        if (( $(echo "$dpu_kernel_time > 0" | bc -l) )); then  
                            echo "DPU Kernel Time is valid: $dpu_kernel_time ms. Skipping ..." | tee -a "$log_directory/$bm.out" &> /dev/null
                            continue
                        fi
                    fi

                    # Run the benchmark  
                    echo "Running: $bench_path/$bm and logging to $UPMEM_log" | tee -a "$log_directory/$bm.out" &> /dev/null  
                    timeout $timeout_value $bench_path/$bm/bin/host_code $PP &> "$UPMEM_log"
                    if [[ $? -eq 124 ]]; then  
                        echo "host_code did not terminate within the timeout period. Removing log ..." | tee -a "$log_directory/$bm.out" &> /dev/null 
                    fi

                    if [[ -f "$UPMEM_log" ]]; then 
                        dpu_kernel_time=$(calculate_dpu_kernel_time "$UPMEM_log")  
                        # Validate dpu_kernel_time  
                        if (( $(echo "$dpu_kernel_time > 0" | bc -l) )); then  
                            echo "DPU Kernel Time is valid: $dpu_kernel_time ms." | tee -a "$log_directory/$bm.out" &> /dev/null
                            echo "UPMEM output saved to $UPMEM_log" | tee -a "$log_directory/$bm.out" &> /dev/null
                            continue
                        else
                            echo "DPU Kernel Time is invalid. Removing log ..." | tee -a "$log_directory/$bm.out" &> /dev/null
                            rm -f "$UPMEM_log"
                        fi
                    fi
                done

                make clean
                
                echo "" | tee -a "$log_directory/$bm.out" &> /dev/null

            done
  
        done  
    done  

    # Log the end of the process
    echo "================================================================================" | tee -a "$log_directory/$bm.out" &> /dev/null  
    echo "Benchmark process for $bm completed at $(date)" | tee -a "$log_directory/$bm.out" &> /dev/null  
    echo "================================================================================" | tee -a "$log_directory/$bm.out" &> /dev/null
done

cd "$current_dir"