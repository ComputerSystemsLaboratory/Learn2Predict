#!/bin/bash  

# Function to display usage/help  
usage() {  
  echo "Usage: $0 --benchmark <BENCHMARKS> -p <MAX_PASS> -r <MAX_PARAM> -d <DIRECTORY> [-o <OUTPUT_DIR>]"  
  echo  
  echo "Arguments:" 
  echo "  -b, --benchmark      Specify the benchmarks to run as a comma-separated list (e.g., BS,GEMV,HST-L)."   
  echo "  -p, --max-pass       Maximum value for 'pass' parameter (required)"  
  echo "  -r, --max-param      Maximum value for 'param' parameter (required)"  
  echo "  -d, --directory      Directory to search for existing files (required)"  
  echo "  -o, --output-dir     Directory to save the missing files (optional, default: log)"  
  echo "  -h, --help           Show this help message and exit"  
  echo  
  exit 1  
}  

# Parse arguments using getopt  
ARGS=$(getopt -o b:p:r:d:o:h --long benchmark:,max-pass:,max-param:,directory:,output-dir:,help -n "$0" -- "$@")  
if [ $? -ne 0 ]; then  
  usage  
fi  

eval set -- "$ARGS"  

# Initialize variables
BENCHMARK=""  
MAX_PASS=""  
MAX_PARAM=""  
OUTPUT_DIR="log"  # Default output directory  
SEARCH_DIR=""  
TOTAL_MISSING=0  # Track the total count of missing files  

# Extract the arguments  
while true; do  
  case "$1" in  
    -b|--benchmark)  
      BENCHMARK="$2"  
      shift 2  
      ;;   
    -p|--max-pass)  
      MAX_PASS="$2"  
      shift 2  
      ;;  
    -r|--max-param)  
      MAX_PARAM="$2"  
      shift 2  
      ;;  
    -d|--directory)  
      SEARCH_DIR="$2"  
      shift 2  
      ;;  
    -o|--output-dir)  
      OUTPUT_DIR="$2"  
      shift 2  
      ;;        
    -h|--help)  
      usage  
      ;;  
    --)  
      shift  
      break  
      ;;  
    *)  
      echo "Unexpected option: $1"  
      usage  
      ;;  
  esac  
done  

# Ensure that all required arguments are provided and valid  
if [ -z "$BENCHMARK" ] || [ -z "$MAX_PASS" ] || [ -z "$MAX_PARAM" ] || [ -z "$SEARCH_DIR" ]; then  
  echo "Error: Missing required arguments."  
  usage  
fi  

# Ensure MAX_PASS and MAX_PARAM are positive integers  
if ! [[ $MAX_PASS =~ ^[0-9]+$ ]] || ! [[ $MAX_PARAM =~ ^[0-9]+$ ]]; then  
  echo "Error: MAX_PASS and MAX_PARAM must be positive integers."  
  usage  
fi  

# Ensure the search directory exists  
if [ ! -d "$SEARCH_DIR" ]; then  
  echo "Error: Directory $SEARCH_DIR does not exist."  
  exit 1  
fi  

# Handle output directory creation  
if [ ! -d "$OUTPUT_DIR" ]; then  
  LAST_LEVEL=$(basename "$OUTPUT_DIR")   # Get the last level of the directory structure  
  PARENT_DIR=$(dirname "$OUTPUT_DIR")   # Get the parent directory  

  # If the parent directory exists, attempt to create the last level  
  if [ -d "$PARENT_DIR" ]; then  
    mkdir "$PARENT_DIR/$LAST_LEVEL" 2>/dev/null
    if [ $? -ne 0 ]; then  
      echo "Error: Failed to create output directory $OUTPUT_DIR."  
      exit 1  
    fi  
  else  
    echo "Error: Parent directory $PARENT_DIR does not exist."  
    exit 1  
  fi  
fi  

# Define the T and D arrays  
T=(1 2 4 8 16)  
D=(1 2 4 8 16)  

# Iterate over all benchmarks  
for bench in "${BENCHMARK[@]}"; do  
  # Create/reset the output file for the current benchmark  
  OUTPUT_FILE="$OUTPUT_DIR/$bench.missing"  
  > "$OUTPUT_FILE"  # Clear the output file if it already exists  

  # Iterate over all possible combinations  
  for t in "${T[@]}"; do  
    for d in "${D[@]}"; do  
      # Skip invalid combinations where T <= D  
      if [ "$t" -le "$d" ]; then  
        continue  
      fi  
      for p in $(seq 0 "$MAX_PASS"); do  
        for r in $(seq 0 "$MAX_PARAM"); do  
          # Generate the expected filename  
          file="UPMEM_${bench}_tasklets_${t}_dpus_${d}_pass_${p}_param_${r}.log"  
          # Check if the file exists in the search directory  
          if [ ! -f "$SEARCH_DIR/$file" ]; then  
            # If the file is missing, add it to the current benchmark's output file  
            echo "$file" >> "$OUTPUT_FILE"  
            TOTAL_MISSING=$((TOTAL_MISSING + 1))  # Increment the missing file counter  
          fi  
        done  
      done  
    done  
  done  

  # Print per-benchmark completion message  
  echo "Missing files for benchmark $bench saved to $OUTPUT_FILE."  
done  

# Print overall results  
echo "Missing file lists saved in directory $OUTPUT_DIR."  
echo "Total number of missing files: $TOTAL_MISSING."
