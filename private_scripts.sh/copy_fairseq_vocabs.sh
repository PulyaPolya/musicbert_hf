#!/bin/bash

#!/bin/bash

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_dir="$1"
output_dir="$2"

# Check if input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Error: Input directory does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through immediate subdirectories
for subdir in "$input_dir"/*/; do
    if [ -d "$subdir" ]; then
        # Get the base name of the subdirectory
        subdir_name=$(basename "$subdir")

        # Check if dict.txt exists in this subdirectory
        if [ -f "${subdir}dict.txt" ]; then
            # Copy the file to output directory with new name
            cp "${subdir}dict.txt" "${output_dir}/${subdir_name}.txt"
            echo "Copied ${subdir}dict.txt to ${output_dir}/${subdir_name}.txt"
        fi
    fi
done
