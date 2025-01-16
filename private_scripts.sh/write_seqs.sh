#!/bin/bash

#SBATCH --job-name=write_seqs
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

set -u

if [[ -z "$1" ]]; then
    echo Error: 1 positional arguments required.
    echo Usage: bash write_seqs.sh [data_preprocessing_config]
    exit 1
fi

if [[ -z "${WRITE_SEQS_ENV// /}" ]]; then
    WRITE_SEQS_ENV="conda activate write_chord_tones_seqs"
fi

if [[ -z "${WRITE_SEQS_FOLDER// /}" ]]; then
    WRITE_SEQS_FOLDER="/home/ms3682/code/write_seqs"
fi

if [[ -z "${MUSICBERT_HF_ENV// /}" ]]; then
    MUSICBERT_HF_ENV="conda activate musicbert_hf"
fi

# if [[ -z "${MUSICBERT_FOLDER// /}" ]]; then
#     MUSICBERT_FOLDER="/home/ms3682/code/musicbert_fork"
# fi

if [[ -z "${WRITE_SEQS_USE_VENV}" ]]; then
    module load miniconda
fi

# if [[ "$(uname)" == "Darwin" ]]; then
#     # readlink is not available, but shouldn't be necessary either
#     DATA_SETTINGS="${1}"
#     INPUT_DIR="${2}"
#     TEMP_DIR=$(mktemp -d)
#     OUTPUT_DIR="${3}"
# else
#     # Assume Linux
#     DATA_SETTINGS=$(readlink -f "${1}")
#     INPUT_DIR=$(readlink -f "${2}")
#     TEMP_DIR=$(mktemp -d)
#     TEMP_DIR=$(readlink -f "${TEMP_DIR}")
#     OUTPUT_DIR=$(readlink -f "${3}")
# fi

INPUT_DIR=~/project/raw_data/salami_slice_dedoubled_no_suspensions_q16
WRITE_SEQS_SETTINGS=~/code/musicbert_hf/rn_write_seqs_settings.yaml
RAW_DIR=~/project/musicbert_hf/data/raw/
OUTPUT_DIR=~/project/musicbert_hf/data/rns_v1/

echo eval "${WRITE_SEQS_ENV}"
eval "${WRITE_SEQS_ENV}"

if [[ "${INPUT_DIR}" =~ .*\.zip$ ]]; then
    INPUT_DIR_TMP="${INPUT_DIR%.zip}"
    unzip -o "${INPUT_DIR}" -d "${INPUT_DIR_TMP}"
    INPUT_DIR="${INPUT_DIR_TMP}"
fi

# I'm not sure if the module is installed in the env so we cd into the directory to
#   be sure it will run
set -e
set -x
cd "${INPUT_DIR}"
python -m write_seqs \
    --src-data-dir "${INPUT_DIR}" \
    --data-settings "${WRITE_SEQS_SETTINGS}" --output-dir "${OUTPUT_DIR}"

echo eval "${MUSICBERT_HF_ENV}"
eval "${MUSICBERT_HF_ENV}"

# This is sort of stupid because we are setting the same directories both in this script
# as RAW_DIR and OUTPUT_DIR and in the config file as input_base_folder
# and output_base_folder.

python scripts/data_preprocessing.py \
    --config "${1}"
