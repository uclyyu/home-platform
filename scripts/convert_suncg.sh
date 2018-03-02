#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Counting
num_seen=0
num_egg_errors=0
num_bam_errors=0

# Temporarily setting the internal field seperator (IFS) to the newline character.
IFS=$'\n';

# Recursively loop through all OBJ files in the specified directory
OBJ_DIRECTORY=$1
has_err=0

echo "Finding all objects to convert... (be patient)"
objects=($(find ${OBJ_DIRECTORY} -name '*.obj'))
num_objects=${#objects[*]}
echo "Found ${num_objects} objects..."
printf "Start converting... (be patient)\n\n"

for obj in ${objects[*]}; do
	((++num_seen))
	echo "Processing file ${num_seen}/${num_objects}"
	echo "Processing OBJ file ${obj}"

	INPUT_OBJ_FILE="${obj}"
	OUTPUT_EGG_FILE="${obj%.obj}.egg"
	OUTPUT_BAM_FILE="${obj%.obj}.bam"

	if [ -f $OUTPUT_BAM_FILE ]; then
		echo "Output BAM file ${OUTPUT_BAM_FILE} already found"
		echo "Skipping conversion for OBJ file ${INPUT_OBJ_FILE}"
		continue
	fi

	INPUT_OBJ_DIR=$(dirname "${INPUT_OBJ_FILE}")
	cd ${INPUT_OBJ_DIR}

	python ${DIR}/obj2egg.py --coordinate-system=y-up-right ${INPUT_OBJ_FILE}
	if ! [ -f $OUTPUT_EGG_FILE ]; then
		echo "Could not find output file ${OUTPUT_EGG_FILE}. An error probably occured during conversion."
		((has_err=1))
		((++num_egg_errors))
	else
        egg2bam -ps rel -o ${OUTPUT_BAM_FILE} ${OUTPUT_EGG_FILE}
        if ! [ -f $OUTPUT_BAM_FILE ]; then
		echo "Could not find output file ${OUTPUT_BAM_FILE}. An error probably occured during conversion."
        ((has_err=1))	
		((++num_bam_errors))
        fi
    fi
done

printf 'All done.\n\n'

if [[ has_err -eq 1 ]]; then
    echo "Failure! There were some errors. This happens sometimes. Please run this again."
    echo "Number of egg errors: ${num_egg_errors}"
    echo "Number of bam errors: ${num_bam_errors}"
else
    echo "Success! No errors occured."
fi
