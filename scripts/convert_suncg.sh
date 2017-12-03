#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Temporarily setting the internal field seperator (IFS) to the newline character.
IFS=$'\n';

# Recursively loop through all OBJ files in the specified directory
OBJ_DIRECTORY=$1
has_err=0
echo "Finding all objects to convert... (may take a moment)"
for obj in $(find ${OBJ_DIRECTORY} -name '*.obj'); do
	echo "Processing OBJ file ${obj}"

	INPUT_OBJ_FILE="${obj}"
	OUTPUT_EGG_FILE="${obj%.obj}.egg"
	OUTPUT_BAM_FILE="${obj%.obj}.bam"

	if [ -f $OUTPUT_EGG_FILE ]; then
		echo "Output EGG file ${OUTPUT_EGG_FILE} already found"
		echo "Skipping conversion for OBJ file ${INPUT_OBJ_FILE}"
		continue
	fi

	INPUT_OBJ_DIR=$(dirname "${INPUT_OBJ_FILE}")
	cd ${INPUT_OBJ_DIR}

	python ${DIR}/obj2egg.py --coordinate-system=y-up-right ${INPUT_OBJ_FILE}
	if ! [ -f $OUTPUT_EGG_FILE ]; then
		echo "Could not find output file ${OUTPUT_EGG_FILE}. An error probably occured during conversion."
		has_err=1
	else
        egg2bam -ps rel -o ${OUTPUT_BAM_FILE} ${OUTPUT_EGG_FILE}
        if ! [ -f $OUTPUT_BAM_FILE ]; then
            echo "Could not find output file ${OUTPUT_BAM_FILE}. An error probably occured during conversion."
            has_err=1
        fi
    fi
done

echo 'All done.'

if [[ has_err -eq 1 ]]; then
    echo 'IMPORTANT! There were some errors. This happens sometimes. Please run this again.'
    #TODO: actually count errors and display here the number of errors
fi