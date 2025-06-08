set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH=""
# Possible values: aime, amc, math, minerva, olympiad_bench
DATATYPES=("proofwriter")
OUTPUT_DIR="$HOME"  # Add default output directory

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=1 \
        data.path=dataset/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.parquet \
        data.n_samples=32 \
        data.batch_size=32 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.8 \
        rollout.response_length=2048 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=1
done
