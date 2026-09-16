# Source this file to isolate PBPF from the host ROS Python packages.
source /data/cwj/PBPF/.venv/bin/activate
unset PYTHONPATH PYTHONHOME
export HF_HOME=/data/cwj/PBPF/local/model-cache
export TOKENIZERS_PARALLELISM=false
