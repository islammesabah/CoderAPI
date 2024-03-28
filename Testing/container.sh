srun \
  -n 4 \
  --mem=100000  \
  -p V100-32GB  \
  --gpus=4   \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"   \
  --container-image=/netscratch/$USER/training.sqsh  \
  --container-workdir="`pwd`"   \
  --time=3-00:00 \
  --pty /bin/bash
