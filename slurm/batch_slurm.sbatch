#!/bin/bash
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bareeva@hhi.fraunhofer.de
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

source "/etc/slurm/local_job_dir.sh"

# The next line is optional and for job statistics only. You may omit it if you do not need statistics.
echo "$PWD/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg

# Make a folder locally on the node for job_results. This folder ensures that data is copied back even when the job fails
mkdir -p "${LOCAL_JOB_DIR}/job_results"

echo "Extract miniimageNet.tar to ${LOCAL_JOB_DIR}/miniImageNet"
time tar -C $LOCAL_JOB_DIR -xf $DATAPOOL1/datasets/miniImagenet.tar

echo "Run resnet training"
apptainer run --nv --bind $LOCAL_JOB_DIR:/mnt/dataset --bind ${LOCAL_JOB_DIR}/job_results:/mnt/output ./slingshot.sif  alpha="$1" model_dir="/mnt/output/" output_dir="/mnt/output/" data_dir="/mnt/dataset/" disable_tqdm="True" evaluate="False" device="cuda:0" target_img_path="./assets/image_dep/inet_train_n03496892_19229.JPEG" img_str="tractor"

# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding the job id.
cd "$LOCAL_JOB_DIR"
tar -cf experiment_${SLURM_JOB_ID}.tar job_results
cp experiment_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR/slingshot_output
rm -rf ${LOCAL_JOB_DIR}/job_results