#!/bin/bash
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bareeva@hhi.fraunhofer.de
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

source "/etc/slurm/local_job_dir.sh"

# The next line is optional and for job statistics only. You may omit it if you do not need statistics.
echo "$PWD/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg

# Make a folder locally on the node for job_results. This folder ensures that data is copied back even when the job fails
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/tiny-imagenet-200"


export HYDRA_FULL_ERROR=1

#time tar -C $LOCAL_JOB_DIR -xf $DATAPOOL1/datasets/miniImagenet.tar
cp $DATAPOOL3/datasets/tiny-imagenet-200.zip $LOCAL_JOB_DIR/
time unzip -q $LOCAL_JOB_DIR/tiny-imagenet-200.zip -d $LOCAL_JOB_DIR


echo "Run resnet training"
# MANY IMAGES EXPERIMENT
apptainer run --nv --bind ${SLURM_SUBMIT_DIR}/model_weights:/mnt/models --bind $LOCAL_JOB_DIR:/mnt/dataset --bind ${LOCAL_JOB_DIR}/job_results:/mnt/output ./slingshot.sif  alpha=0.9 model_dir="/mnt/models/" output_dir="/mnt/output/" data_dir="/mnt/dataset/" disable_tqdm="True" evaluate="False" device="cuda:0" target_img_path="$2" train_original="False" epochs=50 replace_relu="False" data="tinyimagenet" model="resnet_18_for_tiny" img_str=null fv_sd=1e-2

#KERNEL EXPERIMENT
#apptainer run --nv --bind ${SLURM_SUBMIT_DIR}/model_weights:/mnt/models --bind $LOCAL_JOB_DIR:/mnt/dataset --bind ${LOCAL_JOB_DIR}/job_results:/mnt/output ./slingshot.sif model_dir="/mnt/models/" output_dir="/mnt/output/" data_dir="/mnt/dataset/" disable_tqdm="True" evaluate="False" device="cuda:0" train_original="True" epochs=50 replace_relu="False" model=resnet_18_K_32_P_64 model.model.kernel_size="$1" model.model.inplanes="$2" model.model.num_classes=1000 model.n_out=1000 img_str=K_"$1"_P_"$2" model.original_weights_path=resnet_18_K_"$1"_P_"$2".pth target_img_path="./assets/image_dep_rs18_224/inet_val_ILSVRC2012_val_00008714.JPEG" alpha=1e-2 gamma=10.0

# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding the job id.
cd "$LOCAL_JOB_DIR"
tar -cf experiment_${SLURM_JOB_ID}.tar job_results
cp experiment_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR/slingshot_output
rm -rf ${LOCAL_JOB_DIR}/job_results