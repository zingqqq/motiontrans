# bash scripts_data/zarr_robot_data_conversion_batch.sh

# input_dir="/data/zeqingwang/vis_test/raw_data_robot" #"/data/zeqingwang/motiontrans_dataset/motiontrans_dataset/raw_data_robot"
input_dir="/home/ubuntu/Desktop/video_benchmarking_project/data/LBM_sim_egocentric/raw/raw_data_robot_no_corrupted_episodes"
num_use_source=-1               # how many episodes to use for this task, -1 means all
# output_dir="/data/zeqingwang/vis_test/zarr_data/zarr_data_robot"        #"/data/zeqingwang/motiontrans_dataset/zarr_data/zarr_data_robot"
# output_dir="/home/ubuntu/Desktop/video_benchmarking_project/data/zarr_data/zarr_data_robot"
output_dir="/home/ubuntu/Desktop/video_benchmarking_project/data/zarr_data/zarr_data_robot_no_corrupted_episodes_no_idle"
hand_to_eef_file="assets/franka_eef_to_wrist_robot_base.npy"
mode="o"                        # o: origin, s: stereo, p: pointclouds, a: all
n_encoding_threads=16           # how many processes to use for data processing
num_points_final=1024           # if save pointclouds, how many points to sample
points_max_distance_final=1.0   # if save pointclouds, the max distance to keep points

export CUDA_VISIBLE_DEVICES=0

python -m scripts_data.entry.zarr_robot_data_conversion_batch_zeqing_wrist \
    --input_dir ${input_dir} \
    --output ${output_dir} \
    --hand_to_eef_file ${hand_to_eef_file} \
    --mode ${mode} \
    --resolution_resize 640x480 \
    --resolution_crop 640x480 \
    --resolution_image_final 224x224 \
    --num_use_source ${num_use_source} \
    --num_points_final ${num_points_final} \
    --points_max_distance_final ${points_max_distance_final} \
    --n_encoding_threads ${n_encoding_threads} \
    --wrist_cameras wrist_right_plus,wrist_left_plus \