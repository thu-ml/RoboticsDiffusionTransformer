#!/bin/bash

# Define the list of dataset names
DATASETS=(
    'fractal20220817_data'
    'kuka'
    'bridge'
    'taco_play'
    'jaco_play'
    'berkeley_cable_routing'
    'roboturk'
    'nyu_door_opening_surprising_effectiveness'
    'viola'
    'berkeley_autolab_ur5'
    'toto'
    'language_table'
    'columbia_cairlab_pusht_real'
    'stanford_kuka_multimodal_dataset_converted_externally_to_rlds'
    'nyu_rot_dataset_converted_externally_to_rlds'
    'stanford_hydra_dataset_converted_externally_to_rlds'
    'austin_buds_dataset_converted_externally_to_rlds'
    'nyu_franka_play_dataset_converted_externally_to_rlds'
    'maniskill_dataset_converted_externally_to_rlds'
    'furniture_bench_dataset_converted_externally_to_rlds'
    'cmu_franka_exploration_dataset_converted_externally_to_rlds'
    'ucsd_kitchen_dataset_converted_externally_to_rlds'
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds'
    'austin_sailor_dataset_converted_externally_to_rlds'
    'austin_sirius_dataset_converted_externally_to_rlds'
    'bc_z'
    'usc_cloth_sim_converted_externally_to_rlds'
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds'
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds'
    'utokyo_saytap_converted_externally_to_rlds'
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds'
    'utokyo_xarm_bimanual_converted_externally_to_rlds'
    'robo_net'
    'berkeley_mvp_converted_externally_to_rlds'
    'berkeley_rpt_converted_externally_to_rlds'
    'kaist_nonprehensile_converted_externally_to_rlds'
    'stanford_mask_vit_converted_externally_to_rlds'
    'tokyo_u_lsmo_converted_externally_to_rlds'
    'dlr_sara_pour_converted_externally_to_rlds'
    'dlr_sara_grid_clamp_converted_externally_to_rlds'
    'dlr_edan_shared_control_converted_externally_to_rlds'
    'asu_table_top_converted_externally_to_rlds'
    'stanford_robocook_converted_externally_to_rlds'
    'eth_agent_affordances'
    'imperialcollege_sawyer_wrist_cam'
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds'
    'uiuc_d3field'
    'utaustin_mutex'
    'berkeley_fanuc_manipulation'
    'cmu_playing_with_food'
    'cmu_play_fusion'
    'cmu_stretch'
    'berkeley_gnm_recon'
    'berkeley_gnm_cory_hall'
    'berkeley_gnm_sac_son'
    # Additional dataset
    'droid'
    'fmb'
    'dobbe'
)


# Iterate over each dataset name
for dataset_name in "${DATASETS[@]}"; do
    echo "Downloading $dataset_name"

    # Execute gsutil command
    ~/miniconda3/envs/rdt-data/bin/gsutil -m cp -n -r -D "gs://gresearch/robotics/$dataset_name" ../datasets/openx_embod/
    
    # Check if the resulting directory exists
    directory_path="../datasets/openx_embod/$dataset_name"
    if [ ! -d "$directory_path" ]; then
        # If the directory does not exist, then print an error message
        echo "Failed to download $dataset_name"
    else
        # If the directory exists, then print a success message
        echo "Successfully downloaded $dataset_name"
    fi
done
