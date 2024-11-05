python -m scripts.agilex_inference \
    --use_actions_interpolation \
    --pretrained_model_name_or_path="checkpoints/your_finetuned_ckpt.pt" \  # your finetuned checkpoint: e.g., checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>, checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt,
    --lang_embeddings_path="outs/lang_embeddings/your_instr.pt" \
    --ctrl_freq=25    # your control frequency

# inference at simplerenv
python -m scripts.simplerenv_inference \
    --config_path configs/base.yaml \
    --pretrained_model_name_or_path robotics-diffusion-transformer/rdt-1b \
    --lang_embeddings_path outs/widowx_pick_up_the_spoon.pt \
    --ctrl_freq 5 \
    --chunk_size 64 \
    --env_name widowx_spoon_on_towel