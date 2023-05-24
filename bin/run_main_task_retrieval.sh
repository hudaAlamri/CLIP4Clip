python -m torch.distributed.launch --nproc_per_node=1 \
main_task_retrieval.py \
--do_train \
--do_eval \
--num_thread_reader=16 \
--epochs=50 \
--batch_size=32 \
--batch_size_val=32 \
--n_display=100 \
--train_csv=/home/huda/CLIP4Clip/data/msrvtt/MSRVTT_train.9k.csv \
--val_csv=/home/huda/CLIP4Clip/data/msrvtt/MSRVTT_JSFUSION_test.csv \
--data_path=/data/ECLIPSE/charades/avsd \
--features_path=/data/ECLIPSE/charades/Charades_v1_480 \
--output_dir=ckpts/ckpt_avsd_retrieval_looseType \
--lr=1e-4 \
--max_words=30 \
--max_frames=12 \
--datatype="AVSD"  \
--feature_framerate=1 \
--coef_lr=1e-3 \
--freeze_layer_num=0 \
--slice_framepos=2 \
--loose_type \
--linear_patch=2d \
--sim_header=meanP \
--pretrained_clip_name=ViT-B/16 \

