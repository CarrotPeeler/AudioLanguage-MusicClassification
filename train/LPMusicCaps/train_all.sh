nohup python3 train/LPMusicCaps/transfer.py --yaml_config_path=train/LPMusicCaps/exp/transfer/music_instruct/mi_short_long_hparams.yaml < /dev/null > train_short_long.txt 2>&1 &
wait; nohup python3 train/LPMusicCaps/transfer.py --yaml_config_path=train/LPMusicCaps/exp/transfer/music_instruct/mi_long_hparams.yaml < /dev/null > train_long.txt 2>&1 &
wait; nohup python3 train/LPMusicCaps/transfer.py --yaml_config_path=train/LPMusicCaps/exp/transfer/music_instruct/mi_all_hparams.yaml < /dev/null > train_all.txt 2>&1 &

