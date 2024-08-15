
# +++++++++++++++++++++++++++++++++   TSET ResNet   ++++++++++++++++++++++++++++++++#



# TEST ResNet18 dim 11
python Test_Combine_Policynet.py --arch=cifar10_resnet_18_add_key \
                                 -d cifar10 \
                                 --seed=123 \
                                 --mask_size=3 \
                                 --add_layer=3 \
                                 --config_path='/home/lpz/xf/skipnet-master/skipnet-master/cifar/configs/resnet18/config_dim_11.json' \
                                 --save_folder='/home/lpz/xf/skipnet-master/skipnet-master/cifar/save_checkpoints_add_key/' \
                                 --resume='/home/lpz/xf/skipnet-master/skipnet-master/cifar/save_checkpoints_add_key/cifar10_resnet_18_add_key/cifar10_resnet_18_add_key_add_layer_3.pth.tar' \
                                 --resume_policynet='/home/lpz/xf/skipnet-master/skipnet-master/cifar/save_checkpoints_policynet/resnet18/output_dim_11/policy_net_epoch_2_dim_11_' \
                                 --use_mask=0