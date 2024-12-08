# model_figs= {}
# for m, model in enumerate(models):
# Set paths to your config file and model weights

# fold 1
config_file_PSP1 = r"work_dirs\Fold1_PSPNet\20240410_233350\vis_data\config.py"
checkpoint_file_PSP1 = r"work_dirs\Fold1_PSPNet\best_mIoU_iter_19400.pth"

config_file_UNET1 = r"work_dirs\Fold1_UNet\20240410_093613\vis_data\config.py"
checkpoint_file_UNET1 = r"work_dirs\Fold1_UNet\best_mIoU_iter_17800.pth"

config_file_KNET1 = r"work_dirs\Fold1_KNet\20240411_053605\vis_data\config.py"
checkpoint_file_KNET1 = r"work_dirs\Fold1_KNet\best_mIoU_iter_18400.pth"

config_file_FASTSCNN1 = r"work_dirs\Fold1_FastSCNN\20240409_103621\vis_data\config.py"
checkpoint_file_FASTSCNN1 = r"work_dirs\Fold1_FastSCNN\best_mIoU_iter_18800.pth"

config_file_DLabV1 = r"work_dirs\Fold1_DeepLabV3plus\20240410_155953\vis_data\config.py"
checkpoint_file_DLabV1 = r"work_dirs\Fold1_DeepLabV3plus\best_mIoU_iter_19400.pth"

config_file_SegF1 = r"work_dirs\Fold1_Segformer\20240409_151700\vis_data\config.py"
checkpoint_file_SegF1 = r"work_dirs\Fold1_Segformer\best_mIoU_iter_14000.pth"

# fold 2
config_file_PSP2 = r"work_dirs\Fold2_PSPNet\20240412_225812\vis_data\config.py"
checkpoint_file_PSP2 = r"work_dirs\Fold2_PSPNet\best_mIoU_iter_18200.pth"

config_file_UNET2 = r"work_dirs\Fold2_UNet\20240412_085546\vis_data\config.py"
checkpoint_file_UNET2 = r"work_dirs\Fold2_UNet\best_mIoU_iter_18200.pth"

config_file_KNET2 = r"work_dirs\Fold2_KNet\20240413_050213\vis_data\config.py"
checkpoint_file_KNET2 = r"work_dirs\Fold2_KNet\best_mIoU_iter_19400.pth"

config_file_FASTSCNN2 = r"work_dirs\Fold2_FastSCNN\20240411_183859\vis_data\config.py"
checkpoint_file_FASTSCNN2 = r"work_dirs\Fold2_FastSCNN\best_mIoU_iter_18000.pth"

config_file_DLabV2 = r"work_dirs\Fold2_DeepLabV3plus\20240412_152134\vis_data\config.py"
checkpoint_file_DLabV2 = r"work_dirs\Fold2_DeepLabV3plus\best_mIoU_iter_14000.pth"

config_file_SegF2 = r"work_dirs\Fold2_Segformer\20240411_231801\vis_data\config.py"
checkpoint_file_SegF2 = r"work_dirs\Fold2_Segformer\best_mIoU_iter_20000.pth"

# fold 3
config_file_PSP3 = r"work_dirs\Fold3_PSPNet\20240417_133429\vis_data\config.py"
checkpoint_file_PSP3 = r"work_dirs\Fold3_PSPNet\best_mIoU_iter_18800.pth"

config_file_UNET3 = r"work_dirs\Fold3_UNet\20240416_234348\vis_data\config.py"
checkpoint_file_UNET3 = r"work_dirs\Fold3_UNet\best_mIoU_iter_19800.pth"

config_file_KNET3 = r"work_dirs\Fold3_KNet\20240417_193725\vis_data\config.py"
checkpoint_file_KNET3 = r"work_dirs\Fold3_KNet\best_mIoU_iter_16600.pth"

config_file_FASTSCNN3 = r"work_dirs\Fold3_FastSCNN\20240416_092823\vis_data\config.py"
checkpoint_file_FASTSCNN3 = r"work_dirs\Fold3_FastSCNN\best_mIoU_iter_19800.pth"

config_file_DLabV3 = r"work_dirs\Fold3_DeepLabV3plus\20240417_060133\vis_data\config.py"
checkpoint_file_DLabV3 = r"work_dirs\Fold3_DeepLabV3plus\best_mIoU_iter_19600.pth"

config_file_SegF3 = r"work_dirs\Fold3_Segformer\20240416_140840\vis_data\config.py"
checkpoint_file_SegF3 = r"work_dirs\Fold3_Segformer\best_mIoU_iter_18800.pth"

# fold 4
config_file_PSP4 = r"work_dirs\Fold4_PSPNet\20240420_182203\vis_data\config.py"
checkpoint_file_PSP4 = r"work_dirs\Fold4_PSPNet\best_mIoU_iter_16800.pth"

config_file_UNET4 = r"work_dirs\Fold4_UNet\20240419_003453\vis_data\config.py"
checkpoint_file_UNET4 = r"work_dirs\Fold4_UNet\best_mIoU_iter_19000.pth"

config_file_KNET4 = r"work_dirs\Fold4_KNet\20240421_002158\vis_data\config.py"
checkpoint_file_KNET4 = r"work_dirs\Fold4_KNet\best_mIoU_iter_16600.pth"

config_file_FASTSCNN4 = r"work_dirs\Fold4_FastSCNN\20240418_094227\vis_data\config.py"
checkpoint_file_FASTSCNN4 = r"work_dirs\Fold4_FastSCNN\best_mIoU_iter_19800.pth"

config_file_DLabV4 = r"work_dirs\Fold4_DeepLabV3plus\20240419_065525\vis_data\config.py"
checkpoint_file_DLabV4 = r"work_dirs\Fold4_DeepLabV3plus\best_mIoU_iter_13400.pth"

config_file_SegF4 = r"work_dirs\Fold4_Segformer\20240418_142925\vis_data\config.py"
checkpoint_file_SegF4 = r"work_dirs\Fold4_Segformer\best_mIoU_iter_17200.pth"
allBestModels = dict(
    fold1={
		"PSPNet": checkpoint_file_PSP1,
		"UNet":checkpoint_file_UNET1,
		"KNet":checkpoint_file_KNET1,
		"FastSCNN":checkpoint_file_FASTSCNN1,
        "DeepLabV3plus": checkpoint_file_DLabV1,
		"Segformer": checkpoint_file_SegF1},
    fold2={
		"PSPNet": checkpoint_file_PSP2,
		"UNet":checkpoint_file_UNET2,
		"KNet":checkpoint_file_KNET2,
		"FastSCNN":checkpoint_file_FASTSCNN2,
        "DeepLabV3plus": checkpoint_file_DLabV2,
		"Segformer": checkpoint_file_SegF2},
    fold3={
		"PSPNet": checkpoint_file_PSP3,
		"UNet":checkpoint_file_UNET3,
		"KNet":checkpoint_file_KNET3,
		"FastSCNN":checkpoint_file_FASTSCNN3,
        "DeepLabV3plus": checkpoint_file_DLabV3,
		"Segformer": checkpoint_file_SegF3},
    fold4={
		"PSPNet": checkpoint_file_PSP4,
		"UNet":checkpoint_file_UNET4,
		"KNet":checkpoint_file_KNET4,
		"FastSCNN":checkpoint_file_FASTSCNN4,
        "DeepLabV3plus": checkpoint_file_DLabV4,
		"Segformer": checkpoint_file_SegF4})

# print(allBestModels)
allConfigs = dict(
    fold1={
		"PSPNet": config_file_PSP1,
		"UNet":config_file_UNET1,
		"KNet":config_file_KNET1,
		"FastSCNN":config_file_FASTSCNN1,
		"DeepLabV3plus":config_file_DLabV1,
        "Segformer":config_file_SegF1},
    fold2={
		"PSPNet": config_file_PSP2,
		"UNet":config_file_UNET2,
		"KNet":config_file_KNET2,
		"FastSCNN":config_file_FASTSCNN2,
		"DeepLabV3plus":config_file_DLabV2,
        "Segformer":config_file_SegF2},
    fold3={
		"PSPNet": config_file_PSP3,
		"UNet":config_file_UNET3,
		"KNet":config_file_KNET3,
		"FastSCNN":config_file_FASTSCNN3,
		"DeepLabV3plus":config_file_DLabV3,
        "Segformer":config_file_SegF3},
    fold4={
		"PSPNet": config_file_PSP4,
		"UNet":config_file_UNET4,
		"KNet":config_file_KNET4,
		"FastSCNN":config_file_FASTSCNN4,
		"DeepLabV3plus":config_file_DLabV4,
        "Segformer":config_file_SegF4})
