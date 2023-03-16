from colossalai.amp import AMP_TYPE

# hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 20
NUM_CLASSES = 10

# cudnn
cudnn_benchmark = True
cudnn_deterministic = True

CONFIG = dict(fp16=dict(mode=AMP_TYPE.TORCH))
