import ppdet
import paddle
from ppdet.engine import Trainer, init_parallel_env
import ppdet.utils.check as check

cfg_file = 'fire.yaml'
cfg = ppdet.core.workspace.load_config(cfg_file)

init_parallel_env()
place = paddle.set_device('gpu')
check.check_config(cfg)

print(cfg)
trainer = ppdet.engine.Trainer(cfg, mode='train')
trainer.load_weights(cfg['pretrain_weights'])
trainer.train(True)
