import optuna
from mmengine.runner import Runner
from mmengine.config import Config

def objective(trial):
    cfg = Config.fromfile('')

    # 需要搜索的超参数


    runner = Runner.from_cfg(cfg)
    runner.train()
    return max(runner.message_hub.log_scalars['val/mIoU'].data[-2])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)