import warnings

from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from SeqSNN.dataset import DATASETS
from SeqSNN.runner import RUNNERS
from SeqSNN.network import NETWORKS

warnings.filterwarnings("ignore")


@configclass
class SeqSNNConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    runner: RegistryConfig[RUNNERS]
    runtime: RuntimeConfig = RuntimeConfig()


def run_train(config):
    setup_experiment(config.runtime)
    print_config(config)
    trainset = config.data.build(dataset_name="train")
    validset = config.data.build(dataset_name="valid")
    testset = config.data.build(dataset_name="test")
    network = config.network.build(
        input_size=trainset.num_variables, max_length=trainset.max_seq_len,
    #     clamp=config.network.clamp,
    #     # clampMin=config.network.clampMin,
    # quantize=config.network.quantize
    )
    runner = config.runner.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size=config.runner.out_size or trainset.num_classes,
    )
    # runner.calibrate(
    #     calibset=trainset,
    #     num_batches=50,          # 看你数据大小，8~16 都行
    #     q_high=0.95,            # 上界分位点；不够可以调到 0.95
    #     rel_mse_target=0.01,    # 允许 MSE ≈ 方差的 1%
    #     batch_size=24,          # 校准时可以用小 batch
    #     print_summary=True,
    # )
    runner.fit(trainset, validset, testset)
    runner.predict(trainset, "train")
    runner.predict(validset, "valid")
    runner.predict(testset, "test")

    return runner


if __name__ == "__main__":
    _config = SeqSNNConfig.fromcli()
    run_train(_config)
