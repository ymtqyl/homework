import logging  # 用于日志记录的标准库
import time  # 用于处理时间相关操作
from ruamel import yaml  # 用于读写YAML配置文件（支持注释和复杂结构）


class AverageMeter(object):
    """
    用于计算和存储数值的平均值、当前值、总和及计数的工具类
    常用于跟踪训练过程中的损失、准确率等指标
    """

    def __init__(self):
        """初始化所有统计变量"""
        self.reset()  # 调用重置方法初始化参数

    def reset(self):
        """重置所有统计变量为初始状态"""
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数

    def update(self, val, n=1):
        """
        更新统计变量

        参数:
            val: 当前批次的数值（如损失值、准确率）
            n: 该数值对应的样本数量（默认为1，用于加权计算）
        """
        self.val = val  # 记录当前值
        self.sum += val * n  # 累加总数值（加权）
        self.count += n  # 累加样本数量
        self.avg = self.sum / self.count  # 计算平均值


def Prepare_logger(args, eval=False):
    """
    配置并返回日志记录器，同时输出到控制台和文件

    参数:
        args: 包含实验配置的参数对象，需包含`snapshot_pref`（日志存储路径）
        eval: 是否为评估阶段（决定日志文件名）

    返回:
        logger: 配置好的日志记录器
    """
    # 获取名为当前模块的日志器
    logger = logging.getLogger(__name__)
    # 禁止日志向上传播（避免被父 logger 重复处理）
    logger.propagate = False
    # 设置日志级别为INFO（只记录INFO及以上级别日志）
    logger.setLevel(logging.INFO)

    # 配置控制台输出处理器
    handler = logging.StreamHandler()
    # 定义日志格式：时间 + 级别 + 消息
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)  # 输出所有级别的日志
    logger.addHandler(handler)

    # 生成带时间戳的日志文件名
    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))  # 格式：年月日时分
    # 训练阶段和评估阶段使用不同的日志文件名
    if not eval:
        logfile = args.snapshot_pref + date + '.log'  # 训练日志路径
    else:
        logfile = args.snapshot_pref + f'/{date}-Eval.log'  # 评估日志路径

    # 配置文件输出处理器
    file_handler = logging.FileHandler(logfile, mode='w')  # 'w'表示覆盖写入
    file_handler.setLevel(logging.INFO)  # 只记录INFO及以上级别日志
    file_handler.setFormatter(formatter)  # 使用相同的格式
    logger.addHandler(file_handler)

    return logger


def get_configs(dataset):
    """
    从配置文件中获取指定数据集的配置信息

    参数:
        dataset: 数据集名称（用于索引配置）

    返回:
        该数据集对应的配置字典
    """
    # 加载数据集配置文件（YAML格式）
    data = yaml.load(open('./configs/dataset_cfg.yaml'))
    # 返回指定数据集的配置
    return data[dataset]


def get_and_save_args(parser):
    """
    解析命令行参数，与默认配置合并，并保存最终配置到文件

    参数:
        parser: 已定义的命令行参数解析器（argparse.ArgumentParser）

    返回:
        default_config: 合并后的最终配置字典
    """
    # 解析命令行参数
    args = parser.parse_args()

    # 加载默认配置文件（使用RoundTripLoader保留原格式和注释）
    default_config = yaml.load(
        open('./configs/default_config.yaml', 'r'),
        Loader=yaml.RoundTripLoader
    )

    # 将命令行参数转换为字典
    current_config = vars(args)

    # 合并命令行参数到默认配置（命令行参数优先级更高）
    for k, v in current_config.items():
        if k in default_config:  # 只处理默认配置中已有的键
            # 若命令行参数不为None且与默认值不同，则更新
            if (v != default_config[k]) and (v is not None):
                print(f"Updating:  {k}: {default_config[k]} (default) ----> {v}")
                default_config[k] = v

    # 保存合并后的配置到文件（使用RoundTripDumper保留格式）
    yaml.dump(
        default_config,
        open('./current_configs.yaml', 'w'),
        indent=4,
        Dumper=yaml.RoundTripDumper
    )

    return default_config