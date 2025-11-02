import os
import shutil  # 用于文件和目录的复制、移动、删除等操作


class Recorder(object):
    """
    实验记录器类，用于管理实验过程中的文件存储、日志记录和代码备份
    主要功能：创建实验目录、备份代码文件、记录参数配置和实验日志
    """

    def __init__(self, snapshot_pref, ignore_folder):
        """
        初始化记录器，创建实验所需的目录结构

        参数:
            snapshot_pref: 实验结果存储的根目录路径
            ignore_folder: 需要忽略的文件夹名称（备份代码时不复制该文件夹下的内容）
        """
        # 若存储目录不存在，则创建该目录
        if not os.path.isdir(snapshot_pref):
            os.mkdir(snapshot_pref)
        self.save_path = snapshot_pref  # 保存实验根目录路径

        # 定义各类日志文件路径
        self.log_file = self.save_path + "log.txt"  # 实验过程日志文件
        self.readme = self.save_path + "README.md"  # 实验说明文档
        self.opt_file = self.save_path + "opt.log"  # 实验参数配置文件
        self.code_path = os.path.join(self.save_path, "code/")  # 代码备份目录

        # 移除已存在的README文件（避免历史内容干扰）
        if os.path.isfile(self.readme):
            os.remove(self.readme)

        # 若代码备份目录不存在，则创建该目录
        if not os.path.isdir(self.code_path):
            os.mkdir(self.code_path)

        # 备份当前项目代码到code目录（忽略指定文件夹）
        self.copy_code(dst=self.code_path, ignore_folder=ignore_folder)

        # 注释部分：原本计划用于权重文件和权重可视化图的存储目录管理
        # 目前未启用，保留了历史开发痕迹
        """if os.path.isdir(self.weight_folder):
            shutil.rmtree(self.weight_folder, ignore_errors=True)
        os.mkdir(self.weight_folder)
        if os.path.isdir(self.weight_fig_folder):
            shutil.rmtree(self.weight_fig_folder, ignore_errors=True)
        os.mkdir(self.weight_fig_folder)"""

        # 打印实验结果存储路径，方便用户查看
        print("\n======> Result will be saved at: ", self.save_path)

    def copy_code(self, src="./", dst="./code/", ignore_folder='Exps'):
        """
        备份项目代码到指定目录，用于实验复现和版本追溯

        参数:
            src: 源代码根目录（默认为当前工作目录）
            dst: 代码备份目标目录
            ignore_folder: 需要忽略的文件夹名称（不备份该文件夹下的内容）
        """
        import uuid  # 用于生成唯一标识符，避免目录重名
        # 若目标目录已存在，在目录名后添加UUID确保唯一性
        if os.path.isdir(dst):
            # 生成新的目标路径（原路径+唯一标识符）
            dst = "/".join(dst.split("/")) + "code_" + str(uuid.uuid4()) + "/"

        file_abs_list = []  # 存储需要备份的文件绝对路径
        src_abs = os.path.abspath(src)  # 获取源代码根目录的绝对路径

        # 遍历源代码目录下的所有文件
        for root, dirs, files in os.walk(src_abs):
            # 跳过包含忽略文件夹的路径
            if ignore_folder not in root:
                for name in files:
                    # 收集符合条件的文件绝对路径
                    file_abs_list.append(root + "/" + name)

        # 复制收集到的文件到目标目录
        for file_abs in file_abs_list:
            file_split = file_abs.split("/")[-1].split('.')  # 分割文件名和后缀
            # 筛选条件：文件大小小于10MB，且不是.pyc（Python编译文件）
            if os.path.getsize(file_abs) / 1024 / 1024 < 10 and not file_split[-1] == "pyc":
                src_file = file_abs  # 源文件路径
                # 构建目标文件路径（保持原目录结构）
                dst_file = dst + file_abs.replace(src_abs, "")
                # 若目标文件的父目录不存在，则创建
                if not os.path.exists(os.path.dirname(dst_file)):
                    os.makedirs(os.path.dirname(dst_file))
                # 复制文件（捕获可能的错误）
                try:
                    shutil.copyfile(src=src_file, dst=dst_file)
                except:
                    print("copy file error")  # 打印复制错误信息

    def writeopt(self, opt):
        """
        记录实验参数配置（如超参数、路径等）

        参数:
            opt: 包含实验参数的对象（通常是argparse.Namespace）
        """
        with open(self.opt_file, "w") as f:
            # 遍历对象的所有属性，写入文件
            for k, v in opt.__dict__.items():
                f.write(str(k) + ": " + str(v) + "\n")

    def writelog(self, input_data):
        """
        记录实验过程日志（如训练指标、错误信息等）

        参数:
            input_data: 需要写入日志的内容
        """
        txt_file = open(self.log_file, 'a+')  # 以追加模式打开日志文件
        txt_file.write(str(input_data) + "\n")  # 写入内容并换行
        txt_file.close()  # 关闭文件

    def writereadme(self, input_data):
        """
        记录实验说明文档（如实验目的、关键结论等）

        参数:
            input_data: 需要写入README的内容
        """
        txt_file = open(self.readme, 'a+')  # 以追加模式打开README
        txt_file.write(str(input_data) + "\n")  # 写入内容并换行
        txt_file.close()  # 关闭文件

    # 以下方法目前未实现完整功能，可能是预留的网络结构可视化功能
    def gennetwork(self, var):
        """生成网络结构可视化（预留方法）"""
        self.graph.draw(var=var)  # 假设graph是一个可视化工具对象

    def savenetwork(self):
        """保存网络结构可视化图（预留方法）"""
        self.graph.save(file_name=self.save_path + "network.svg")  # 保存为SVG格式

    # 以下方法被注释，原本用于记录和可视化网络权重
    """def writeweights(self, input_data, block_id, layer_id, epoch_id):
        # 记录网络权重信息（按 epoch、block、layer 分类）
        txt_path = self.weight_folder + "conv_weight_" + str(epoch_id) + ".log"
        txt_file = open(txt_path, 'a+')
        write_str = "%d\t%d\t%d\t" % (epoch_id, block_id, layer_id)
        for x in input_data:
            write_str += str(x) + "\t"
        txt_file.write(write_str+"\n")

    def drawhist(self):
        # 绘制权重分布直方图（预留方法）
        drawer = DrawHistogram(txt_folder=self.weight_folder, fig_folder=self.weight_fig_folder)
        drawer.draw()"""