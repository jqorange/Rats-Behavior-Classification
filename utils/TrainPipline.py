import torch
import numpy as np
import os
import glob
import re
from utils.trainer import FusionTrainer
from utils.data_loader import DataLoader  # 使用之前创建的数据加载器


class TrainPipline:
    def __init__(self, data_path, save_path, session_names, N_feat_IMU, N_feat_DLC, num_classes,spilit_num, device='auto', max_samples_per_session=None):
        """
        初始化真实数据训练器

        Args:
            data_path (str): 数据路径
            device (str): 设备选择 ('auto', 'cuda', 'cpu')
            max_samples_per_session (int): 每个session最大样本数，用于减少内存使用
        """
        self.data_path = data_path
        self.save_path = save_path
        self.device = self._setup_device(device)
        self.max_samples_per_session = max_samples_per_session

        # 数据参数
        self.N_feat_IMU = N_feat_IMU
        self.N_feat_DLC = N_feat_DLC
        self.num_classes = num_classes  # 根据你的实际类别数调整

        # 会话名称
        self.session_names = session_names

        # 训练和测试会话划分
        self.train_sessions = self.session_names[:spilit_num]  # 前5个session用于训练
        self.test_sessions = self.session_names[spilit_num:]  # 最后1个session用于测试

        # 初始化数据加载器
        self.data_loader = DataLoader(self.session_names, data_path)

        print(f"使用设备: {self.device}")
        print(f"训练会话: {self.train_sessions}")
        print(f"测试会话: {self.test_sessions}")
        if max_samples_per_session:
            print(f"每个session最大样本数: {max_samples_per_session:,}")

    def _setup_device(self, device):
        """设置计算设备"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA不可用，改用CPU")
            device = 'cpu'

        return device

    def load_and_prepare_data(self):
        """加载并准备训练数据（对齐各模态长度，并限制最大样本数，无随机采样）"""
        print("\n=== 加载数据 ===")

        # 加载所有数据
        self.data_loader.load_all_data()

        def prepare_session(data_dicts, session, desc):
            """
            对单个 session 的数据做对齐和截断。
            data_dicts: 模态数据字典列表，如 [train_IMU, train_DLC, ...]
            返回：截断后的数组列表，或 None（数据缺失）
            """
            arrays = []
            # 读取并检查原始数据
            for d in data_dicts:
                arr = d.get(session)
                if arr is None or len(arr) == 0:
                    return None
                arrays.append(arr.astype(np.float32))
            # 对齐到最小长度
            min_len = min(len(a) for a in arrays)
            # 限制最大样本数
            if self.max_samples_per_session:
                min_len = min(min_len, self.max_samples_per_session)
            # 截断所有数组
            truncated = [a[:min_len] for a in arrays]
            shapes = ", ".join(str(t.shape) for t in truncated)
            print(f"✓ {session} ({desc}): 截断后长度={min_len}, 形状={shapes}")
            return truncated

        # ——— 无监督训练数据 ———
        train_IMU_list, train_DLC_list = [], []
        self.unsup_by_session = {}
        for session in self.train_sessions:
            out = prepare_session(
                [self.data_loader.train_IMU, self.data_loader.train_DLC],
                session, "无监督 IMU/DLC"
            )
            if out:
                imu, dlc = out
                train_IMU_list.append(imu)
                train_DLC_list.append(dlc)
                self.unsup_by_session[session] = (imu, dlc)
        if not train_IMU_list:
            raise ValueError("没有找到无监督训练数据")
        self.train_data_IMU = np.concatenate(train_IMU_list, axis=0)
        self.train_data_DLC = np.concatenate(train_DLC_list, axis=0)
        print(f"合并后无监督数据: IMU {self.train_data_IMU.shape}, DLC {self.train_data_DLC.shape}")

        # ——— 监督训练数据 ———
        sup_IMU_list, sup_DLC_list, sup_labels_list = [], [], []
        self.sup_by_session = {}
        for session in self.train_sessions:
            out = prepare_session(
                [self.data_loader.train_sup_IMU,
                 self.data_loader.train_sup_DLC,
                 self.data_loader.train_labels],
                session, "监督 IMU/DLC/Labels"
            )
            if out:
                imu, dlc, labels = out
                sup_IMU_list.append(imu)
                sup_DLC_list.append(dlc)
                sup_labels_list.append(labels)
                self.sup_by_session[session] = (imu, dlc, labels)

        if sup_IMU_list:
            self.train_sup_IMU = np.concatenate(sup_IMU_list, axis=0)
            self.train_sup_DLC = np.concatenate(sup_DLC_list, axis=0)
            self.train_labels = np.concatenate(sup_labels_list, axis=0)
            print(
                f"合并后监督数据: IMU {self.train_sup_IMU.shape}, DLC {self.train_sup_DLC.shape}, Labels {self.train_labels.shape}")
        else:
            print("⚠️ 没有找到监督训练数据")
            self.train_sup_IMU = np.empty((0, 0), dtype=np.float32)
            self.train_sup_DLC = np.empty((0, 0), dtype=np.float32)
            self.train_labels = np.empty((0,), dtype=np.float32)

        # ——— 测试数据 ———
        test_IMU_list, test_DLC_list, test_labels_list = [], [], []
        for session in self.test_sessions:
            out = prepare_session(
                [self.data_loader.train_sup_IMU,
                 self.data_loader.train_sup_DLC,
                 self.data_loader.train_labels],
                session, "测试 IMU/DLC/Labels"
            )
            if out:
                imu, dlc, labels = out
                test_IMU_list.append(imu)
                test_DLC_list.append(dlc)
                test_labels_list.append(labels)

        if test_IMU_list:
            self.test_data_IMU = np.concatenate(test_IMU_list, axis=0)
            self.test_data_DLC = np.concatenate(test_DLC_list, axis=0)
            self.test_labels = np.concatenate(test_labels_list, axis=0)
            print(
                f"合并后测试数据: IMU {self.test_data_IMU.shape}, DLC {self.test_data_DLC.shape}, Labels {self.test_labels.shape}")
        else:
            print("⚠️ 没有找到测试数据")
            self.test_data_IMU = np.empty((0, 0), dtype=np.float32)
            self.test_data_DLC = np.empty((0, 0), dtype=np.float32)
            self.test_labels = np.empty((0,), dtype=np.float32)

        return True

    def initialize_trainer(self, **kwargs):
        """初始化FusionTrainer"""
        print("\n=== 初始化模型 ===")

        # 默认参数
        default_params = {
            'N_feat_A': self.N_feat_IMU,
            'N_feat_B': self.N_feat_DLC,
            'num_classes': self.num_classes,
            'mask_type': 'binomial',
            'd_model': 128,
            'nhead': 8,
            'hidden_dim': 256,
            'device': self.device,
            'lr_encoder': 0.0001,
            'lr_classifier': 0.001,
            'batch_size': 32,
            'temporal_unit': 1,
            'contrastive_epochs': 50,
            'mlp_epochs': 100,
            'save_path': self.save_path,
            'save_gap': 3,
            'n_stable': 1,
            'n_adapted': 2,
            'n_all': 3,
            'use_amp': False
        }

        # 更新参数
        default_params.update(kwargs)

        # 创建训练器
        self.trainer = FusionTrainer(**default_params)

        print(f"模型参数数量:")
        print(f"  - Encoder: {sum(p.numel() for p in self.trainer.encoder_fusion.parameters()):,}")


        return self.trainer

    def test_model_components(self):
        """测试模型组件"""
        print("\n=== 测试模型组件 ===")

        # 测试编码器
        self.trainer.encoder_fusion.eval()
        with torch.no_grad():
            # 使用真实数据的一小部分进行测试
            test_size = min(2, len(self.train_data_IMU))
            test_length = min(50, self.train_data_IMU.shape[1])

            test_IMU = torch.tensor(self.train_data_IMU[:test_size, :test_length]).to(self.device)
            test_DLC = torch.tensor(self.train_data_DLC[:test_size, :test_length]).to(self.device)

            test_output = self.trainer.encoder_fusion(test_IMU, test_DLC)
            print(f"✓ 编码器测试通过: 输入IMU {test_IMU.shape}, DLC {test_DLC.shape} -> 输出 {test_output.shape}")
            print(f"  输出范围: [{test_output.min():.4f}, {test_output.max():.4f}]")

        self.trainer.encoder_fusion.train()

    def train_model(self, start_epoch=0, verbose=True):
        """训练模型"""
        print("\n=== 开始训练 ===")

        try:
            # 检查数据
            if len(self.train_sup_IMU) == 0:
                print("⚠️ 没有监督数据，只进行无监督训练")
                train_sup_IMU = None
                train_sup_DLC = None
                train_labels = None
            else:
                train_sup_IMU = self.train_sup_IMU
                train_sup_DLC = self.train_sup_DLC
                train_labels = self.train_labels

            if len(self.test_data_IMU) == 0:
                print("⚠️ 没有测试数据，跳过测试评估")
                test_data_IMU = None
                test_data_DLC = None
                test_labels = None
            else:
                test_data_IMU = self.test_data_IMU
                test_data_DLC = self.test_data_DLC
                test_labels = self.test_labels

            # 训练模型
            losses = self.trainer.fit(
                unsup_sessions=self.unsup_by_session,
                train_data_A=self.train_data_IMU,
                train_data_B=self.train_data_DLC,
                train_data_sup_A=train_sup_IMU,
                train_data_sup_B=train_sup_DLC,
                labels_sup=train_labels,
                test_data_A=test_data_IMU,
                test_data_B=test_data_DLC,
                test_labels=test_labels,
                verbose=verbose,
                start_epoch=start_epoch
            )

            print("\n=== 训练完成 ===")
            print(f"对比学习损失历史: {len(losses['contrastive'])} 个epoch")


            return losses

        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_model(self):
        """评估模型性能"""
        print("\n=== 模型评估 ===")

        if len(self.test_data_IMU) == 0:
            print("⚠️ 没有测试数据，跳过评估")
            return None

        try:
            # 预测
            predictions = self.trainer.predict(self.test_data_IMU, self.test_data_DLC)

            print(f"预测结果形状: {predictions.shape}")
            print(f"真实标签形状: {self.test_labels.shape}")

            # 如果有时间维度，取平均
            if predictions.ndim == 3:  # (N, T, C)
                predictions_avg = predictions.mean(axis=1)  # (N, C)
            else:
                predictions_avg = predictions

            # 计算准确率（如果是多标签分类）
            if self.test_labels.ndim == 2:  # 多标签
                pred_binary = (predictions_avg > 0.5).astype(int)
                accuracy = (pred_binary == self.test_labels).mean()
                print(f"多标签准确率: {accuracy:.4f}")

                # 每个类别的准确率
                for i in range(self.num_classes):
                    class_acc = (pred_binary[:, i] == self.test_labels[:, i]).mean()
                    print(f"  类别 {i}: {class_acc:.4f}")

            return predictions_avg

        except Exception as e:
            print(f"❌ 评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def run_full_pipeline(self, resume=True, **trainer_kwargs):
        """运行完整的训练流水线

        Args:
            resume (bool): Whether to resume from the latest checkpoint
        """
        """运行完整的训练流水线"""
        print("🚀 开始完整训练流水线...")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"设备: {self.device}")

        try:
            # 1. 加载数据
            self.load_and_prepare_data()

            # 2. 初始化模型
            self.initialize_trainer(**trainer_kwargs)
            if resume:
                pattern = os.path.join(self.save_path, "encoder_*.pkl")
                checkpoints = glob.glob(pattern)
                max_cycle = -1
                for ckpt in checkpoints:
                    m = re.search(r"encoder_(\d+)\.pkl", os.path.basename(ckpt))
                    if m:
                        num = int(m.group(1))
                        if num > max_cycle:
                            max_cycle = num
                if max_cycle >= 0:
                    print(f"Resuming from checkpoint epoch {max_cycle}")
                    self.trainer.load(max_cycle)
                    start_epoch = max_cycle + 1
            else:
                start_epoch = 0

                # 3. 测试模型组件
            self.test_model_components()

            # 4. 训练模型
            losses = self.train_model(start_epoch=start_epoch)

            if losses is None:
                print("❌ 训练失败")
                return False


            print("\n" + "=" * 60)
            print("🎉 训练流水线完成!")
            print(f"✅ 数据加载: 成功")
            print(f"✅ 模型训练: 成功")


            return True

        except Exception as e:
            print(f"\n❌ 流水线执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


