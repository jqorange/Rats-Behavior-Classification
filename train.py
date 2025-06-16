import torch
import numpy as np
import os
from utils.TrainPipline import TrainPipline
import argparse

def main(resume=True):
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
   # ["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor","F5D10_outdoor", "F6D5_outdoor_1", "F6D5_outdoor_2"]
    # 数据路径
    data_path = r"D:\Jiaqi\TrainData"
    save_path = r"./checkpoints"
    session_name = ["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor","F5D10_outdoor", "F6D5_outdoor_1", "F6D5_outdoor_2"]
    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        print("请检查数据路径是否正确")
        return False

    # 创建训练器，设置每个session最大样本数以减少内存使用
    real_trainer = TrainPipline(
        data_path,
        save_path,
        session_name,
        N_feat_IMU=29,
        N_feat_DLC=36,
        num_classes=14,
        spilit_num=5,
        device='auto',
        max_samples_per_session=5000000  # 限制每个session最多5万个样本，根据内存情况调整
    )

    # 训练参数
    trainer_params = {
        'mask_type': 'binomial',
        'd_model': 128,
        'nhead': 4,
        'hidden_dim': 4,
        'lr_encoder': 0.001,
        'lr_classifier': 0.001,
        'batch_size': 128,
        'contrastive_epochs': 1,
        'mlp_epochs': 1,
        'save_path': save_path,
        'save_gap': 5,
        'n_cycles': 500,
        'n_stable': 150
    }

    # 运行完整流水线
    success = real_trainer.run_full_pipeline(
        resume=resume,
        **trainer_params
    )

    if success:
        print("\n🚀 训练成功！模型已保存，可以用于推理")

        # 示例：如何使用训练好的模型进行推理
        print("\n=== 推理示例 ===")
        try:
            # 使用一小部分测试数据进行推理演示
            if len(real_trainer.test_data_IMU) > 0:
                sample_size = min(5, len(real_trainer.test_data_IMU))
                sample_IMU = real_trainer.test_data_IMU[:sample_size]
                sample_DLC = real_trainer.test_data_DLC[:sample_size]

                # 编码
                encoded = real_trainer.trainer.encode(sample_IMU, sample_DLC)
                print(f"编码结果形状: {encoded.shape}")

                # 预测
                predictions = real_trainer.trainer.predict(sample_IMU, sample_DLC)
                print(f"预测结果形状: {predictions.shape}")

                print("推理演示完成")
        except Exception as e:
            print(f"推理演示失败: {e}")
    else:
        print("\n❌ 训练失败，请检查数据和代码")

    return success


if __name__ == "__main__":
    success = main()
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', default=True, help='resume training from last checkpoint')
    args = parser.parse_args()
    success = main(resume=args.resume)

    if success:
        print("\n所有流程执行成功!")
    else:
        print("\n存在错误，请检查日志")