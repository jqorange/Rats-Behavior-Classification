import torch
import numpy as np
import pandas as pd
import os


class DataLoader:
    def __init__(self, session_names, base_path):
        """
        初始化数据加载器

        Args:
            base_path (str): 数据的基础路径，应该包含IMU、DLC、sup_data、sup_label文件夹
        """
        self.base_path = base_path
        self.session_names = session_names

        # 定义各个数据路径
        self.imu_path = os.path.join(base_path, "IMU")
        self.dlc_path = os.path.join(base_path, "DLC")
        self.sup_data_path = os.path.join(base_path, "sup_data")
        self.sup_label_path = os.path.join(base_path, "sup_label")

        # 初始化数据字典
        self.train_IMU = {}
        self.train_DLC = {}
        self.train_sup = {}
        self.train_labels = {}
        self.train_sup_IMU = {}  # 从train_sup分离出的IMU数据
        self.train_sup_DLC = {}  # 从train_sup分离出的DLC数据

    def load_original_data(self):
        """加载原始的IMU和DLC数据"""
        print("正在加载原始IMU和DLC数据...")

        for session in self.session_names:
            try:
                # 加载IMU数据
                imu_file = os.path.join(self.imu_path, f"samples_{session}_IMU_Final_{session}.npy")
                self.train_IMU[session] = np.load(imu_file)

                # 加载DLC数据
                dlc_file = os.path.join(self.dlc_path, f"samples_{session}_DLC_{session}.npy")
                self.train_DLC[session] = np.load(dlc_file)

                print(f"✓ {session}: IMU {self.train_IMU[session].shape}, DLC {self.train_DLC[session].shape}")

            except FileNotFoundError as e:
                print(f"✗ 文件未找到: {session} - {e}")
            except Exception as e:
                print(f"✗ 加载错误: {session} - {e}")

    def load_supervised_data(self):
        """加载监督学习数据和标签"""
        print("\n正在加载监督学习数据...")

        for session in self.session_names:
            try:
                # 加载监督数据
                sup_data_file = os.path.join(self.sup_data_path, f"sup_data_{session}.npy")
                self.train_sup[session] = np.load(sup_data_file)

                # 加载监督标签
                sup_label_file = os.path.join(self.sup_label_path, f"sup_label_{session}.npy")
                self.train_labels[session] = np.load(sup_label_file)

                print(
                    f"✓ {session}: Sup_data {self.train_sup[session].shape}, Labels {self.train_labels[session].shape}")

            except FileNotFoundError as e:
                print(f"✗ 文件未找到: {session} - {e}")
            except Exception as e:
                print(f"✗ 加载错误: {session} - {e}")

    def split_supervised_data(self):
        """将train_sup的最后一维分割为IMU(前35列)和DLC(后20列)数据"""
        print("\n正在分割监督数据...")

        for session in self.session_names:
            if session in self.train_sup and len(self.train_sup[session]) > 0:
                try:
                    # 分割最后一维：前35列为IMU，后20列为DLC
                    self.train_sup_IMU[session] = self.train_sup[session][:, :, :35]
                    self.train_sup_DLC[session] = self.train_sup[session][:, :, 35:]

                    print(
                        f"✓ {session}: Sup_IMU {self.train_sup_IMU[session].shape}, Sup_DLC {self.train_sup_DLC[session].shape}")

                except Exception as e:
                    print(f"✗ 分割错误: {session} - {e}")
            else:
                print(f"⚠ {session}: 没有监督数据可分割")
                self.train_sup_IMU[session] = np.array([])
                self.train_sup_DLC[session] = np.array([])

    def load_all_data(self):
        """加载所有数据"""
        print("开始加载所有数据...\n")

        self.load_original_data()
        self.load_supervised_data()
        self.split_supervised_data()

        print("\n数据加载完成！🚀")
        return self.get_data_summary()

    def get_data_summary(self):
        """获取数据摘要信息"""
        summary = {
            'train_IMU': self.train_IMU,
            'train_DLC': self.train_DLC,
            'train_sup': self.train_sup,
            'train_labels': self.train_labels,
            'train_sup_IMU': self.train_sup_IMU,
            'train_sup_DLC': self.train_sup_DLC
        }
        return summary

    def get_session_data(self, session_name):
        """获取特定session的所有数据"""
        if session_name not in self.session_names:
            print(f"错误: {session_name} 不在可用的session列表中")
            return None

        return {
            'train_IMU': self.train_IMU.get(session_name, np.array([])),
            'train_DLC': self.train_DLC.get(session_name, np.array([])),
            'train_sup': self.train_sup.get(session_name, np.array([])),
            'train_labels': self.train_labels.get(session_name, np.array([])),
            'train_sup_IMU': self.train_sup_IMU.get(session_name, np.array([])),
            'train_sup_DLC': self.train_sup_DLC.get(session_name, np.array([]))
        }

    def print_data_info(self):
        """打印所有数据的详细信息"""
        print("\n" + "=" * 60)
        print("数据信息汇总")
        print("=" * 60)

        for session in self.session_names:
            print(f"\n📁 Session: {session}")
            print("-" * 40)

            # 原始数据
            if session in self.train_IMU:
                print(f"  原始IMU数据: {self.train_IMU[session].shape}")
            if session in self.train_DLC:
                print(f"  原始DLC数据: {self.train_DLC[session].shape}")

            # 监督数据
            if session in self.train_sup:
                print(f"  监督数据: {self.train_sup[session].shape}")
            if session in self.train_labels:
                print(f"  监督标签: {self.train_labels[session].shape}")

            # 分割后的监督数据
            if session in self.train_sup_IMU:
                print(f"  监督IMU数据: {self.train_sup_IMU[session].shape}")
            if session in self.train_sup_DLC:
                print(f"  监督DLC数据: {self.train_sup_DLC[session].shape}")


