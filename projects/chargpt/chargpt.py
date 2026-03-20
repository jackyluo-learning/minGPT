"""
训练一个字符级语言模型。
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # 系统设置
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # 数据设置
    C.data = CharDataset.get_default_config()

    # 模型设置
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # 训练器设置
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # 我们使用的模型非常小，因此可以训练得快一点

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    发射字符批次
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('数据共有 %d 个字符，其中唯一字符有 %d 个。' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # 从数据中抓取一个包含 (block_size + 1) 个字符的块
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # 将每个字符编码为整数
        dix = [self.stoi[s] for s in chunk]
        # 以张量形式返回
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # 获取默认配置和来自命令行的覆盖参数（如果有）
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # 构造训练数据集
    text = open('input.txt', 'r').read() # 别担心，我们不会耗尽文件句柄
    train_dataset = CharDataset(config.data, text)

    # 构造模型
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # 构造训练器对象
    trainer = Trainer(config.trainer, model, train_dataset)

    # 迭代回调函数
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; 迭代 {trainer.iter_num}: 训练损失 {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # 评估训练和测试得分
            model.eval()
            with torch.no_grad():
                # 从模型采样...
                context = "O God, O God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # 保存最新模型
            print("正在保存模型")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # 将模型恢复为训练模式
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # 运行优化过程
    trainer.run()
