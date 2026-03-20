"""
训练一个 GPT 模型来进行 n 位数加法。
"""

import os
import sys
import json

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
    C.system.work_dir = './out/adder'

    # 数据设置
    C.data = AdditionDataset.get_default_config()

    # 模型设置
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'

    # 训练器设置
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # 我们使用的模型非常小，因此可以训练得快一点

    return C

# -----------------------------------------------------------------------------

class AdditionDataset(Dataset):
    """
    创建 n 位数加法问题。例如，如果 n=2，那么一个加法问题的例子
    是 85 + 50 = 135。这个问题将被表示为 GPT 的以下字符串：

    "8550531"

    这是因为：
    - 我们丢弃了 + 和 =，它们是不必要的。我们只是对输入数字的数字位进行级联编码。
    - 结果 135 被反向编码为 531，这是为了让 GPT 模型更容易学习加法，
      因为加法算法的工作原理就是从低位到高位。

    再举一个例子，问题 6 + 39 = 45 将被编码为：

    "0639054"

    你会注意到我们使用了零填充，以确保生成的字符串大小始终相同：n + n + (n + 1)。
    当 n=2 时，长度为 7。
    在测试时，我们将输入加法问题的前 2n 个数字，并希望 GPT 模型
    能正确完成序列的后 (n+1) 个数字。
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 2
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split # 训练/测试 (train/test)

        # 将所有加法问题拆分为训练数据或测试数据
        ndigit = self.config.ndigit
        assert ndigit <= 3, "下面的行将非常消耗内存，未来可能需要重构以支持更大位数"
        num = (10**ndigit)**2 # 使用 n 位数进行加法的全量可能问题数
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num*0.2), 500) # 整个数据集的 20%，或者最多 500 个
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 # 数字 0..9

    def get_block_size(self):
        # a, b, a+b，以及由于潜在进位导致的 +1，
        # 但随后也要 -1，因为最后一个数字永远不会被回传作为输入
        # 因为没有显式的 <EOS> 标记需要预测，它是隐含的
        return 3*self.config.ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        # 给定一个问题索引 idx，首先恢复关联的 a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # 计算加法问题 a + b 的“标签”
        c = a + b
        # 将 a, b, c 的数字编码为字符串
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # 反转 c 使加法学习更容易
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # 将每个字符转换为其标记索引
        # x 将是 GPT 的输入，y 将是关联的期望输出
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # 预测序列中的下一个标记
        y[:ndigit*2-1] = -1 # 我们只在输出位置进行训练。-1 将使损失被屏蔽（mask）为零
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # 获取默认配置和来自命令行的覆盖参数（如果有）
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # 构造训练和测试数据集
    train_dataset = AdditionDataset(config.data, split='train')
    test_dataset  = AdditionDataset(config.data, split='test')

    # 构造模型
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # 构造训练器对象
    trainer = Trainer(config.trainer, model, train_dataset)

    # 模型评估的辅助函数
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        ndigit = config.data.ndigit
        results = []
        mistakes_printed_already = 0
        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            # 仅隔离输入序列的前两位（数字位）
            d1d2 = x[:, :ndigit*2]
            # 让模型对序列的其余部分进行采样
            d1d2d3 = model.generate(d1d2, ndigit+1, do_sample=False) # 使用贪婪的 argmax，而非采样
            # 隔离采样序列的最后一部分（结果位）
            d3 = d1d2d3[:, -(ndigit+1):]
            d3 = d3.flip(1) # 将数字反转回“正常”顺序
            # 从单个数字位解码为整数
            d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
            d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
            d3i_pred = (d3 * factors).sum(1)
            d3i_gt = d1i + d2i # 手动计算真值（ground truth）
            # 在这一行，Software 1.0 与 Software 2.0 正在激烈对决，哈哈
            correct = (d3i_pred == d3i_gt).cpu()
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 5: # 仅打印最多 5 个错误以了解情况
                    mistakes_printed_already += 1
                    print("GPT 认为 %d + %d = %d，但真值是 %d" % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s 最终得分: %d/%d = %.2f%% 正确" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum()

    # 迭代回调函数
    top_score = 0
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; 迭代 {trainer.iter_num}: 训练损失 {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # 评估训练集和测试集得分
            # 如果 ndigit=2，我们可以负担整个训练集的评估，否则不行
            train_max_batches = {1: None, 2: None, 3: 5}[config.data.ndigit]
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, 'train', max_batches=train_max_batches)
                test_score  = eval_split(trainer, 'test',  max_batches=None)
            score = train_score + test_score
            # 如果这是我们目前见过的最高得分，则保存模型
            if score > top_score:
                top_score = score
                print(f"保存模型，新的最高得分为 {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # 将模型恢复为训练模式
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # 运行优化过程
    trainer.run()
