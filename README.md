# minGPT

![mingpt](mingpt.jpg)

[GPT](https://github.com/openai/gpt-2) 的 PyTorch 重实现，包含训练和推理。minGPT 旨在保持小巧、整洁、可解释且具有教育意义，因为目前大多数 GPT 模型实现都显得有些庞杂。GPT 并不是一个复杂的模型，这个实现大约只有 300 行代码（参见 [mingpt/model.py](mingpt/model.py)）。其核心逻辑是将索引序列输入到 [Transformer](https://arxiv.org/abs/1706.03762) 中，然后输出序列中下一个索引的概率分布。大部分复杂性仅在于为了效率而巧妙地处理批次（batching）（包括跨样本和跨序列长度）。

**注（2023年1月）**：虽然我可能会继续接受并更改一些细节，但 minGPT 目前处于半归档状态。有关最新进展，请参阅我的重写版本 [nanoGPT](https://github.com/karpathy/nanoGPT)。基本上，minGPT 在广泛的地方（笔记本、博客、课程、书籍等）被引用，这使我不太愿意进行我想要进行的重大更改以推进代码。我还想稍微改变一下方向，从纯粹专注于教育转变为仍然简单且可黑客攻击但具有实际应用价值（能够复现中型工业基准，接受一些权衡以获得运行效率等）。

minGPT 库由三个文件组成：[mingpt/model.py](mingpt/model.py) 包含实际的 Transformer 模型定义；[mingpt/bpe.py](mingpt/bpe.py) 包含一个稍微重构的字节对编码器（Byte Pair Encoder），它在文本和整数序列之间进行转换，与 OpenAI 在 GPT 中的做法完全一致；[mingpt/trainer.py](mingpt/trainer.py) 是（与 GPT 无关的）PyTorch 样板代码，用于训练模型。此外，`projects` 文件夹中还有许多使用该库的演示和项目：

- `projects/adder` 从头开始训练一个 GPT 来进行加法运算（灵感来自 GPT-3 论文中的加法部分）
- `projects/chargpt` 训练一个 GPT 在某些输入文本文件上作为字符级语言模型
- `demo.ipynb` 以笔记本格式在一个简单的排序示例上展示了 `GPT` 和 `Trainer` 的极简用法
- `generate.ipynb` 展示了如何加载预训练的 GPT2 并根据某些提示生成文本

### 库安装

如果你想在项目中 `import mingpt`：

```
git clone https://github.com/karpathy/minGPT.git
cd minGPT
pip install -e .
```

### 用法

以下是如何实例化一个 GPT-2（1.24亿参数版本）：

```python
from mingpt.model import GPT
model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257 # OpenAI 的模型词汇量
model_config.block_size = 1024  # OpenAI 的模型块大小（即输入上下文长度）
model = GPT(model_config)
```

以下是如何训练它：

```python
# 你的 torch.utils.data.Dataset 子类，它发出示例
# 长度高达 1024 的 torch LongTensor，其中的整数范围在 [0, 50257) 之间
train_dataset = YourDataset()

from mingpt.trainer import Trainer
train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # 许多可能的选项，见文件
train_config.max_iters = 1000
train_config.batch_size = 32
trainer = Trainer(train_config, model, train_dataset)
trainer.run()
```

有关更具体的示例，请参见 `demo.ipynb`。

### 单元测试

覆盖率目前还不是非常惊人，但是：

```
python -m unittest discover tests
```

### 待办事项 (todos)

- 添加在任意给定文本文件上的 gpt-2 微调演示
- 添加对话代理演示
- 完善现有项目（adder, chargpt）结果的文档
- 添加混合精度和相关的训练扩展功能
- 分布式训练支持
- 在 projects/ 中复现一些基准，例如 text8 或其他语言建模
- 使用正式的日志记录，而不是简陋的打印语句，哈哈
- 我可能应该有一个 requirements.txt 文件...
- 应该能够加载除 gpt2-* 之外的许多其他模型权重

### 参考资料

代码：

- [openai/gpt-2](https://github.com/openai/gpt-2) 在 TensorFlow 中有模型定义，但没有训练代码
- [openai/image-gpt](https://github.com/openai/image-gpt) 在其代码中包含了一些更现代的类似 gpt-3 的修改，也是很好的参考
- [huggingface/transformers](https://github.com/huggingface/transformers) 有一个 [语言建模示例](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)。它功能齐全，但因此追踪起来也有些挑战。例如，一些大型函数在各种分支语句后面有多达 90% 的未使用代码，这在简单语言建模的默认设置中是未使用的。

论文 + 一些实现说明：

#### Improving Language Understanding by Generative Pre-Training (GPT-1)

- 我们的模型很大程度上遵循了最初的 transformer 工作
- 我们训练了一个 12 层的仅解码器（decoder-only）transformer，带有掩码自注意力头（768 维状态和 12 个注意力头）。对于逐位置前馈网络，我们使用了 3072 维的内部状态。
- Adam 最大学习率为 2.5e-4。（后来的 GPT-3 对于这种模型尺寸使用 6e-4）
- 学习率衰减：在前 2000 次更新中从零线性增加，并使用余弦退火方案退火到 0
- 我们在 64 个随机采样的、连续的 512 个标记序列的小批次（minibatches）上训练 100 个轮次（epochs）。
- 由于层归一化（layernorm）在整个模型中被广泛使用，N(0, 0.02) 的简单权重初始化就足够了
- 具有 40,000 次合并的字节对编码（BPE）词汇表
- 残差、嵌入和注意力丢弃（dropout）率为 0.1，用于正则化。
- 修改版的 L2 正则化，建议见 (37)，所有非偏置或增益权重 w = 0.01
- 对于激活函数，我们使用了高斯误差线性单元（GELU）。
- 我们使用了学习的位置嵌入，而不是原始工作中提出的正弦版本
- 对于微调：我们在分类器中添加了丢弃率为 0.1 的 dropout。学习率为 6.25e-5，批次大小为 32。3 个轮次。我们使用线性学习率衰减方案，在 0.2% 的训练中进行预热（warmup）。λ 设置为 0.5。
- GPT-1 模型为 12 层，d_model 为 768，约 1.17 亿参数

#### Language Models are Unsupervised Multitask Learners (GPT-2)

- 层归一化（LayerNorm）被移动到每个子块的输入，类似于预激活残差网络
- 在最后的自注意力块之后添加了一个额外的层归一化。
- 使用了一种改进的初始化，它考虑了随模型深度增加而在残差路径上的累积。我们在初始化时将残差层的权重缩放 1/√N，其中 N 是残差层的数量。（奇怪的是，在他们发布的代码中，我只能找到旧的 0.02 的简单用法……在他们发布的 image-gpt 中，我发现它被用于 c_proj，而且即使那样也只用于 attn，不用于 mlp。嗯。https://github.com/openai/image-gpt/blob/master/src/model.py）
- 词汇表扩展到 50,257
- 上下文大小从 512 增加到 1024 个标记
- 使用了更大的批次大小 512
- GPT-2 使用了 48 层，d_model 为 1600（而原始为 12 层，d_model 为 768）。约 15.42 亿参数

#### Language Models are Few-Shot Learners (GPT-3)

- GPT-3：96 层，96 个头，d_model 为 12,288（1750 亿参数）。
- 类似 GPT-1 的：12 层，12 个头，d_model 768（1.25 亿）
- 我们使用与 GPT-2 相同的模型和架构，包括其中描述的改进初始化、预归一化和可逆标记化
- 我们在 transformer 的层中交替使用稠密和局部带状稀疏注意力模式，类似于 Sparse Transformer
- 我们总是让前馈层的大小是瓶颈层的四倍，dff = 4 * dmodel
- 所有模型都使用 nctx = 2048 个标记的上下文窗口。
- Adam，β1 = 0.9，β2 = 0.95，eps = 10−8
- 所有模型都使用 0.1 的权重衰减来提供少量的正则化。（注意：GPT-1 我相信使用了 0.01，见上文）
- 将梯度的全局范数剪切在 1.0
- 在前 3.75 亿个标记中进行线性学习率预热。然后使用余弦衰减将学习率降低到其值的 10%，在 2600 亿个标记中进行。
- 训练的前 40-120 亿个标记中，将批次大小从一个小值（32k 标记）线性增加到完整值，具体取决于模型大小。
- 始终使用完整的 2048 大小的时间上下文窗口，并带有特殊的 END OF DOCUMENT 标记分隔符

#### Generative Pretraining from Pixels (Image GPT)

- 在处理图像时，我们选择恒等置换 πi = i 对于 1 ≤ i ≤ n，也称为光栅顺序（raster order）。
- 我们通过使用 k-means（k = 512）对 (R, G, B) 像素值进行聚类，创建了我们自己的 9 位调色板。
- 我们最大的模型 iGPT-XL 包含 L = 60 层，并使用 d = 3072 的嵌入尺寸，总计 68 亿参数。
- 我们下一个最大的模型 iGPT-L 与 L = 48 层的 GPT-2 基本上相同，但包含略小的嵌入尺寸 d = 1536（对比 1600），总计 14 亿参数。
- 我们使用与 GPT-2 相同的模型代码，除了我们像在 Sparse Transformer (Child et al., 2019) 中那样以层依赖的方式初始化权重，并对产生 logits 的所有投影进行零初始化。
- 我们还训练了 iGPT-M，一个具有 L = 36 和 d = 1024 的 4.55 亿参数模型
- iGPT-S，一个具有 L = 24 和 d = 512 的 7600 万参数模型（好的，那么有多少个头？看起来 Github 代码声称是 8 个）
- 在预训练 iGPT-XL 时，我们使用 64 的批次大小并训练 200 万次迭代，对于所有其他模型，我们使用 128 的批次大小并训练 100 万次迭代。
- Adam，β1 = 0.9 和 β2 = 0.95
- 学习率预热一个轮次，然后衰减到 0
- 我们没有使用权重衰减，因为应用 0.01 的小权重衰减并没有改变表示质量。
- iGPT-S 学习率 0.003
- 不使用 dropout。

### 许可证

MIT
