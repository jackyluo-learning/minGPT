"""
GPT 语言模型的完整定义，全部包含在这一单个文件中。

参考资料：
1) OpenAI 发布的官方 GPT-2 TensorFlow 实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch 实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    目前在 Google BERT 仓库中的 GELU 激活函数实现（与 OpenAI GPT 相同）。
    参考资料：Gaussian Error Linear Units (GELU) 论文：https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    一个普通的带有最后投影层的多头掩码自注意力层（multi-head masked self-attention layer）。
    这里可以使用 torch.nn.MultiheadAttention，但我在这里包含了一个显式的实现，
    以展示这里并没有什么太可怕的东西。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 所有头的键（key）、查询（query）、值（value）投影，但是以批处理方式进行
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 正则化
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 因果掩码（causal mask），确保注意力只应用于输入序列中的左侧
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # 批次大小 (batch size), 序列长度 (sequence length), 嵌入维度 (embedding dimensionality) (n_embd)

        # 在批次中计算所有头的查询、键、值，并将头维度移到前面作为批次维度
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 因果自注意力；自注意力机制：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 将所有头的输出并排重新组装

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ 一个普通的 Transformer 块 """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP 前向传播

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class AttentionResiduals(nn.Module):
    """
    Kimi团队《Attention Residuals》论文实现。
    
    传统的残差连接是简单的 x = x + f(x)，随着网络加深，信号会被不断稀释。
    AttnRes 引入了一个学习到的“伪查询”（Pseudo-Query），对之前所有层的输出进行
    动态加权聚合。这种方法能让模型自主选择哪些层的特征对当前状态最重要。
    """
    def __init__(self, config):
        super().__init__()
        # 学习一个 Query 向量，用于询问之前哪些层的信息更有价值
        self.query = nn.Parameter(torch.zeros(config.n_embd))
        # 用于计算注意力的归一化层
        self.ln = nn.LayerNorm(config.n_embd)
        # 初始化
        nn.init.normal_(self.query, std=0.02)

    def forward(self, history):
        # history: 存储之前所有 Block 输出的列表 [(B, T, C), ...]
        if not history:
            return 0
        
        # 1. 堆叠历史状态: (L, B, T, C)，L 为历史层数
        x_stack = torch.stack(history, dim=0)
        
        # 2. 计算 Key: 经过 Norm 的历史状态，保持物理含义一致
        x_norm = self.ln(x_stack)
        
        # 3. 计算注意力分数: Query 与各个历史层输出点积
        # (1, 1, 1, C) * (L, B, T, C) -> sum over C -> (L, B, T)
        logits = (self.query.view(1, 1, 1, -1) * x_norm).sum(dim=-1)
        
        # 4. Softmax 归一化: 得到在“深度”维度上的权重分布
        weights = F.softmax(logits, dim=0) # (L, B, T)
        
        # 5. 加权聚合: 产生这一层的动态残差基底
        out = (weights.unsqueeze(-1) * x_stack).sum(dim=0)
        return out

class GPT(nn.Module):
    """ GPT 语言模型 """

    @staticmethod
    def get_default_config():
        C = CN()
        # 配置中必须给出 model_type 或 (n_layer, n_head, n_embd) 之一
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # 这些选项必须在外部填充
        C.vocab_size = None
        C.block_size = None
        # dropout 超参数
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        # Kimi Attention Residuals 选项
        C.use_attn_res = False
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # 必须且仅能给定其中一个 (XOR)
        if type_given:
            # 从 model_type 转换为详细配置
            config.merge_from_dict({
                # 名称遵循 huggingface 的命名约定
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 1.17 亿参数
                # GPT-2 配置
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 1.24 亿参数
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 3.5 亿参数
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 7.74 亿参数
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 15.58 亿参数
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (还有更多...)
                # 我杜撰的微型模型
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Kimi Attention Residuals 初始化
        self.use_attn_res = config.use_attn_res
        if self.use_attn_res:
            self.attn_res = nn.ModuleList([AttentionResiduals(config) for _ in range(config.n_layer)])
            
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化所有权重，并根据 GPT-2 论文对残差投影应用特殊的缩放初始化
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 报告参数数量（注意我们不计算 lm_head 中的解码器参数）
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        # 为 AttnRes 的 query 提供初始化支持
        elif isinstance(module, AttentionResiduals):
            nn.init.normal_(module.query, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        通过从 huggingface/transformers 检查点复制权重来初始化预训练的 GPT 模型。
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # 创建一个从头开始初始化的 minGPT 模型
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # OpenAI 的模型词汇量
        config.block_size = 1024  # OpenAI 的模型块大小
        model = GPT(config)
        sd = model.state_dict()

        # 初始化一个 huggingface/transformers 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制权重的同时确保所有参数对齐，并在名称和形状上匹配
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # 忽略这些
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本上，OpenAI 的检查点使用 "Conv1D" 模块，但我们只想使用普通的 nn.Linear。
        # 这意味着我们在导入这些权重时必须对它们进行转置
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # 对我们需要转置的 Conv1D 权重进行特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 对其他参数进行普通复制
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        这个冗长的函数不幸地在做一件非常简单且非常防御性的事情：
        我们将模型的所有参数分成两个桶：那些将体验用于正则化的权重衰减（weight decay）的参数，
        以及那些不会的参数（偏置，以及层归一化/嵌入权重）。
        然后我们返回 PyTorch 优化器对象。
        """

        # 将所有参数分为会经历和不会经历正则化权重衰减的两类
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # 完整参数名称
                # 随机提示：因为 named_modules 和 named_parameters 是递归的
                # 我们会多次看到相同的张量 p。但以这种方式进行
                # 允许我们知道任何张量 p 属于哪个父模块……
                if pn.endswith('bias'):
                    # 所有偏置都不会衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # 白名单模块的权重将进行权重衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # 黑名单模块的权重将不会进行权重衰减
                    no_decay.add(fpn)
                # Kimi AttnRes 的 query 向量通常需要权重衰减，因为它们决定了层的重要性
                elif pn.endswith('query') and isinstance(m, AttentionResiduals):
                    decay.add(fpn)

        # 验证我们考虑了每个参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # 创建 PyTorch 优化器对象
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # 形状 (1, t)

        # GPT 模型本身的前向传播
        tok_emb = self.transformer.wte(idx) # 形状为 (b, t, n_embd) 的标记嵌入
        pos_emb = self.transformer.wpe(pos) # 形状为 (1, t, n_embd) 的位置嵌入
        x = self.transformer.drop(tok_emb + pos_emb)
        
        if self.use_attn_res:
            # 初始化历史记录，包含 Embedding 输出
            history = [x]
            for i, block in enumerate(self.transformer.h):
                # 1. 动态聚合之前所有层的状态作为本层的残差基底 (x_base)
                x_base = self.attn_res[i](history)
                # 2. 计算残差增量 delta = block(x_base) - x_base
                # 这种实现方式最为直接，不需要修改 Block 内部逻辑
                delta = block(x_base) - x_base
                # 3. 更新当前状态为 基底 + 增量
                x = x_base + delta
                # 4. 将新状态存入历史
                history.append(x)
        else:
            # 标准 GPT 前向传播
            for block in self.transformer.h:
                x = block(x)
                
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # 如果我们也被给了一些期望的目标，则计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        接收一个索引的条件序列 idx (形状为 (b,t) 的 LongTensor)，并完成该序列 max_new_tokens 次，
        每次将预测结果反馈回模型。最可能的情况是，你会希望确保处于 model.eval() 运行模式。
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文变得太长，我们必须在 block_size 处对其进行裁剪
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # 前向传播模型以获取序列中索引的 logits
            logits, _ = self(idx_cond)
            # 提取最后一步的 logits 并按所需温度（temperature）进行缩放
            logits = logits[:, -1, :] / temperature
            # 可选地，将 logits 裁剪为仅保留前 k 个选项
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用 softmax 将 logits 转换为（归一化的）概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样或选择最可能的元素
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # 将采样的索引追加到正在运行的序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
