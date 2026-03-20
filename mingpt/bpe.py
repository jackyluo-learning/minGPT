"""
bpe 是字节对编码器（Byte Pair Encoder）的缩写。它将任意 utf-8 字符串转换为
整数序列，其中每个整数代表常见字符的小块。此实现基于 openai 的 gpt2 encoder.py：
https://github.com/openai/gpt-2/blob/master/src/encoder.py
但进行了轻微修改，因为原始实现有点令人困惑。
我还尝试尽可能多地添加注释，以表达我对自己所发生事情的理解。
"""

import os
import json
import regex as re
import requests

import torch

# -----------------------------------------------------------------------------

def bytes_to_unicode():
    """
    每个可能的字节（实际上是 0..255 的整数）都被 OpenAI 映射到一个在视觉上代表它的 unicode 字符。
    一些字节因为不会引起任何麻烦而保留了它们的外观。这些定义在列表 bs 中。
    例如：chr(33) 返回 "!"，所以在返回的字典中我们简单地有 d[33] -> "!"。
    然而，例如 chr(0) 是 '\x00'，看起来很丑。所以 OpenAI 将这些字节映射到 chr() 
    返回单个漂亮字符的新范围内的字符。
    所以在最终字典中，我们有 d[0] -> 'Ā'，这只是 chr(0 + 2**8)。
    特别地，空格字符是 32，我们可以通过 ord(' ') 看到。相反，这个函数将空格 (32) 
    偏移 256 到 288，所以 d[32] -> 'Ġ'。
    所以这只是将字节 0..255 简单地一对一映射到“看起来不错”的 unicode 字符，
    无论是它们原来的形式，还是像 'Ā' 或 'Ġ' 等有趣的偏移字符。
    """
    # 188 个原始形式渲染良好且不需要偏移的整数
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:] # bs 中的所有整数 b 将简单地映射到输出字典中的 chr(b)
    # 现在获取其他 68 个确实需要偏移的整数的表示
    # 每个都将映射到 chr(256 + n)，其中 n 将在循环中从 0...67 增长
    n = 0
    for b in range(2**8):
        if b not in bs:
            # 如果这个字节“很丑”，则将其映射到下一个可用的“漂亮”字符
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return d

def get_pairs(word):
    """
    以元组集合的形式返回可迭代对象 word 中所有连续元素的二元组（bigrams）。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:

    def __init__(self, encoder, bpe_merges):
        # 字节编码器/解码器
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        # bpe 标记（token）编码器/解码器
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        # bpe 合并列表，定义了 bpe “树”，由要合并为标记 ab 的元组 (a,b) 组成
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 用于预分词（pre-tokenization）的分隔模式
        # 应该添加 re.IGNORECASE，以便 BPE 合并可以发生在缩写词的大写版本上 <-- 原始 openai 注释
        """
        好的，那么这个正则表达式到底在寻找什么？
        python re 参考：https://docs.python.org/3/library/re.html
        - 垂直杠 | 是“或”，所以 re.findall 将在匹配时从左到右对文本进行分块
        - '\'s' 将拆分像 Andrej's 这样的东西 -> (Andrej, 's)
        - ' ?\p{L}': 可选空格后跟 1 个或多个“字母（letter）”类别的 unicode 码位
        - ' ?\p{N}': 可选空格后跟 1 个或多个“数字（number）”类别的 unicode 码位
        - ' ?[^\s\p{L}\p{N}]+': 可选空格，然后是 1 个或多个既不是空白、字母也不是数字的东西
        - '\s+(?!\S)': 1 个或多个空白字符（例如空格或制表符等），除非它们后面跟着非空白字符
                       所以这将消耗序列中的空白字符，但不包括该序列中的最后一个空白字符。
                       最后一个空白字符随后有机会匹配早期模式中可选的 ' ?'。
        - '\s+': 1 个或多个空白字符，可能旨在捕获字符串末尾的完整尾随空白序列
        所以总而言之（TLDR）：
        - 我们对一些常见的撇号结构（'s, 't, 're, ...）进行特殊处理，并将它们制作为单独的标记
        - 然后我们将字符串分隔为连续的块：1) 字母，2) 数字，3) 非字母数字，4) 空白符
        """
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token):
        """
        此函数使用 self.bpe_ranks 迭代地合并树上所有可能的 bpe 标记。
        token 是单个“单词”的字符串（经过正则分词后）
        且经过字节编码，例如 'Ġthere'。
        """
        # token 是单个“单词”的字符串，经过字节编码，例如 'Ġthere'

        # 记忆化，为了效率
        if token in self.cache:
            return self.cache[token]

        word = tuple(token) # 组成标记的单个字符，放在元组中
        pairs = get_pairs(word) # 获取所有二元组

        if not pairs:
            return token

        while True:

            # 找到下一个可以合并的最低排名的二元组
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break # 没有更多符合条件的二元组可以合并
            first, second = bigram

            # 我们现在将当前单词列表中所有出现的 (first, second) 替换为
            # 一个合并后的标记 first_second，放在输出列表 new_word 中
            new_word = []
            i = 0
            while i < len(word):

                # 在当前单词序列中找到下一次出现的 first
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # 如果这次出现后面也跟着 second，则将它们合并为一个
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # 所有出现的 (first, second) 都已合并为 first_second
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # 将所有单词连接成一个字符串，并使用 ' ' 作为分隔符。注意，
        # 到目前为止所有字符都经过了字节编码，保证了数据中不会使用 ' '，
        # 且它是一个“特殊”的分隔符字符。
        word = ' '.join(word)

        # 缓存结果并返回
        self.cache[token] = word
        return word

    def encode(self, text):
        """ 输入字符串，输出整数列表 """
        bpe_idx = []
        # 将输入文本预分词为字符串标记（大致说是单词）
        tokens = re.findall(self.pat, text)
        # 将每个标记处理为 BPE 整数
        for token in tokens:
            # 将标记编码为字节 (b'') 对象
            token_bytes = token.encode('utf-8')
            # 将所有字节转换为它们的 unicode 字符串表示并展平
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            # 根据 self.bpe_ranks 执行所有适用的 bpe 合并
            token_merged = self.bpe(token_translated).split(' ')
            # 将所有 bpe 标记转换为整数
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            # 扩展我们所有输出整数的运行列表
            bpe_idx.extend(token_ix)
        return bpe_idx

    def encode_and_show_work(self, text):
        """ 调试函数，与 encode 相同，但返回所有中间工作过程 """
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append({
                'token': token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': token_merged,
                'token_ix': token_ix,
            })
        out = {
            'bpe_idx': bpe_idx, # 实际的输出序列
            'tokens': tokens, # 预分词的结果
            'parts': parts, # 每个标记部分的中间过程
        }
        return out

    def decode(self, bpe_idx):
        """ 输入整数列表，输出字符串 """
        # 反向映射整数以获取标记
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        # 反转字节编码器，例如恢复 'Ġ' -> ' '，并获取字节
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        # 恢复完整的 utf-8 字符串
        text = tokens_bytes.decode('utf-8', errors='replace')
        return text

def get_file(local_file, remote_file):
    """ 如果有必要，将 remote_file 下载到 local_file """
    if not os.path.isfile(local_file):
        print(f"正在将 {remote_file} 下载到 {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content)

def get_encoder():
    """
    返回 GPT BPE 编码器/解码器实例，并处理“数据库”文件的缓存。
    """
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', 'mingpt')
    os.makedirs(cache_dir, exist_ok=True)

    # 加载 encoder.json，其中包含标记到 bpe 索引的原始映射
    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)
    assert len(encoder) == 50257 # 256 个单独的字节标记，50,000 个合并标记，以及 1 个特殊的 <|endoftext|> 标记

    # 加载 vocab.bpe，其中包含 bpe 合并，即 bpe 树结构
    # 形式为元组 (a, b)，表示 (a, b) 将合并为一个标记 ab
    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    # 轻微后处理：去掉第一行的版本号，最后一行是空白
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    assert len(bpe_merges) == 50000 # 50,000 个合并标记

    # 构造 Encoder 对象并返回
    enc = Encoder(encoder, bpe_merges)
    return enc

# -----------------------------------------------------------------------------

class BPETokenizer:
    """ 封装上述 Encoder 的 PyTorch 感知类 """

    def __init__(self):
        self.encoder = get_encoder()

    def __call__(self, text, return_tensors='pt'):
        # 仅限 PyTorch；放在这里是因为我们想匹配 huggingface/transformers 接口
        assert return_tensors == 'pt'
        # 目前仅支持单字符串输入，未来可能支持字符串列表
        assert isinstance(text, str)
        # 编码并创建一个大小为 1 的“批次维度（batch dimension）”
        idx = [self.encoder.encode(text)]
        # 包装到 PyTorch 张量中
        out = torch.tensor(idx, dtype=torch.long)
        return out

    def decode(self, idx):
        # 目前确保是一个简单的 1D 张量
        assert idx.ndim == 1
        # 将索引解码为文本
        text = self.encoder.decode(idx.tolist())
        return text


if __name__ == '__main__':

    # 这是一个编码示例
    text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
    e = get_encoder()
    r = e.encode_and_show_work(text)

    print("原始文本为：")
    print(text)
    print("首先文本被预分词，拆分成块，结果是：")
    print(r['tokens'])
    # ['Hello', '!!', ' I', "'m", ' Andrej', ' Karpathy', '.', ' It', "'s", ' 2022', '.', ' w', '00', 't', ' :', 'D', ' 🤗']
    print("然后我们迭代每个块并依次处理它们...")
    for part in r['parts']:
        print(part)
    # {'token': 'Hello', 'token_bytes': b'Hello', 'token_translated': 'Hello', 'token_merged': ['Hello'], 'token_ix': [15496]}
    # {'token': '!!', 'token_bytes': b'!!', 'token_translated': '!!', 'token_merged': ['!!'], 'token_ix': [3228]}
    # {'token': ' I', 'token_bytes': b' I', 'token_translated': 'ĠI', 'token_merged': ['ĠI'], 'token_ix': [314]}
    # {'token': "'m", 'token_bytes': b"'m", 'token_translated': "'m", 'token_merged': ["'m"], 'token_ix': [1101]}
    # {'token': ' Andrej', 'token_bytes': b' Andrej', 'token_translated': 'ĠAndrej', 'token_merged': ['ĠAndre', 'j'], 'token_ix': [10948, 73]}
    # {'token': ' Karpathy', 'token_bytes': b' Karpathy', 'token_translated': 'ĠKarpathy', 'token_merged': ['ĠK', 'arp', 'athy'], 'token_ix': [509, 5117, 10036]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' It', 'token_bytes': b' It', 'token_translated': 'ĠIt', 'token_merged': ['ĠIt'], 'token_ix': [632]}
    # {'token': "'s", 'token_bytes': b"'s", 'token_translated': "'s", 'token_merged': ["'s"], 'token_ix': [338]}
    # {'token': ' 2022', 'token_bytes': b' 2022', 'token_translated': 'Ġ2022', 'token_merged': ['Ġ2022'], 'token_ix': [33160]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' w', 'token_bytes': b' w', 'token_translated': 'Ġw', 'token_merged': ['Ġw'], 'token_ix': [266]}
    # {'token': '00', 'token_bytes': b'00', 'token_translated': '00', 'token_merged': ['00'], 'token_ix': [405]}
    # {'token': 't', 'token_bytes': b't', 'token_translated': 't', 'token_merged': ['t'], 'token_ix': [83]}
    # {'token': ' :', 'token_bytes': b' :', 'token_translated': 'Ġ:', 'token_merged': ['Ġ:'], 'token_ix': [1058]}
    # {'token': 'D', 'token_bytes': b'D', 'token_translated': 'D', 'token_merged': ['D'], 'token_ix': [35]}
    # {'token': ' 🤗', 'token_bytes': b' \xf0\x9f\xa4\x97', 'token_translated': 'ĠðŁ¤Ĺ', 'token_merged': ['ĠðŁ', '¤', 'Ĺ'], 'token_ix': [12520, 97, 245]}
    # (关于这些中间过程是什么，请参考 Encoder.encode 中的代码)
    print("最终结果是连接并展平所有的 token_ix：")
    print(r['bpe_idx'])
    # [15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83, 1058, 35, 12520, 97, 245]
    # 这随后将成为 Transformer 的整数输入序列
    print("准备好输入 Transformer 了！")
