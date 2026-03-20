# chargpt

chargpt 训练一个字符级语言模型。

我们支持三种设置：1 个便捷设置和 2 个具有学术文献结果的“基准”设置：

- 用户指定的 `input.txt` 文件，我们在其上训练语言模型（LM）（例如，从[这里](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)获取 tiny-shakespear（1.1MB 数据））
- TODO [text8](http://mattmahoney.net/dc/textdata.html)：同样源自维基百科文本，但删除了所有 XML，并转换为仅包含 26 个小写英文字母加空格的格式
- TODO [enwik8](http://prize.hutter1.net) 基准（“Hutter Prize”），维基百科 XML 转储的前 1 亿字节，包含 205 个唯一标记
