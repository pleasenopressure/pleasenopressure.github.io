# 环境配置中常见的错误，gpu isn't available
这个基本上每次训练都会碰到这类问题，我就基于我的显卡举例，我的显卡是5070ti。其他同学如果使用的是50系，这里就和我保持一致就行。
## 确定gpu和cuda版本
我们从错误的底层开始一步步递归上去：
首先容易错的是显卡型号和cuda驱动的队友
```
nvidia-smi  #这个命令可以查gpu型号和cuda支持的最高版本
```
## 确定pytorch版本
一般gpu架构有一个计算能力对照表

|GPU 系列|架构名称|计算能力|最低 PyTorch 支持|
|---|---|---|---|
|RTX 20xx|Turing|sm_75|PyTorch 1.x+|
|RTX 30xx|Ampere|sm_86|PyTorch 1.9+|
|RTX 40xx|Ada Lovelace|sm_89|PyTorch 2.0+|
|RTX 50xx|**Blackwell**|**sm_120**|**PyTorch 2.8+ (cu128)**|
这里是最容易出错的地方，常见的就是pytorch过低。以下是我常碰到的错误。

| 报错信息                                                                   | 含义                         | 解决方案                             |
| ---------------------------------------------------------------------- | -------------------------- | -------------------------------- |
| `sm_120 is not compatible with the current PyTorch installation`       | PyTorch 编译时没有包含你 GPU 的计算能力 | 升级 PyTorch 到支持该架构的版本             |
| `CUDA error: no kernel image is available for execution on the device` | 同上，在运行时才暴露                 | 同上                               |
| `torch.cuda.is_available()` 返回 `False`                                 | CUDA 完全不可用                 | 检查是否安装了 GPU 版 PyTorch（而不是 CPU 版） |
这边顺便说明如何下载指定版本的torch。
去 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
然后按照你的配置搞，这里下载命令中--index-url很重要，下面会说到。
![[Pasted image 20260429164000.png]]
## pytorch下载中遇到的坑
这个经常发生在一个项目中提供了requirement.txt，然后其中一项就写pytorch>=... ,一旦你直接pip install -r requirement.txt.你就会发现它默认给你下载的是CPU版本或者低CUDA版。所以一定要指定--index-url.
```
# 示例：安装支持 CUDA 12.8 的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
## 上层库兼容性问题
配置过环境的朋友都知道，这玩意就像是雪崩，你底层升级了一个依赖，上层的一堆都会完蛋，出现修了一个bug，出来10个。
我这次实际碰到的几个问题
1. numpy and pandas 不兼容
```
   ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```
原因：`pandas` 的核心计算部分是用 C/Cython 编写的，在安装（编译）时会依赖 `numpy` 的底层 C 头文件。这些头文件里定义了 `numpy.dtype` 这类核心数据结构在内存中的大小（字节数）。
当 `numpy` 从 1.x 升级到 2.0 时，底层 C API 发生了破坏性变更——`numpy.dtype` 的内存布局从 88 字节变成了 96 字节。但你环境中的 `pandas` 是之前在 `numpy 1.x` 环境下编译安装的，它内部的 C 代码仍然按照旧的 88 字节大小来访问 `dtype` 对象。当运行时实际加载的是新版 `numpy 2.0`（96 字节），两边对不上，就会崩溃。

修复:升级pandas（要是这还报错，可能你的python版本有点老了）
```
pip install --upgrade #pip会自动安装当前python版本支持的最新版pandas
```
规律：当你看到size changed or binary incompatibility这类关键词，基本都是这个包的C扩展和依赖的库版本不一致，基本解法就是升级报错的包。
还有一个方法是可以看报错信息的release notes/changelog: 通常会注明"Added support for numpy 2.0"之类的信息，帮你快速定位最低兼容版本。
2. 升级一个包（多为pytorch）后缺少间接依赖
   解决：缺啥就下载啥。
## 针对反复配置同一环境依赖的通用解决方案
就是拍摄快照了
```
pip freeze > requirement.txt
```
等出问题了就rollback
```
pip install or requirement.txt
```
你可以在升级一个包钱看看他会影响到其他哪些包
```
# 查看一个包被哪些其他包依赖
pip show numpy    # 看 "Required-by" 字段
```
# tokenizer的使用
  函数的关键是看懂输入输出（inputs为例）
```
  tokenizer = transformers.AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
  model = transformers.BertModel.from_pretrained("google-bert/bert-base-uncased")
  inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
```
这里的inputs**本身并不是一个张量 (Tensor)**，而是一个 **类似字典（Dictionary-like）的数据结构**（在源码中它的类型叫做 `BatchEncoding`）。

当你调用 `tokenizer(..., return_tensors="pt")` 时，它之所以返回一个“字典”，是因为模型在处理自然语言时，不仅仅需要文字对应的 ID，还需要其他辅助信息。这个字典把所有需要传给模型的张量都打包在了一起。
```
inputs = 
{
    'input_ids': tensor([[ 101, 7592, 1010, 2026, 3899, 2003, 10146,  102]]),
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
}
```
可以发现这其实是个字典，其中
1. **`input_ids`**：这是最重要的核心数据。句子里的每一个词（或者子词）被映射成了词表里的数字 ID。例如 101 通常代表开头的 `[CLS]`，102 代表结尾的 `[SEP]`。
2. **`token_type_ids`**：用来告诉模型“这是哪一句话”。BERT 经常被用来处理句子对（比如问答任务），如果是第一句话全填 0，第二句话全填 1。这里只有一句话，所以全都是 0。
3. **`attention_mask`**：注意力掩码。很多时候我们送进模型的是一个批次（Batch）的好几句话，为了让它们长度对齐，不够长的句子会在末尾填充（Padding）一些 0（通常是 ID 0）。`attention_mask` 里填 1 代表这是真实的单词，填 0 代表这是无意义的 Padding，模型在计算时就会自动忽略这些 0。
- 顺带一提，后面会讲这个tokenize后的输出传给model，这里会使用python的解包语法，把字典里的键值对当做输入参数一次性传给model,model(**inputs)
# tensor是什么
刚开始学习tensor的同学容易把tensor，numpy，和普通的list当做是一个东西。事实上这几个有很大的不同。
- python中普通的list，底层是一个存储指针的动态数组，这使它能够同时存放多种不同的数据结构，但是内存上不连续。
- numpy中的ndarray是类似C语言中存放同构的数据，且内存空间连续，支持广播机制，但只能在cpu上运行。
- Tensor区别前两者最大的不同就是，他不是我们正常理解的数组，而是更像一个数据类，除了存储实际的矩阵数据，这个类内部还维护了非常多用于深度学习的核心属性和方法。

你观察到的 `grad_fn` 属性，正是 PyTorch 最核心的魔法——**自动求导（Autograd）机制**的关键所在。
![[Pasted image 20260427215429.png]]
具体来说，一个 Tensor 类除了存数据之外，至少还包含以下几类重要属性：

### 1. 计算图与自动求导属性

- **`grad_fn`**：顾名思义是 Gradient Function（梯度函数）。如果一个 Tensor 是通过某种数学运算产生的（比如加法、乘法或者神经网络的一层），`grad_fn` 就会记录**是谁（哪个函数）创造了它**。比如终端里打印的 `grad_fn=<AddmmBackward0>`，意思是这个张量是通过矩阵乘法加法（Linear 层）算出来的。这就相当于记录了整个“计算历史图”，当训练模型需要反向传播更新权重时，PyTorch 就能顺藤摸瓜，用链式法则自动计算导数。
- **`requires_grad`**：一个布尔值（True/False），告诉系统“在进行计算时，要不要跟踪这个张量的所有操作以便之后求导”。一般训练的时候开启，推理的时候关闭。
- **`grad`**：存放算出来的梯度值本身（也是一个 Tensor）。
![[Pasted image 20260429173700.png]]
### 2. 硬件与内存属性

- **`device`**：记录这个张量目前存在哪里的内存中（是电脑的主存 `cpu`，还是显卡的显存 `cuda:0`）。这在普通数组里是没有的概念。

### 3. 数据描述属性

- **`dtype`**：数据类型（比如 `torch.float32`, `torch.int64`）。
- **`shape` / `size()`**：张量的维度大小。
   
# 模型架构的解析以及与transformer的对比
```
   model architecture (BERT without heads):
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 x BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)
```
### embeding layer
```
# call the embedding layer manually so we can inspect the output tensor

# the embedding layer has dim of 768

batch_size, seq_length = input_shape

embedding_output = model.embeddings(

    input_ids=inputs["input_ids"],

    position_ids=None,

    token_type_ids=inputs["token_type_ids"],

    inputs_embeds=None,

    past_key_values_length=0,

)
```
这里通过embedding的输入和模型架构可以看出这一层的作用是把30552个词映射到768维的向量空间中，再加上位置编码和token_type_embedding区别不同句子的编码（传统的transformer架构中这里只有positional encoding +word embedding)。最后经过normalization和dropout输出。
注意这里的positional embedding和transformer中的positional embedding不同，这里将512个位置映射到512个768维向量中，这是一个学习训练的过程，也就是**可学习的位置编码**。而transformer中这里是用固定公式生成的。
### 核心：encoder
- 你可以看到 `(0-11): 12 x BertLayer`，说明有 **12 层**一模一样的处理层（标准的 BERT-base 结构）。在每一层里：
    - `(attention)`: **自注意力机制**（包含了 `query`, `key`, `value`）。这让模型在看当前单词时，能“注意”到句子中其他相关的单词，从而理解上下文。
    - `(intermediate)`: **前馈神经网络**，先将特征维度从 768 猛增到 3072，并使用 `GELU` 激活函数增加非线性表达能力。
    - `(output)`: 再次把特征从 3072 压回 768 维，加上 `LayerNorm` 输出。
#### bert和经典transformer架构中encoder进行对比
![[Pasted image 20260427221057.png]]
其实就是将经典transformer架构中encoder堆叠了12层。里面有些细节的地方有所区别例如
1. 激活函数不同。
2. 另外一些操作隐含在bert架构中（和transformer中一致的），例如residual connection(残差连接，图片中的ADD操作),即这里的attention output的部分其实是
```
hidden_states = LayerNorm(hidden_states + attention_output)
```
3. 以及在feed forward network部分会先升维再降维（对应d_model and d_ff)。
4. 最重要的不同是bert最后有个pooler layer，介绍这个层的功能前我要先介绍一下这个单encoder架构主要解决怎样的问题，这个架构是将大量的上下文通过encoder形成一个高层次的理解，基于这种能力，我们实际上可以用它做文本分类，句子匹配等任务（对比transformer这种enconder+decoder的架构其实就是先产生对上下文的总体理解，然后通过docoder来generate对应的文本）。言归正传，这种分类任务通常只需要一个输出向量（判断0,1这种就行了）。那我们看一下encoder network的输出是（batch size, seq_len,hidden_size=786),这说明输出给每个token都有一个向量表示。那如何转化过去呢？这里就用到常见的图像处理中的池化操作了，即pooling。
  
#### BERT 的 pooler 具体做了什么？

BERT 的 pooler 默认取第一个 token，也就是 `[CLS]` 的向量（没有为什么就是习惯默认，理论上来说取哪个都一样）：

```
last_hidden_state[:, 0, :]
```

它的 shape 是：

```
[batch_size, 768]
```

然后通过一层线性层和 `Tanh`：

```
pooled_output = tanh(W * h_cls + b)
```

对应你的结构：

```
pooler:  dense: Linear(768 → 768)  activation: Tanh()
```

所以它不是把所有 token 平均起来，而是：

```
取 [CLS] 向量 → Linear → Tanh → pooled_output
```

---

#### 4. 为什么 `[CLS]` 可以代表整句话？

关键在于 **self-attention**。

在 BERT 里面，每个 token 都可以 attend 到其他 token。`[CLS]` 也一样。

输入：

```
[CLS] I like machine learning [SEP]
```

在多层 self-attention 后，`[CLS]` 的向量不再只是 `[CLS]` 这个符号本身的信息，而是融合了整句话的信息。

例如：

```
h_cls = f([CLS], I, like, machine, learning, [SEP])
```

也就是说，经过 12 层 self-attention 后，`[CLS]` 可以吸收其他 token 的语义信息。

因此可以把 `[CLS]` 最后一层的 hidden state 当作整句表示。

---

#### 5. 那 pooler 为什么能起作用？

因为它做了一个“句向量变换”。

原始的 `[CLS]` hidden state 是：

```
h_cls
```

pooler 做的是：

```
pooled_output = tanh(W h_cls + b)
```

这相当于让模型学习一个变换，把 `[CLS]` 向量变成更适合句子级任务的表示。

举个直观例子：

```
h_cls 原始空间：可能混合了语法信息、词义信息、上下文信息、位置相关信息pooler 后：通过可训练的 Linear 层，把这些信息重新组合，变成更适合分类/句子关系判断的表示
```

`Tanh` 的作用是把输出压到：

```
[-1, 1]
```

范围内，使表示更稳定。
### multi head在bert中的体现
bert的架构这里没有显式的写出来但实际上是采用了多头注意力机制的。这个很重要对多头注意力机制理解不深的朋友可以通过这个理解（以下内容建立在对注意力机制计算有初步理解的基础上）。
![[Pasted image 20260429222626.png]]multi-head attention其实就是把原先一个大的token表示维度拆分成几个小的表示维度。
例如我这里multi head = 12,则768/12 = 64.即每一个head负责一个64维向量空间。
如图中所示，我介绍一下优势的前三点是如何实现的：
一个常见的误区就是以为多头只是单纯的把原先的768维向量拆开后QKV矩阵对应位置算attention，但是这里其实有两个误区。
1. 每个head的QKV矩阵都是在各自的向量空间（64维）训练得到的，天然会有不同角度的观察关系的能力
2. softmax机制就是把attention分数转化为概率分布或者说把原始相似度分数转换成“关注比例”，结合下面的attention公式   ![[Pasted image 20260429231938.png]]那么12个head每个head都会产生一个attention权重分布，如
```
like → [CLS] : 0.05
like → I     : 0.20
like → like  : 0.10
like → cats  : 0.60
like → [SEP] : 0.05
```
这比原先只有一个token attention权重分布更加不容易遗漏重要关系，泛化更好。

至于优势的第四点，这里也有个普遍的误区，就是所谓的并行性好对比的不是单头注意力机制，同样是矩阵并行运算，单头多头计算量基本没区别。这里实际对比的是其他架构例如RNN。