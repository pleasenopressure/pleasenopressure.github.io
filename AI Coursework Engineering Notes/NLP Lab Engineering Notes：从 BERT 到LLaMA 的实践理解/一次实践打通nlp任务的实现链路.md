# 环境配置中常见的错误，gpu isn't available

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
这里通过embedding的输入和模型架构可以看出这一层的作用是把30552个词映射到768维的向量空间中，再加上位置编码和token_type_embedding区别不同句子的编码。最后经过normalization和dropout输出。
### 核心：encoder
- 你可以看到 `(0-11): 12 x BertLayer`，说明有 **12 层**一模一样的处理层（标准的 BERT-base 结构）。在每一层里：
    - `(attention)`: **自注意力机制**（包含了 `query`, `key`, `value`）。这让模型在看当前单词时，能“注意”到句子中其他相关的单词，从而理解上下文。
    - `(intermediate)`: **前馈神经网络**，先将特征维度从 768 猛增到 3072，并使用 `GELU` 激活函数增加非线性表达能力。
    - `(output)`: 再次把特征从 3072 压回 768 维，加上 `LayerNorm` 输出。
这个和经典transformer架构中encoder进行对比![[Pasted image 20260427221057.png]]