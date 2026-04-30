# 模型架构介绍
对比上一篇我介绍的是encoder-only architecture, 这次我将介绍decoder-only architecture. 基本上来说decoder是用来generate content，所以大规模用在生成文字，图片，音频这些方面，我们熟知的chatgpt就是decoder-only architecture.
同样这边讲解我通过和transformer对比的方式呈现。
Llama Architecture：
Embedding
 ↓
26 × LlamaDecoderLayer
 ↓
Final RMSNorm
 ↓
lm_head
 ↓
预测下一个 token

transformer decoder:
Target Embedding + Positional Encoding
 ↓
Masked Multi-Head Self-Attention
 ↓
Add & Norm
 ↓
Encoder-Decoder Attention
 ↓
Add & Norm
 ↓
Feed Forward Network
 ↓
Add & Norm
 ↓
Linear + Softmax
```
model architecture:
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 3200, padding_idx=0)
    (layers): ModuleList(
      (0-25): 26 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=3200, out_features=3200, bias=False)
          (k_proj): Linear(in_features=3200, out_features=3200, bias=False)
          (v_proj): Linear(in_features=3200, out_features=3200, bias=False)
          (o_proj): Linear(in_features=3200, out_features=3200, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=3200, out_features=8640, bias=False)
          (down_proj): Linear(in_features=8640, out_features=3200, bias=False)
          (up_proj): Linear(in_features=3200, out_features=8640, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=3200, out_features=32000, bias=False)
)

```
## embedding layer
这里将32000个token映射到3200维空间中，即每个token用一个3200维的tensor表示。注意这里不同之处是，他在这里没有加上位置编码。
## attention layer
这两者都是masked self-attention也就是说，在生成第 `t` 个 token 时，只能看到前面的 token，不能看到未来 token。
这里的q_proj和transformer中的q是同一个东西，这里的o_proj是指output projection。作用是多个 head 拼接后，只是简单地放在一起而output projection 负责重新混合这些 head 的信息。（这个在transformer和bert中都有只是不同形式存在，bert中是BertSelfOutput.dense）
```
head_i = Attention(Q_i, K_i, V_i)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
```

注意：rotary_emb是其中一个最大的不同。
这里LLaMA 的 Q/K 会经过 RoPE（旋转位置编码）将位置信息加入进来，这和transformer不同，后者是通过embedding layer中通过直接和位置信息相加的方式加入的。至于为什么只对QK做RoPE，因为这里算的是attention，即当前token对其他位置的token有多少注意力，而V编码的内容是这个token真正代表的内容。我们希望token能通过位置信息的辅助找到它真正应该注意的内容，而不是改变这个内容本身。
### RoPE具体实现
**RoPE 旋转变换**不是抽象说法，它真的就是把 Q 和 K 向量的维度两两分组，然后对每一组做二维旋转。

假设某个 head 的维度是 `64`，也就是：

```
q = [q0, q1, q2, q3, ..., q62, q63]k = [k0, k1, k2, k3, ..., k62, k63]
```

RoPE 会把它拆成 32 对：

```
(q0, q1), (q2, q3), (q4, q5), ..., (q62, q63)
```

然后每一对二维向量按照当前位置旋转一个角度。
## MLP layer
这里对标的是transformer中的feed forward network。
经典 Transformer FFN 通常是：

```
Linear(d_model → d_ff)ReLULinear(d_ff → d_model)
```

例如：

```
512 → 2048 → 512
```

而你的 LLaMA MLP 是（这里依然可以观察到先升维再降维的操作类似于d_model和d_ff之间的转换）：

```
gate_proj: 3200 → 8640
up_proj:   3200 → 8640
down_proj: 8640 → 3200
act_fn: SiLU
```

它不是普通的：

```
Linear → Activation → Linear
```

而是 gated 结构，大致形式是：

```
MLP(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```
这种 gated MLP 比普通 FFN 表达能力更强，因为它可以学习“哪些特征应该通过，哪些特征应该被抑制”。这里用到的思想和LSTM中的门控思想很像，这里通过训练gate_proj,up_proj，down_proj 让模型训练哪些信息应该通过，以及具体传递了哪些信息。实现效果如下：
你可以把 `up_proj(x)` 看成模型提取出来的一批候选特征：

```
value = up_proj(x)
```

例如：

```
value_1：语法信息value_2：实体信息value_3：情感信息value_4：长距离依赖信息...
```

然后 `gate_proj(x)` 生成一个“开关”：

```
gate = SiLU(gate_proj(x))
```

它决定哪些特征应该被放大，哪些应该被压低。

例如：

```
gate_1 = 0.9   语法信息通过gate_2 = 0.1   实体信息被压低gate_3 = 0.8   情感信息通过gate_4 = 0.0   长距离依赖暂时不用
```

最后：

```
hidden = gate * value
```

所以每个特征通道都会被动态调节。
## output layer
bert中是将一整句话用一个tensor来表示，而这里则是将每个位置的 hidden state 转成词表上的 logits。
```
hidden state: [B, L, 3200]
 ↓
lm_head
 ↓
logits: [B, L, 32000]
```
然后模型可以对最后一个位置的 logits 做 softmax，得到下一个 token 的概率分布。
## normalization
顺带介绍一下normalization的作用：它让每一层输入的数值尺度更稳定，避免随着层数加深，hidden states 的分布不断漂移，从而让训练更稳定、梯度更容易传播。所以一般这个都是加在一个功能层的输入前和输出后（相当于下一个功能层的输入前）。
这里容易误解的是它们**不是在 forward 里直接连续执行**的。模型打印出来只是列出了这个 `LlamaDecoderLayer` 里包含的模块。
事实上的数据流向应该是这样的：
```
hidden_states
     │
     ├─────────────── residual ───────────────┐
     ↓                                        │
input_layernorm                               │
     ↓                                        │
self-attention                                │
     ↓                                        │
attention_output                              │
     ↓                                        │
hidden_states = residual + attention_output ←┘
     │
     ├─────────────── residual ───────────────┐
     ↓                                        │
post_attention_layernorm                      │
     ↓                                        │
MLP                                           │
     ↓                                        │
mlp_output                                    │
     ↓                                        │
hidden_states = residual + mlp_output ←──────┘
```
另一点需要注意的是pre_normalization残差链接中的信息不通过normalization直接传过去的,这样残差可以保留原始信息。
```
output = x + attention(norm(x))
```
但是post_normalization则是残差路径也被 norm 改了一遍，深层模型训练时梯度传播可能更困难。
```
output = Norm(x + attention(x))
```