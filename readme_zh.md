

`train_cost_compute`  


1. mem的构成 

模型权重 + 梯度 + 优化器状态 + 中间激活值 

优化器状态包括： momentum 和 variance， 分别是梯度的一阶和二阶动量，要求fp32存储。 

全量训练的模型，梯度大小和模型权重大小相同，优化器大小是fp32的权重的2倍。

对于 微调的模型， 例如lora，其本身不占用太多内存，一般在rank=10以内的时候，lora占的内存不会超过百M。lora在模型中一般为fp32权重

本工程是如何分析mem的？
- 模型权重大小：传入的graph json文件需要包括所有的权重大小（本工程不计算权重大小，虽然也可以做到）
- 梯度大小： 通过分析所有需要求梯度的module（目前只支持linear），计算得出所有的梯度
- 激活值大小: 从所有梯度节点开始遍历整张图（往前遍历），得到所有与梯度相关的节点（这里称为激活节点）。理论上来说所有激活节点的输出都有可能是激活值。但实际上，激活节点的输出是否是激活值取决于：
    - 该节点是否会有隐藏的tensor，如dropout 会有一个隐藏的mask tensor， norm类的节点会有mean和std
    - 该节点在计算梯度时，是否会用到原来的output，如softmax的梯度是`grad_c * c * (1 - c), c = F.softmax(a, dim=1)`
  目前工程没有完全遵循这一准则，需要修复 

项目还会分析：


2. 时间构成






3. 如何生成表格 





4. todo



