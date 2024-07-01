
** 注意：需要修改仓库中json文件路径才能跑通 ** 


```md
graph.json 是反向图 
sd_forward.json 是前向图 
tsbwg 是 反向svg 图
help.py 是反向图的遍历，需要加入 搬运量和计算量的计算 
help_forward.py 是前向图的遍历，有搬运量和计算量的计算
```
搬运量计算： 

1. 数据需要搬运到特定运算单元的内存中，这里为从`system->local`, 搬运量计算是 tensor 的大小，一般为输入的tensor大小。
2. 在特定运算单元运算完后，需要搬运会system 内存,会有一个 `local->system` 一般就是输出的tensor大小 
3. 考虑bais之类是否需要搬运 

计算量可以参考网上公式。 


