from graphviz import Digraph

dot = Digraph(comment='MLP')

# 添加輸入節點
dot.node('input', 'Input\n(batch_size, seq_len, input_channels)', shape='box')

# 添加第一個線性變換層節點
dot.node('linear1', 'Linear\n(input_channels, input_channels * expansion)', shape='box')

# 添加 GELU 激活函數節點
dot.node('gelu', 'GELU', shape='box')

# 添加第二個線性變換層節點
dot.node('linear2', 'Linear\n(input_channels * expansion, input_channels)', shape='box')

# 添加輸出節點
dot.node('output', 'Output\n(batch_size, seq_len, input_channels)', shape='box')

# 添加邊
dot.edge('input', 'linear1')
dot.edge('linear1', 'gelu')
dot.edge('gelu', 'linear2')
dot.edge('linear2', 'output')

dot.render('MLP', view=True)

