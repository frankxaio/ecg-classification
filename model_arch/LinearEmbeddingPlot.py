from graphviz import Digraph

# 創建有向圖
dot = Digraph(comment='LinearEmbedding Architecture')

# 添加輸入節點
dot.node('input', 'Input (b, n, input_channels)', shape='box')

# 添加線性層節點
dot.node('linear', 'nn.Linear(input_channels, output_channels)', shape='box')
dot.edge('input', 'linear')

# 添加 LayerNorm 節點
dot.node('layernorm', 'nn.LayerNorm(output_channels)', shape='box')
dot.edge('linear', 'layernorm')

# 添加 GELU 激活函數節點
dot.node('gelu', 'nn.GELU()', shape='box')
dot.edge('layernorm', 'gelu')

# 添加 Embedded 輸出節點
dot.node('embedded', 'Embedded (b, n, output_channels)', shape='box')
dot.edge('gelu', 'embedded')

# 添加 cls_token 節點
dot.node('cls_token', 'cls_token (1, output_channels)', shape='box')

# 添加 Concatenate 節點
dot.node('concat', 'Concatenate', shape='box')
dot.edge('embedded', 'concat')
dot.edge('cls_token', 'concat')

# 添加最終輸出節點
dot.node('output', 'Output (b, n+1, output_channels)', shape='box')
dot.edge('concat', 'output')

# 渲染並顯示圖形
dot.render('LinearEmbedding_architecture', view=True)

