from graphviz import Digraph

dot = Digraph(comment='ECGformer Architecture')

# 添加節點
dot.node('input', 'Input Signal\n(batch_size, signal_length, input_channels)', shape='box')
dot.node('embedding', 'Linear Embedding\nCLS Token + Positional Encoding', shape='box')
dot.node('encoder', 'Transformer Encoder Layers', shape='box')
dot.node('classifier', 'Classifier\nAverage Pooling + Linear + LayerNorm + Linear', shape='box')
dot.node('output', 'Output Classes\n(batch_size, num_classes)', shape='box')

# 添加邊
dot.edge('input', 'embedding', label='(batch_size, signal_length+1, embed_size)')
dot.edge('embedding', 'encoder', label='(batch_size, signal_length+1, embed_size)')
dot.edge('encoder', 'classifier', label='(batch_size, signal_length+1, embed_size)')
dot.edge('classifier', 'output')

# 添加 Transformer 編碼器層的詳細內容
with dot.subgraph(name='cluster_encoder') as encoder:
    encoder.attr(label='Transformer Encoder Layer', style='dashed')
    encoder.node('mha', 'Multi-Head Attention', shape='box')
    encoder.node('add1', 'Add & Norm', shape='box')
    encoder.node('mlp', 'MLP', shape='box')
    encoder.node('add2', 'Add & Norm', shape='box')

    encoder.edge('mha', 'add1')
    encoder.edge('add1', 'mlp')
    encoder.edge('mlp', 'add2')
    encoder.edge('add2', 'mha', style='dashed', label='Repeat N times')

dot.render('ECGformer_architecture', view=True)

