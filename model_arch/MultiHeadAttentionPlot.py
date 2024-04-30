from graphviz import Digraph

dot = Digraph(comment='Multi-Head Attention')

# 添加輸入節點
dot.node('input', 'Input: Q, K, V\n(batch_size, seq_len, embed_size)', shape='box')

# 添加線性變換層節點
dot.node('q_proj', 'Linear(Q)', shape='box')
dot.node('k_proj', 'Linear(K)', shape='box')
dot.node('v_proj', 'Linear(V)', shape='box')

# 添加維度調整節點
dot.node('q_split', 'Rearrange(Q)\n(batch_size, seq_len, num_heads, head_dim)', shape='box')
dot.node('k_split', 'Rearrange(K)\n(batch_size, seq_len, num_heads, head_dim)', shape='box')
dot.node('v_split', 'Rearrange(V)\n(batch_size, seq_len, num_heads, head_dim)', shape='box')

# 添加注意力計算節點
dot.node('attn_weights', 'Attention Weights\nsoftmax(QK^T / sqrt(head_dim))', shape='box')
dot.node('head_outputs', 'Head Outputs\nattention_weights * V', shape='box')

# 添加拼接和線性變換節點
dot.node('concat', 'Concatenate Head Outputs\n(batch_size, seq_len, embed_size)', shape='box')
dot.node('output', 'Output\nLinear(concat)', shape='box')

# 添加邊
dot.edge('input', 'q_proj', label='Q')
dot.edge('input', 'k_proj', label='K')
dot.edge('input', 'v_proj', label='V')
dot.edge('q_proj', 'q_split')
dot.edge('k_proj', 'k_split')
dot.edge('v_proj', 'v_split')
dot.edge('q_split', 'attn_weights', label='Q')
dot.edge('k_split', 'attn_weights', label='K^T')
dot.edge('attn_weights', 'head_outputs')
dot.edge('v_split', 'head_outputs', label='V')
dot.edge('head_outputs', 'concat')
dot.edge('concat', 'output')

dot.render('Multi_Head_Attention', view=True)
