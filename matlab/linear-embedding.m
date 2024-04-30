% 創建有向圖
digraph = digraph();

% 添加輸入節點
digraph = addnode(digraph, 'Input', 'Input (b, n, input_channels)');

% 添加線性層節點
digraph = addnode(digraph, 'Linear', 'nn.Linear(input_channels, output_channels)');
digraph = addedge(digraph, 'Input', 'Linear');

% 添加 LayerNorm 節點
digraph = addnode(digraph, 'LayerNorm', 'nn.LayerNorm(output_channels)');
digraph = addedge(digraph, 'Linear', 'LayerNorm');

% 添加 GELU 激活函數節點
digraph = addnode(digraph, 'GELU', 'nn.GELU()');
digraph = addedge(digraph, 'LayerNorm', 'GELU');

% 添加 Embedded 輸出節點
digraph = addnode(digraph, 'Embedded', 'Embedded (b, n, output_channels)');
digraph = addedge(digraph, 'GELU', 'Embedded');

% 添加 cls_token 節點
digraph = addnode(digraph, 'cls_token', 'cls_token (1, output_channels)');

% 添加 Concatenate 節點
digraph = addnode(digraph, 'Concatenate', 'Concatenate');
digraph = addedge(digraph, 'Embedded', 'Concatenate');
digraph = addedge(digraph, 'cls_token', 'Concatenate');

% 添加最終輸出節點
digraph = addnode(digraph, 'Output', 'Output (b, n+1, output_channels)');
digraph = addedge(digraph, 'Concatenate', 'Output');

% 設置節點形狀為方形
digraph.Nodes.Shape = 'box';

% 繪製圖形
plot(digraph);
