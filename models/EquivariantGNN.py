from torch import nn
from torch.nn.functional import sigmoid
import dgl
from dgl.nn.pytorch import EGNNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from models.Transformer_Encoder import EncoderLayer



class EGTF(nn.Module):
    def __init__(
        self, # EGCL params
        hidden_channels, out_channels, num_egcl,
        node_emb=32, coords_emb=16, edge_emb=16,
        act_fn=nn.SiLU(), residual=True, attention=True,
        normalize=False, max_atom_type=100, cutoff=5.0,
        max_num_neighbors=32, static_coord=True, freeze_egcl=True,
        # Transformer-Encoder params
        d_model=256, num_encoder=1, num_heads=8,
        num_ffn=256, act_fn_ecd=nn.SiLU(), dropout_r=0.1,
        # Energy Head params
        num_neurons=512):

        super().__init__()
        # self.hidden_channels = hidden_channels
        self.n_layers = num_egcl
        # self.max_atom_type = max_atom_type
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        
        # Create embeddings of dimension (hidden_channels, ) for each atom type
        self.node_embedding = nn.Linear(9, node_emb) # nn.Embedding
        self.coords_embedding = nn.Linear(3, coords_emb)
        self.edge_embedding = nn.Linear(3, edge_emb) # nn.Embedding


        # Add EGNN Conv layer
        self.add_module(f"gcl_{0}", # in_size(node), hidden_size, out_size, edge_feat_size
                            EGNNConv(node_emb, hidden_channels, hidden_channels*2, edge_emb))
        self.add_module("bn0", nn.BatchNorm1d(hidden_channels*2))
        self.add_module("x_bn0", nn.BatchNorm1d(edge_emb))

        for i in range(1, num_egcl-1):
            self.add_module(f"gcl_{i}", # in_size(node), hidden_size, out_size, edge_feat_size
                            EGNNConv(hidden_channels*2, hidden_channels*2, hidden_channels*2, edge_emb))
            self.add_module(f"bn{i}", nn.BatchNorm1d(hidden_channels*2))
            self.add_module(f"x_bn{i}", nn.BatchNorm1d(edge_emb))


        self.add_module(f"gcl_{num_egcl-1}", # in_size(node), hidden_size, out_size, edge_feat_size
                            EGNNConv(hidden_channels*2, hidden_channels*2, out_channels, edge_emb))
        self.add_module(f"bn{num_egcl-1}", nn.BatchNorm1d(out_channels))
        self.add_module(f"x_bn{num_egcl-1}", nn.BatchNorm1d(edge_emb))

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads,
                                            num_ffn, dropout_r, act_fn_ecd) 
                                            for _ in range(num_encoder)])

        # Energy head
        self.energy_fc = nn.Sequential(
            nn.Linear(d_model, num_neurons),
            act_fn_ecd,
            nn.Linear(num_neurons, 1)
        )

    def forward(self, node_feats, coords_feats, batch, edge_index=None, edge_feats=None):
        h = self.node_embedding(node_feats)
        x = self.coords_embedding(coords_feats)
        e = self.edge_embedding(edge_feats)

        g = dgl.graph((edge_index[0], edge_index[1]))
        # EGC layers
        for i in range(0, self.n_layers):
            h, x = self._modules["gcl_%d" % i](g, h, x, e) # dgl graph, node_f, coord_f, edge_f
            h = self._modules["bn%d" % i](h)
            x = self._modules["x_bn%d" % i](x)
        # Encoder layers
        for layer in self.encoder_layers:
            h = layer(h)
        # # Energy Head
        h = h.squeeze(0)  # Assuming the batch dimension is at dim 0
        h = global_mean_pool(h, batch) # global_add_pool
        
        out = self.energy_fc(h)

        return sigmoid(out)
