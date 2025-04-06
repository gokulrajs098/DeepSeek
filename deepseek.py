import torch
import torch.nn as nn

class MLA(nn.Module):
    """
        MLA(Multi-Headed-Attention-Layer) Attributes:

        dim(int): dimensionality of input features,
        n_heads(int): Number of Attention heads,
        n_local_heads(int): number of local attention heads for distributes systems,
        q_lora_rank(int): Rank for low-rank query projection,
        kv_lora_rank(int): Rank for low-rank key/value projection,
        qk_nope_head_dim(int): Dimensionality of non-positional query/key projections,
        qk_rope_head_dim(int): Dimensionality of rotary-positional query/key projections,
        v_head_dim(int): Dimensionality of value projections,
        softmax_scale(float): Scaling factor for softmax in attention computation.
    """ 
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads*self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads*self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads*(self.qk_nope_head_dim+self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads*self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional(torch.Tensor)):
        """
        Forward  pass for Multi-headed Attention Layer(MLA).

        Args:
            x(torch.Tensor): Input tensor of shape(batch_size, seq_len, dim).
            start_pos(int): starting position in the sequence for caching.
            freqs_cis(torch.Tensor): PreComputed complex exponential values for rotary embedings.
            mask(Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input
        """

        bsz, seqlen = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b((self.q_norm(self.wq_a(x))))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsquueze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, (self.qk_nope_head_dim, self.v_head_dim), dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd, bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd, hdc->bshc", q_nope, wkv_b[:,:self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshd, btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])+
                      torch.einsum("bshr, btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim])
        x = self.wo(x.flatten)
class Gate:
    """
    Gating mechanisms for routing inputs in mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk(int): number of top experts activated for each input.
        n_groups(int): number of groups for routing.
        topk_groups(int): Number of groups to route inputs to.
        score_func(str): Scoring function('Softmax' or 'Sigmoid').
        route_scale(float): Scaling factor for routing weights.
        weight(torch.nn.Parameter): Learnable weights for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initialize the Gate module.

        Args:
            args(ModelArgs): Model argument containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Forward pass for the gating mechanism.

            Args:
                x (torch.Tensor): Input Tensor.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else: 
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)


class ColumnParallelLinear(Linear):
    """
        Linear Layer with column parallelism, splittting output features across distributed process

        Args:
            in_features(int): input features,
            out_features(int): output features,
            bias(bool): whether to include a bias term (defaults to false),
            data_type(optional): Data type for the layer (defaults to 'torch.bfloat16').
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world_size"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for column parallel linear layer.

            Args:
                x(torch.Tensor):Input tensor.

            Returns:
                torch.Tensor: Transformed tensor with column-parallel computation.        
        """

        y = linear(x, self.weight, self.bias)
        return y
    