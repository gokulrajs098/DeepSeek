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