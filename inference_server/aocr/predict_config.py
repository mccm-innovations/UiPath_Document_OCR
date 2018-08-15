class config:
    attn_num_layers=2
    attn_num_hidden=128
    batch_size=1
    use_gru=False
    gpu_id=1
    valid_target_len = float('inf')
    img_width_range=(12, 700)
    img_height=32
    word_len=50
    target_vocab_size=2 * 26 + 10 + 3 + 1
    target_embedding_size=10
    visualize = True