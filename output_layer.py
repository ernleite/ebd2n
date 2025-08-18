class OutputLayer:
    def __init__(self, input_size, num_classes, rank, master_rank=0, bias=True):
        """
        Final classification/output layer.
        """
        self.rank = rank
        self.master_rank = master_rank
        self.input_size = input_size
        self.num_classes = num_classes
        self.weight = torch.randn(num_classes, input_size) * 0.01
        self.bias = torch.zeros(num_classes) if bias else None

    def forward(self, microbatch_id, x_in):
        """
        Forward pass:
          z = W*x + b
          probs = softmax(z)
        """
        z = torch.matmul(x_in, self.weight.T)
        if self.bias is not None:
            z += self.bias

        probs = torch.softmax(z, dim=1)

        # Send predictions back to master
        try:
            microbatch_size_tensor = torch.tensor([x_in.size(0)], dtype=torch.long)
            microbatch_id_tensor = torch.tensor([microbatch_id], dtype=torch.long)

            dist.send(microbatch_size_tensor, dst=self.master_rank)
            dist.send(microbatch_id_tensor, dst=self.master_rank)
            dist.send(probs.flatten(), dst=self.master_rank)

            print(f"[OutputLayer Rank {self.rank}] Sent predictions for microbatch {microbatch_id} to Master")

        except Exception as e:
            print(f"[OutputLayer Rank {self.rank}] Send error: {e}")

        return probs
