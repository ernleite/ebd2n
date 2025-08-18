import torch
import torch.distributed as dist

class WeightedLayer:
    def __init__(self, input_size, output_size, rank, next_rank, bias=True):
        """
        Weighted layer equivalent to a fully connected linear transformation.
        Distributed between ranks.
        """
        self.rank = rank
        self.next_rank = next_rank
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.randn(output_size, input_size) * 0.01
        self.bias = torch.zeros(output_size) if bias else None

    def forward(self, microbatch_id, x_split):
        """
        Forward pass of weighted layer:
          y = W*x + b
        """
        # x_split shape: (micro_batch_size, input_size_split)
        y = torch.matmul(x_split, self.weight.T)
        if self.bias is not None:
            y += self.bias

        # Send results to next layer (activation node or next WeightedLayer)
        try:
            microbatch_size_tensor = torch.tensor([x_split.size(0)], dtype=torch.long)
            microbatch_id_tensor = torch.tensor([microbatch_id], dtype=torch.long)

            dist.send(microbatch_size_tensor, dst=self.next_rank)
            dist.send(microbatch_id_tensor, dst=self.next_rank)
            dist.send(y.flatten(), dst=self.next_rank)

            print(f"[WeightedLayer Rank {self.rank}] Sent microbatch {microbatch_id} to rank {self.next_rank}")

        except Exception as e:
            print(f"[WeightedLayer Rank {self.rank}] Send error: {e}")

        return y
