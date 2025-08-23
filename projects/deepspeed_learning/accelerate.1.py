from accelerate import Accelerator, DeepSpeedPlugin
import torch
from torch.utils.data import TensorDataset, DataLoader

class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    input_dim = 10
    hidden_dim = 256
    output_dim = 2
    batch_size = 64
    data_size = 10000
    input_data = torch.randn(data_size, input_dim)
    labels = torch.randn(data_size, output_dim)

    data_set = TensorDataset(input_data, labels)
    dataloader = DataLoader(data_set, batch_size = batch_size)

    model = SimpleNet(input_dim, hidden_dim, output_dim)
    deepspeed = DeepSpeedPlugin(zero_stage = 2, gradient_clipping = 1.0)
    accelerator = Accelerator(deepspeed_plugin = deepspeed)
    optimizator = torch.optim.Adam(model.parameters(), lr=0.001)
    crition = torch.nn.MSELoss()

    model, optimizator, dataloader = accelerator.prepare(model, optimizator, dataloader)

    for epoch in range(1000):
        model.train()
        for batch in dataloader:
            inputs, labels = batch
            optimizator.zero_grad()
            outputs = model(inputs)
            loss = crition(outputs, labels)
            accelerator.backward(loss)
            optimizator.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    accelerator.save(model.state_dict(), "model.pth")