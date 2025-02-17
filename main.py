import torch
import torch.optim as optim
from dataload_process import get_dataloader
from encoder_decoder import TransformerEncoderDecoder
import wandb

# Hyperparameters
config = {
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4
}

# Initialize WandB
wandb.init(project="mnist-transformer", config=config)

# DataLoader
train_loader = get_dataloader(config["batch_size"])

# Model, optimizer, and loss function
model = TransformerEncoderDecoder(**config).to(device)
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(config["num_epochs"]):
    model.train()
    for batch in train_loader:
        src = batch["embeddings"].to(device)
        tgt = batch["caption_input"].to(device)
        tgt_labels = batch["caption_label"].to(device)

        # Forward pass
        output = model(src, tgt)
        loss = criterion(output.view(-1, output.size(-1)), tgt_labels.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log to WandB
        wandb.log({"train_loss": loss.item()})

# Inference function
def infer(model, src):
    model.eval()
    with torch.no_grad():
        tgt = torch.zeros(1, 1, dtype=torch.long).to(device)  # Start with <s> token
        for _ in range(10):  # Max caption length
            output = model(src, tgt)
            next_token = output.argmax(dim=-1)[-1].item()
            tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)
            if next_token == sp.piece_to_id("</s>"):  # End token
                break
        return sp.decode_ids(tgt.squeeze().tolist())
    
    