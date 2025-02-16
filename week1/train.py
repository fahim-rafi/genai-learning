import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (MiniGPT, TrainingConfig,
                          get_shakespeare_data, ShakespeareDataset)


def setup_training():
    config = TrainingConfig()
    text = get_shakespeare_data()
    dataset = ShakespeareDataset(text)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True
    )

    model = MiniGPT(
        vocab_size=dataset.tokenizer.vocab_size
    ).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    return model, optimizer, dataloader, dataset.tokenizer


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    print(f"Model is on: {next(model.parameters()).device}")

    if torch.cuda.is_available():
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = y.view(B * T)

        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
            # if torch.cuda.is_available():
            #     print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    return total_loss / len(dataloader)


def save_model(model, tokenizer, filepath='model_checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'tokenizer_chars': tokenizer.chars,
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def train_model(model, optimizer, dataloader, config):
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    try:
        for epoch in range(config.epochs):
            avg_loss = train_epoch(model, dataloader, optimizer, config.device)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

            # Save if best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, tokenizer, 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("No improvement for 3 epochs. Stopping training.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    print(f"Final loss: {best_loss:.4f}")
    return best_loss


if __name__ == "__main__":
    config = TrainingConfig()
    model, optimizer, dataloader, tokenizer = setup_training()

    train_model(model, optimizer, dataloader, config)
