import torch
import torch.nn.functional as F
from transformers import CharacterTokenizer, MiniGPT


def load_model(filepath='best_model.pth', device='cuda'):
    checkpoint = torch.load(filepath, map_location=device)

    tokenizer = CharacterTokenizer("")
    tokenizer.chars = checkpoint['tokenizer_chars']
    tokenizer.vocab_size = len(tokenizer.chars)
    tokenizer.char_to_idx = {ch: i for i, ch in enumerate(tokenizer.chars)}
    tokenizer.idx_to_char = {i: ch for i, ch in enumerate(tokenizer.chars)}

    model = MiniGPT(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.5):
    model.eval()
    context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(next(model.parameters()).device)
    generated = list(context[0].cpu().numpy())
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            next_token = next_token.unsqueeze(0)
            context = torch.cat([context, next_token.squeeze(0)], dim=1)
            
            generated.append(next_token.squeeze().item())
    
    return tokenizer.decode(generated)


def interactive_generation():
    model, tokenizer = load_model()
    model.eval()

    while True:
        prompt = input("\nEnter prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break

        text = generate_text(model, tokenizer, prompt, temperature=0.5)
        print(f"\nGenerated Text:\n{text}")


if __name__ == "__main__":
    interactive_generation()
