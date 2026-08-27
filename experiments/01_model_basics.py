import torch
from transformer_lens import HookedTransformer


def main():
    # use apple's gpu through metal when available
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using device: {device}")

    # load gpt-2 small
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device,
    )

    # simple prompt to check that everything works (note that output is not paris...)
    prompt = "The Eiffel Tower is located in"
    
    # convert text into tokens
    tokens = model.to_tokens(prompt)
    
    # look up learned embedding vector for each input token
    embeddings = model.embed(tokens)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"First token embedding:")
    print(embeddings[0, 0])
    
    # decode tokens back into text
    token_strings = [model.tokenizer.decode(token_id) for token_id in tokens[0]]

    for i, (token_id, token_string) in enumerate(zip(tokens[0], token_strings)):
        print(f"{i}: {token_id.item()} -> {token_string!r}")

    print(f"Tokens shape: {tokens.shape}")
    print(f"Tokens: {tokens}")

    # run tokens through full transformer to obtain logits
    logits = model(tokens)
    

    print(f"Logits shape: {logits.shape}")

    # select logits at final position, i.e. representing the model's predictions for the next token
    next_token_logits = logits[0, -1]

    print(f"Next-token logits shape: {next_token_logits.shape}")

    # select token with highest logit
    next_token = next_token_logits.argmax()
    
    # convert token back into text
    next_word = model.tokenizer.decode(next_token)

    print(f"Prompt: {prompt}")
    print(f"Predicted next token: {next_word}")


if __name__ == "__main__":
    main()