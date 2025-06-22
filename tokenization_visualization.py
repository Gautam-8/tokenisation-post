import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

texts = [
    "The cat sat on the mat because it was tired.",
    "supercalifragilisticexpialidocious",
    "🚀 AI is amazing! 🎉",
    "¡Hola! ¿Cómo estás?"
]

tokenizers = {
    'GPT-2 (BPE)': AutoTokenizer.from_pretrained("gpt2"),
    'BERT (WordPiece)': AutoTokenizer.from_pretrained("bert-base-uncased"),
    'T5 (SentencePiece)': AutoTokenizer.from_pretrained("t5-small")
}

token_counts = {}
for text in texts:
    token_counts[text] = {}
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        token_counts[text][name] = len(tokens)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

x = np.arange(len(texts))
width = 0.25

for i, (name, color) in enumerate([('GPT-2 (BPE)', '#FF6B6B'), 
                                  ('BERT (WordPiece)', '#4ECDC4'), 
                                  ('T5 (SentencePiece)', '#45B7D1')]):
    counts = [token_counts[text][name] for text in texts]
    ax1.bar(x + i*width, counts, width, label=name, color=color, alpha=0.8)

ax1.set_xlabel('Sample Texts')
ax1.set_ylabel('Token Count')
ax1.set_title('Tokenization Comparison Across Different Models', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels(['Simple English', 'Long Word', 'Emojis', 'Spanish'], rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

simple_text = "The cat sat on the mat because it was tired."
tokens_by_model = {}

for name, tokenizer in tokenizers.items():
    tokens = tokenizer.tokenize(simple_text)
    tokens_by_model[name] = tokens

models = list(tokens_by_model.keys())
token_lists = list(tokens_by_model.values())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (model, tokens, color) in enumerate(zip(models, token_lists, colors)):
    y_pos = i * 2
    ax2.barh(y_pos, len(tokens), color=color, alpha=0.8, height=1.5)
    ax2.text(len(tokens) + 0.5, y_pos, f'{len(tokens)} tokens', 
             va='center', fontweight='bold')

ax2.set_xlabel('Number of Tokens')
ax2.set_title('Token Breakdown for: "The cat sat on the mat because it was tired."', 
              fontsize=12, fontweight='bold')
ax2.set_yticks([0, 2, 4])
ax2.set_yticklabels(models)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tokenization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'tokenization_comparison.png'") 