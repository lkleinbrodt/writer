import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from time import localtime, strftime

def log(s):
    t = strftime('%H:%M:%S', localtime())
    print(t,s)

TOKEN_MODEL_PARAMS = {
    'block_size': 12,
    'n_embed': 8,
    'n_head': 2,
    'n_layer': 2,
    'dropout': .2
}

CHARACTER_MODEL_PARAMS = {
    'block_size': 24,
    'n_embed': 64,
    'n_head': 2,
    'n_layer': 2,
    'dropout': .2
}

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

print('PyTorch using: ', DEVICE)

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2,-1) * (k.shape[-1] ** -.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim = -1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v

        return out
    
class MultiHeadAttention(nn.Module):
    """apply multiple attentions in parallel and concatenate results"""

    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        out = self.dropout(out)

        return out

class FeedForward(nn.Module):
    """ a single linear layer followed by a non-linearity"""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4), #*4 comes from paper. Higher dim in inner layers of residual blocks
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication then computation"""

    def __init__(self, n_embed, n_head, block_size, dropout) -> None:
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed, block_size, dropout)
        self.feed_forward = FeedForward(n_embed, dropout)
        self.layer_norm = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.layer_norm(x))
        x = x + self.feed_forward(self.layer_norm2(x))

        return x

class Writer(nn.Module):

    def __init__(self, vocab_size, n_embed, n_layer, n_head, block_size, dropout, encoder, decoder) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout
        self.encoder = encoder
        self.decoder = decoder

        self.model_params = {
            'vocab_size': vocab_size,
            'n_embed': n_embed,
            'n_layer': n_layer,
            'n_head': n_head,
            'block_size': block_size,
            'dropout': dropout,
            'encoder': encoder,
            'decoder': decoder,
        }

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        #TODO: check this logic
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = .02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=.02)
        
    def forward(self, idx, targets = None):
        B, T = idx.shape

        #idx and targets are both (B,T) tensors of integers
        # each int grabs a row from our embedding table
        # predict what comes next from the current token
        # i'm token 5, what comes next

        token_embedding = self.token_embedding_table(idx) #(B,T,C) - (batch size, block_size, n_embed)
        position_embedding = self.position_embedding_table(torch.arange(T, device = DEVICE)) #(T,C)
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            #cropt to last block
            idx_cond = idx[:, -self.block_size:]
            #predict 
            logits, loss = self(idx_cond)
            #focus on last time step
            logits = logits[:, -1, :] # (B,C)
            #convert to probs
            probs = F.softmax(logits, dim = -1) # (B,C)
            #sample one
            idx_next = torch.multinomial(probs, num_samples = 1) #(B,1)
            #append to running seq
            idx = torch.cat((idx, idx_next), dim = 1)
            #TODO: to allow real time speech, adjust this for loop
        
        return idx

    def write(self, context = None, max_new_tokens = 100):
        self.eval()
        if context is None:
            context = torch.zeros((1,1), dtype = torch.long, device = DEVICE)
        else:
            context = torch.tensor(self.encoder(context), device = DEVICE).reshape(1,-1)
        
        tokens = self.generate(context, max_new_tokens=max_new_tokens)[0].tolist() #first batch
        self.train()
        return self.decoder(tokens)

class EarlyStopper:
    def __init__(self, path = None, patience = 1, min_delta = 0.0):
        self.path = path
        self.patience = patience
        self.min_delta = min_delta

        self.counter = 0
        self.min_validation_loss = np.inf
    
    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if self.path is not None:
                save_model_state(model, self.path)
                log('saved best model')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size, ))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(DEVICE), y.to(DEVICE)

    return x,y

def save_model_state(model: Writer, path: str):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': model.model_params
    }, path)

def init_from_path(path):
    model_state = torch.load(path)
    model = Writer(**model_state['model_params'])
    model.load_state_dict(model_state['model_state_dict'])
    model.to(DEVICE)
    return model

@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, batch_size):
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if split == 'train':
            data = train_data
        else:
            data = val_data
        
        for k in range(eval_iters):
            x, y = get_batch(data, model.block_size, batch_size)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out