from model import *

torch.manual_seed(94903)


CORPUS_PATH = './data/clean_shakespeare.txt'
MODEL_PATH = './data/writer_character.pt'

MAX_ITERS = 200
BATCH_SIZE = 64
LEARNING_RATE = 3E-4
EVAL_INTERVAL = 100
EVAL_ITERS = 10

###
# assert '.tar' in MODEL_PATH

with open(CORPUS_PATH, 'r') as f:
    text = f.read()

print('Corpus Length: ', '{:,}'.format(len(text)))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Vocab Size: ', '{:,}'.format(vocab_size))
print(chars)

s_to_i = {char: i for i,char in enumerate(chars)}
i_to_s = {i: char for i,char in enumerate(chars)}

def encode(s):
    return [s_to_i[c] for c in s]

def decode(l):
    return ''.join([i_to_s[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)

train_val_split = .9
n = int(train_val_split * len(data))

train_data = data[:n]
val_data = data[n:]

model_params = CHARACTER_MODEL_PARAMS
model_params['vocab_size'] = vocab_size
model_params['encoder'] = encode
model_params['decoder'] = decode

model = Writer(**model_params)
model.to(DEVICE)

print('Model has ', sum(p.numel() for p in model.parameters())/1e6, ' million parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

early_stopping = EarlyStopper(MODEL_PATH, patience = 1000, min_delta = .01)

# history = pd.DataFrame
for iter in range(MAX_ITERS):
    if (iter % EVAL_INTERVAL == 0) or (iter == MAX_ITERS - 1):
        losses = estimate_loss(model, EVAL_INTERVAL, train_data, val_data, BATCH_SIZE)
        log(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if early_stopping.early_stop(losses['val'], model):
            log('stopping early!')
            break
        # log(model.write(max_new_tokens=50))

    x,y = get_batch(train_data, model.block_size, BATCH_SIZE)

    logits, loss = model(x,y)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

save_model_state(model, MODEL_PATH.replace('.pt', '_trained.pt'))