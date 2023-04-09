from model import *
import io
import sentencepiece as spm

torch.manual_seed(94903)

CORPUS_PATH = './data/clean_shakespeare.txt'
MODEL_PATH = '/Users/landon/Projects/writer/data/writer_token.tar'

MAX_ITERS = 100_000
BATCH_SIZE = 64
LEARNING_RATE = 3E-4
EVAL_INTERVAL = 500
EVAL_ITERS = 20

###
assert '.tar' in MODEL_PATH

with open(CORPUS_PATH, 'r') as f:
    text = f.read()

print('Corpus Length: ', '{:,}'.format(len(text)))

spm_model = io.BytesIO()

with open(CORPUS_PATH, 'r') as f:
    spm.SentencePieceTrainer.train(sentence_iterator = f, model_writer = spm_model)

with open('spm.model', 'wb') as f:
    f.write(spm_model.getvalue()) #why is this getvalue but on my work mac it's get_value?

sp = spm.SentencePieceProcessor(model_proto = spm_model.getvalue())

encoded_text = sp.encode(text)
tokens = sorted(list(set(encoded_text)))
vocab_size = sp.vocab_size()
# vocab_size = len(tokens)
print('Vocab Size: ', '{:,}'.format(vocab_size))

encode = sp.encode
decode = sp.decode

data = torch.tensor(encode(text), dtype = torch.long)

train_val_split = .9
n = int(train_val_split * len(data))

train_data = data[:n]
val_data = data[n:]

model_params = TOKEN_MODEL_PARAMS
model_params['vocab_size'] = vocab_size
model_params['encode'] = encode
model_params['decode'] = decode

model = Writer(**model_params)
model.to(DEVICE)

print('Model has ', sum(p.numel() for p in model.parameters())/1e6, ' million parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

early_stopping = EarlyStopper(MODEL_PATH, patience = 1000, min_delta = .01)

for iter in range(MAX_ITERS):
    if (iter % EVAL_INTERVAL == 0) or (iter == MAX_ITERS - 1):
        losses = estimate_loss(model, EVAL_INTERVAL, train_data, val_data, BATCH_SIZE)
        log(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if early_stopping.early_stop(losses['val'], model):
            log('stopping early!')
            break

    x,y = get_batch(train_data, model.block_size, BATCH_SIZE)

    logits, loss = model(x,y)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

save_model_state(model, MODEL_PATH.replace('.tar', '_trained.tar'))