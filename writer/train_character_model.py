from model import *
import torch

torch.manual_seed(94903)


CORPUS_PATH = './data/clean_frost.txt'
MODEL_PATH = './data/frost.tar'
PRETRAINED_PATH = './data/writer_character.tar'

MAX_ITERS = 10_000
BATCH_SIZE = 64
LEARNING_RATE = 3E-4
EVAL_INTERVAL = 500
EVAL_ITERS = 15

###
assert '.tar' in MODEL_PATH

def main():

    with open(CORPUS_PATH, 'r') as f:
        text = f.read()

    print('Corpus Length: ', '{:,}'.format(len(text)))

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    if PRETRAINED_PATH:
        #TODO: huge todo. better pretraining
        model = init_from_path(PRETRAINED_PATH)
        vocabulary = set(model.vocabulary)
        if vocabulary != set(chars):
            log('cleaning text')
            #TODO: better vocab reconciliation
            text = ''.join([x for x in text if x in vocabulary])
        model.train()
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
    else:

        model_params = CHARACTER_MODEL_PARAMS
        model_params['vocab_size'] = vocab_size
        model_params['vocabulary'] = chars

        model = Writer(**model_params)
        model.to(DEVICE)
    
    print('Vocab Size: ', '{:,}'.format(vocab_size))
    print(chars)
    
    data = torch.tensor(model.encode(text), dtype = torch.long)

    train_val_split = .9
    n = int(train_val_split * len(data))

    train_data = data[:n]
    val_data = data[n:]

    print('Model has ', sum(p.numel() for p in model.parameters())/1e6, ' million parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    early_stopping = EarlyStopper(MODEL_PATH, patience = 20, min_delta = .01)

    # history = pd.DataFrame
    for iter in range(MAX_ITERS):
        if (iter % EVAL_INTERVAL == 0) or (iter == MAX_ITERS - 1):
            stime = time()
            losses = estimate_loss(model, EVAL_INTERVAL, train_data, val_data, BATCH_SIZE)
            elapsed = time() - stime
            log(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} (eval took {int(elapsed)} secs)")
            if early_stopping.early_stop(losses['val'], model):
                log('stopping early!')
                break
            # log(model.write(max_new_tokens=50))

        x,y = get_batch(train_data, model.block_size, BATCH_SIZE)

        logits, loss = model(x,y)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

    save_model_state(model, MODEL_PATH.replace('.tar', '_trained.tar'))

if __name__ == '__main__':
    log('---START---')
    main()
    log('---END---')