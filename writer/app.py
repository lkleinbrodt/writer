import streamlit as st
from model import *
import re
from datetime import datetime
from config import *
from random import randint, sample

_, c = st.columns([5, 1])
link = '[See the code](https://github.com/lkleinbrodt/writer/blob/main/writer/model.py)'
c.markdown(link, unsafe_allow_html=True)

if 'output' not in st.session_state:
    st.session_state['output'] = ''

if 'context' not in st.session_state:
    st.session_state['context'] = ''

if 'model_dict' not in st.session_state:
    st.session_state['model_dict'] = {}

if 'text_dict' not in st.session_state:
    st.session_state['text_dict'] = {}

if 'starting_index_dict' not in st.session_state:
    st.session_state['starting_index_dict'] = {}

if 'generating' not in st.session_state:
    st.session_state['generating'] = False

if 'starting_text' not in st.session_state:
    st.session_state['starting_text'] = ''

if 'first_run' not in st.session_state:
    st.session_state['first_run'] = True


def change_author():
    author = st.session_state['author']
    log(f'changing author to: {author}')
    st.session_state['output'] = ''
    st.session_state['context'] = ''
    st.session_state['starting_text'] = ''
    st.session_state['first_run'] = True

author = st.selectbox(
    'Choose your author:',
    key = 'author',
    options = AUTHOR_DICT.keys(),
    on_change=change_author
)

if author not in st.session_state['model_dict']:
    model = init_from_s3(AUTHOR_DICT[author]['s3_model_path'])
    model.eval()
    st.session_state['model_dict'][author] = model

    text = load_text_from_s3(AUTHOR_DICT[author]['text'])
    
    valid_starting_indices = []
    if author in ['William Shakespeare', 'Robert Frost']:
        sep = ' '
    else:
        sep = '\n'
    
    valid_starting_indices = [i for i,x in enumerate(text) if x == sep]
    assert len(valid_starting_indices) > 0
    
    st.session_state['text_dict'][author] = text
    st.session_state['starting_index_dict'][author] = valid_starting_indices

writer = st.session_state['model_dict'][author]
text = st.session_state['text_dict'][author]

st.session_state['writer'] = writer
st.session_state['text'] = text

if st.session_state['starting_text'] == '':
    # starter_text = AUTHOR_DICT[author]['starter']
    starting_index = sample(st.session_state['starting_index_dict'][author], 1)[0] + 1
    starting_text = ' '.join(text[starting_index:starting_index+100].split()[:-1])
    st.session_state['starting_text'] = starting_text
    st.session_state['output'] = starting_text
    st.session_state['context'] = torch.tensor(writer.encode(starting_text), device = DEVICE).reshape(1,-1)

def set_context():
    starting_text = st.session_state['starting_text']
    st.session_state['context'] = torch.tensor(writer.encode(starting_text), device = DEVICE).reshape(1,-1)
    st.session_state['output'] = starting_text

starting_text = st.text_area(
    'Enter a prompt:',
    # value = st.session_state['starting_text'],
    key = 'starting_text',
    on_change = set_context
)


def toggle_generate():
    st.session_state['generating'] = not st.session_state['generating']
    st.session_state['first_run'] = False

if not st.session_state['generating']:
    label = 'Start Writing!'
else:
    label = 'Stop Writing!'

toggle = st.button(label, on_click = toggle_generate)

output_box = st.empty()
MAX_NEW_TOKENS = 5 #streamlit cloud is able to handle 5 pretty seamlessly

#TODO: tolkien does long lines, need a way to break that.
# text supports \n but does not wrap text. markdown and write wrap text but dont support \n
if author == 'JRR Tolkien':
    display = lambda x: output_box.write(x)
else:
    display = lambda x: output_box.text(x)

while st.session_state['generating']:
    output = st.session_state['output']
    context = st.session_state['context']
    output_box.empty()
    display(output)
    # new_context = writer.generate(context, max_new_tokens=10)[0]
    # new_output = writer.decode(new_context.tolist())
    
    context = writer.generate(context.reshape(1,-1), max_new_tokens=MAX_NEW_TOKENS)[0]#.reshape(1,-1)
    new_output = writer.decode(context.tolist()) #better than just assigning output? this allows us to crop output and context differently?
    output += new_output[-MAX_NEW_TOKENS:]
    # output += new_output
    # new_context = new_context.reshape(1,-1)
    # context = torch.cat((context, new_context), dim = 1).reshape(1,-1)
    # print('writing', datetime.now() - stime)
    # n_lines = re.split('\n ', string_to_split)
    if len(output) > 750:
        output = output[-750:]
    if len(context) > writer.block_size:
        context = context[-writer.block_size:]
    st.session_state['output'] = output
    st.session_state['context'] = context
    # print(new_output)

if not st.session_state['first_run']:
    display(st.session_state['output'])