import streamlit as st
from model import *
import re
from datetime import datetime
from config import *

_, c = st.columns([5, 1])
link = '[See the code](https://github.com/lkleinbrodt/writer/blob/main/writer/model.py)'
c.markdown(link, unsafe_allow_html=True)

if 'output' not in st.session_state:
    st.session_state['output'] = ''

if 'context' not in st.session_state:
    st.session_state['context'] = ''

if 'model_dict' not in st.session_state:
    st.session_state['model_dict'] = {}


def change_author():
    author = st.session_state['author']
    log(f'changing author to: {author}')
    writer = st.session_state['writer']
    st.session_state['output'] = ''
    st.session_state['context'] = ''

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

writer = st.session_state['model_dict'][author]
st.session_state['writer'] = writer

generate = st.checkbox('Write!')

if st.session_state['context'] == '':
    st.session_state['context'] = torch.tensor(writer.encode(AUTHOR_DICT[author]['starter']), device = DEVICE).reshape(1,-1)

output_box = st.empty()
MAX_NEW_TOKENS = 5 #streamlit cloud is able to handle 5 pretty seamlessly

#TODO: tolkien does long lines, need a way to break that.
# text supports \n but does not wrap text. markdown and write wrap text but dont support \n
if author == 'JRR Tolkien':
    display = lambda x: output_box.write(x)
else:
    display = lambda x: output_box.text(x)

while generate:
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

display(st.session_state['output'])