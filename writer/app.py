import streamlit as st
from model import *
import re
from datetime import datetime

#TODO: load appropriate model

if 'output' not in st.session_state:
    st.session_state['output'] = ''

_, c = st.columns([5, 1])

link = '[See the code](https://github.com/lkleinbrodt/writer/blob/main/writer/model.py)'
c.markdown(link, unsafe_allow_html=True)

author_dict = {
    'William Shakespeare': {
        'text': './data/clean_shakespeare.txt', 
        'model': './data/shakespeare.tar', 
        'starter': 'By William Shakespeare'
    },
    'Robert Frost': {
        'text': './data/clean_frost.txt', 
        'model': './data/frost.tar',
        'starter': '\n\n'
    },
    'JRR Tolkien': {
        'text': './data/clean_tolkein.txt',
        'model': './data/tolkien.tar',
        'starter': '>\n',
    }
}

def change_author():
    st.session_state['output'] = ''

author = st.selectbox(
    'Choose your author:',
    options = author_dict.keys(),
    on_change=change_author
)

@st.cache_resource(show_spinner=False)
def load_model(author):

    model = init_from_path(author_dict[author]['model'])
    model.eval()
    # with open(author_info['text'], 'r') as f:
    #     txt = f.read()
    return model
writer = load_model(author)

generate = st.checkbox('Write!')

output_box = st.empty()
context = torch.tensor(writer.encode(author_dict[author]['starter']), device = DEVICE).reshape(1,-1)
MAX_NEW_TOKENS = 1

#TODO: tolkien does long lines, need a way to break that.
# text supports \n but does not wrap text. markdown and write wrap text but dont support \n
if author == 'JRR Tolkien':
    display = lambda x: output_box.write(x)
else:
    display = lambda x: output_box.text(x)

while generate:
    output = st.session_state['output']
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

    output_box.empty()
    st.session_state['output'] = output
    # print(new_output)

display(st.session_state['output'])