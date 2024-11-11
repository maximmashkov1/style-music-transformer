import gradio as gr
import mido
from pydub import AudioSegment
import tempfile
import os
from midi_utilities import create_midi_from_timings, get_midi_notes
from model import *
from tqdm import tqdm
from tokenizer import MusicalTokenizer
import torch
from sklearn.decomposition import PCA
import gc

if not os.path.exists("./temp"):
    os.makedirs("./temp")
if not os.path.exists("./checkpoint"):
    os.makedirs("./checkpoint")
if not os.path.exists("./generation_outputs"):
    os.makedirs("./generation_outputs") 
  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer=MusicalTokenizer(notes=128, 
                           time_shifts=125, 
                           longest_time_token=1000)

model_path = './checkpoint/'+os.listdir('./checkpoint')[0]
model=torch.load(model_path).to(DEVICE)
model.tokenizer=tokenizer
model.eval()
FILE = None

FILES = []
NAMES = []

START_IDX = 0
PRIMER_NOTES = 100
TEMP = 1.1
K=8
LENGTH=500

info_bar = None

LAST_GENERATED = None
MULTIPLE_STYLES = False

def plot_l2_distance_min():
    
    model.eval()
    if len(FILES) < 3:
        return None

    style_sequences, _ = get_selected_style_and_primer()
    
    x = model.compute_style_tokens(torch.stack(style_sequences).to(DEVICE)).flatten(start_dim=1).detach().to('cpu')
    x = x / torch.max(x)
    

    best_embedding=None
    distances = torch.cdist(x, x) 
    best_error=999

    for tries in range(10):
        embedding = nn.Parameter(torch.randn(x.size(0), 2))  
        optimizer = optim.SGD([embedding], lr=0.2) 
        
        for i in range(100):
            optimizer.zero_grad()
            distances_2d = torch.cdist(embedding, embedding)
            
            loss = torch.nn.functional.mse_loss(distances_2d, distances)
            loss.backward(retain_graph=True)
            if i == 99 and loss.item() < best_error:
                best_embedding=embedding.detach().clone()
            

            optimizer.step()
            for param_group in optimizer.param_groups:
                param_group['lr']*=0.97 

    x_2d = best_embedding.detach().cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c='yellow', edgecolor='k', s=100)
    for i, name in enumerate(NAMES):
        plt.text(x_2d[i, 0] -0.5, x_2d[i, 1] + 0.05, name, fontsize=11)

    plt.title("Pairwise L2-minimized 2D plot of style vectors")
    plt.grid(True)
    plt.savefig('./temp/l2_plot.png')
    plt.close('all')
    del x, x_2d, distances, distances_2d, style_sequences, loss 
    del embedding, optimizer 
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return './temp/l2_plot.png'



def get_selected_style_and_primer():
    def pad(sequence, target_length):
        x = torch.zeros(target_length, dtype=torch.long)
        x[:sequence.shape[0]] = sequence
        return x
    if MULTIPLE_STYLES:
        return [pad(style[:model.seq_length], model.seq_length) for style in FILES], None
    
    else:
        if FILE is None or START_IDX > FILE.shape[0]: return

        primer = None if PRIMER_NOTES==0 else FILE[START_IDX:START_IDX+PRIMER_NOTES]

        return pad(FILE[START_IDX:START_IDX+model.seq_length], model.seq_length), primer

def generate_midi(progress=gr.Progress()):
    progress(0, desc="Starting")
    print('Generating...')
    if MULTIPLE_STYLES:
        if len(FILES) == 0: return gr.Button.update(visible=False), gr.Text.update(f"No files loaded.", visible=True)
    else:
        if FILE is None: return gr.Button.update(visible=False), gr.Text.update(f"No file loaded.", visible=True)

    style, start_with = get_selected_style_and_primer()

    iter_ =  model.generate(style, start_with, temperature=TEMP, top_k=K, max_length=LENGTH)
    for _ in progress.tqdm(range(LENGTH+1), desc="Generating"):
        try:
            iter_.__next__()
        except Exception as e:
            print(e)
            pass
    out = model.generation_result
    
    existing_files = list(os.listdir("./generation_outputs"))
    global LAST_GENERATED
    LAST_GENERATED = fr"{os.path.dirname(os.path.abspath(__file__))}\generation_outputs\generated_{len(existing_files)}.mid"
    create_midi_from_timings([tokenizer.detokenize(out)], LAST_GENERATED)
    return gr.Button.update(visible=True), gr.Text.update(f"Generated sequence saved as {LAST_GENERATED}", visible=True)

def preview_midi():
    if FILE is None: return
    selected_primer = get_selected_style_and_primer()[1]
    create_midi_from_timings([tokenizer.detokenize(selected_primer)], f'{os.path.dirname(os.path.abspath(__file__))}/temp/temp.mid')
    
    os.startfile(f'{os.path.dirname(os.path.abspath(__file__))}/temp/temp.mid')

def play_midi():
    os.startfile(LAST_GENERATED)



def update_midi_file(midi_input):
    if MULTIPLE_STYLES:
        global FILES
        if midi_input is None: 
            return gr.File.update(None), gr.Label.update(visible=True), gr.Image.update()
        try:
            FILES.append(tokenizer.tokenize(get_midi_notes(midi_input.name)))
            NAMES.append(midi_input.name[midi_input.name.rfind('\\')+1:midi_input.name.rfind('.')])
        except Exception as e:
            print(e)
            return gr.File.update(None), gr.Label.update("Error occurred. Check console.", visible=True), gr.Image.update()
        return gr.File.update(None), gr.Label.update(f"Loaded {len(FILES)} styles", visible=True), gr.Image.update(plot_l2_distance_min())
    else:
        global FILE
        if midi_input is None: 
            FILE = None; 
            return gr.File.update(), gr.Label.update(visible=True), gr.Image.update()
        
        try:
            FILE = tokenizer.tokenize(get_midi_notes(midi_input.name))
        except Exception as e:
            print(e)
            return  gr.File.update(), gr.Label.update("Error occurred", visible=True), gr.Image.update()
        return  gr.File.update(), gr.Label.update(f"Loaded {FILE.shape[0]} tokens", visible=True), gr.Image.update()



def update_is_multiple(value):
    global MULTIPLE_STYLES
    global FILE
    global FILES
    global NAMES
    MULTIPLE_STYLES = value
    FILES.clear()
    NAMES.clear()
    FILE = None
    return gr.Number.update(visible=not value), gr.Number.update(visible=not value), gr.Button.update(visible=not value), gr.Label.update("Load a reference midi file" if not value else "Drop midi files one by one"), gr.File.update(None), gr.Button.update(visible=value), gr.Image.update(None, visible=value)


with gr.Blocks() as app:
    gr.Markdown("## Music transformer playground")

    with gr.Row():
        with gr.Column():
            is_multiple = gr.Checkbox(label='Blend multiple styles', scale=3)
            with gr.Group():
                
                files_loaded_info = gr.Label(value="Load a reference midi file",  label="Info")
                midi_input = gr.File(label="Upload a MIDI file")

                
                start_idx = gr.Number(label="Start from token", minimum=0, value=START_IDX)
                primer_notes = gr.Number(label="Primer tokens", minimum=0, value=PRIMER_NOTES)
                preview_btn = gr.Button("Play selected sequence", variant='secondary')
                clear_files_btn = gr.Button("Unload files", variant='secondary', visible=False)
                
        with gr.Column():
            temp_slider = gr.Slider(label="Temperature", minimum=0, maximum=3, step=0.05, value=TEMP)
            top_k_slider = gr.Slider(label="Top-k", minimum=1, maximum=128, step=1, value=K)
            length_slider = gr.Slider(label="Tokens to generate", minimum=1, maximum=10000, step=1, value=LENGTH)
            gr.Markdown("")
            generate_btn = gr.Button("Generate", variant='primary')
            play_btn = gr.Button("Play generated sequence", variant='primary', visible=False)

        with gr.Column(scale=1):
            similarity_plot = gr.Image(label="2D plot of style similarity (upload at least 3 files)", visible=False)
        

    info_bar = gr.Text(label='Output', value="\n")

    is_multiple.change(update_is_multiple, inputs=[is_multiple], outputs=[start_idx, primer_notes, preview_btn, files_loaded_info, midi_input, clear_files_btn, similarity_plot])
    midi_input.change(update_midi_file, inputs=[midi_input], outputs=[midi_input, files_loaded_info, similarity_plot])
    clear_files_btn.click(update_is_multiple, inputs=[is_multiple], outputs=[start_idx, primer_notes, preview_btn, files_loaded_info, midi_input, clear_files_btn, similarity_plot])
    generate_btn.click(generate_midi, outputs=[play_btn, info_bar], queue=True)
    preview_btn.click(preview_midi)
    play_btn.click(play_midi)
    
    start_idx.change(lambda value: globals().update(START_IDX=int(value)), inputs=start_idx)
    primer_notes.change(lambda value: globals().update(PRIMER_NOTES=int(value)), inputs=primer_notes)
    temp_slider.change(lambda value: globals().update(TEMP=value), inputs=temp_slider)
    top_k_slider.change(lambda value: globals().update(K=int(value)), inputs=top_k_slider)
    length_slider.change(lambda value: globals().update(LENGTH=int(value)), inputs=length_slider)

app.queue().launch()
