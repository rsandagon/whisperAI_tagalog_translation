import os
import openai
os.system("pip install git+https://github.com/openai/whisper.git")
import gradio as gr
import whisper

from share_btn import community_icon_html, loading_icon_html, share_js

model = whisper.load_model("medium")


#OpenAi call
# def gpt3(texts):
#     openai.api_key = os.environ["Secret"]
#     response = openai.Completion.create(
#       engine="text-davinci-002",
#       prompt= texts,
#           temperature=0,
#           max_tokens=750,
#           top_p=1,
#           frequency_penalty=0.0,
#           presence_penalty=0.0,
#           stop = (";", "/*", "</code>")
#     )
#     x = response.choices[0].text 
    
#     return x
        
def inference(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    _, probs = model.detect_language(mel)
    
    options = whisper.DecodingOptions(language="en",fp16 = False)
    result = whisper.decode(model, mel, options)
    
    options = dict(language="Tagalog", beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    translation = model.transcribe(audio, **translate_options)["text"]

    
    print(result.text)
    return translation, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)




css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
     
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .prompt h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; margin-top: 1.5rem !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
        }
        #share-btn * {
            all: unset;
        }
"""

block = gr.Blocks(css=css)



with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Translating Tagalog Voices to English via Whisper
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification. This demo cuts audio after around 30 secs.
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                audio = gr.Audio(
                    label="Input Audio",
                    show_label=False,
                    source="microphone",
                    type="filepath"
                )

                btn = gr.Button("Translate")
        text = gr.Textbox(show_label=False, elem_id="result-textarea")
        with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
        


        
        btn.click(inference, inputs=[audio], outputs=[text, community_icon, loading_icon, share_button],api_name="translate-audio")
        share_button.click(None, [], [], _js=share_js)
 
        gr.HTML('''
        <div class="footer">
                    <p>Model by <a href="https://github.com/openai/whisper" style="text-decoration: underline;" target="_blank">OpenAI</a> - Gradio Demo by 🤗 Hugging Face
                    </p>
        </div>
        ''')

block.launch()