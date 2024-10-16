print("WARNING: You are running this unofficial E2/F5 TTS demo locally, it may not be as up-to-date as the hosted version (https://huggingface.co/spaces/mrfakename/E2-F5-TTS)")

import os
import re
import torch
import torchaudio
import gradio as gr
import numpy as np
import tempfile
from einops import rearrange
from ema_pytorch import EMA
from vocos import Vocos
from pydub import AudioSegment, silence
from model import CFM, UNetT, DiT, MMDiT
from cached_path import cached_path
from model.utils import (
    get_tokenizer, 
    convert_char_to_pinyin, 
    save_spectrogram,
)
from transformers import pipeline
import librosa
import soundfile as sf
from txtsplit import txtsplit

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device=device,
)

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# --------------------- Settings -------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = 'euler'
sway_sampling_coef = -1.0
speed = 1.0
# fix_duration = 27  # None or float (duration in seconds)
fix_duration = None

def load_model(repo_name, exp_name, model_cls, model_cfg, ckpt_step):
    checkpoint = torch.load(str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt")), map_location=device)
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=model_cls(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    ema_model = EMA(model, include_online_model=False).to(device)
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    ema_model.copy_params_from_ema_to_model()

    return model

# load models
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)

F5TTS_ema_model = load_model("F5-TTS", "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000)
E2TTS_ema_model = load_model("E2-TTS", "E2TTS_Base", UNetT, E2TTS_model_cfg, 1200000)

def infer(ref_audio_orig, ref_text, gen_text, exp_name, remove_silence, progress = gr.Progress()):
    print(gen_text)
    gr.Info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)
        # remove long silence in reference audio
        non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave
        # Convert to mono
        aseg = aseg.set_channels(1)
        audio_duration = len(aseg)
        if audio_duration > 15000:
            gr.Warning("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name
    if exp_name == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif exp_name == "E2-TTS":
        ema_model = E2TTS_ema_model
    
    if not ref_text.strip():
        gr.Info("No reference text provided, transcribing reference audio...")
        ref_text = outputs = pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )['text'].strip()
        gr.Info("Finished transcription")
    else:
        gr.Info("Using custom reference text...")
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text) / (audio.shape[-1] / sr) * (30 - audio.shape[-1] / sr))
    # Audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)
    # Chunk
    chunks = txtsplit(gen_text, 0.7*max_chars, 0.9*max_chars) # 100 chars preferred, 150 max
    results = []
    generated_mel_specs = []
    for chunk in progress.tqdm(chunks):
        # Prepare the text
        text_list = [ref_text + chunk]
        final_text_list = convert_char_to_pinyin(text_list)
    
        # Calculate duration
        ref_audio_len = audio.shape[-1] // hop_length
        # if fix_duration is not None:
        #     duration = int(fix_duration * target_sample_rate / hop_length)
        # else:
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
        chunk = len(chunk.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * chunk / speed)
    
        # inference
        gr.Info(f"Generating audio using {exp_name}")
        with torch.inference_mode():
            generated, _ = ema_model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
    
        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, '1 n d -> 1 d n')
        gr.Info("Running vocoder")
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms
    
        # wav -> numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()
        results.append(generated_wave)
    generated_wave = np.concatenate(results)
    if remove_silence:
        gr.Info("Removing audio silences... This may take a moment")
        # non_silent_intervals = librosa.effects.split(generated_wave, top_db=30)
        # non_silent_wave = np.array([])
        # for interval in non_silent_intervals:
        #     start, end = interval
        #     non_silent_wave = np.concatenate([non_silent_wave, generated_wave[start:end]])
        # generated_wave = non_silent_wave
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, generated_wave, target_sample_rate)
            aseg = AudioSegment.from_file(f.name)
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave
            aseg.export(f.name, format="wav")
            generated_wave, _ = torchaudio.load(f.name)
        generated_wave = generated_wave.squeeze().cpu().numpy()

    # spectogram
    # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
    #     spectrogram_path = tmp_spectrogram.name
    #     save_spectrogram(generated_mel_spec[0].cpu().numpy(), spectrogram_path)

    return (target_sample_rate, generated_wave)

with gr.Blocks() as app:
    gr.Markdown("""
# E2/F5 TTS

This is an unofficial E2/F5 TTS demo. This demo supports the following TTS models:

* [E2-TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)
* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)

This demo is based on the [F5-TTS](https://github.com/SWivid/F5-TTS) codebase, which is based on an [unofficial E2-TTS implementation](https://github.com/lucidrains/e2-tts-pytorch).

The checkpoints support English and Chinese.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 15s, and shortening your prompt. If you're still running into issues, please open a [community Discussion](https://huggingface.co/spaces/mrfakename/E2-F5-TTS/discussions).

Long-form/batched inference + speech editing is coming soon!

**NOTE: Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<15s). Ensure the audio is fully uploaded before generating.**
""")

    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(label="Text to Generate (longer text will use chunking)", lines=4)
    model_choice = gr.Radio(choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS")
    generate_btn = gr.Button("Synthesize", variant="primary")
    with gr.Accordion("Advanced Settings", open=False):
        ref_text_input = gr.Textbox(label="Reference Text", info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.", lines=2)
        remove_silence = gr.Checkbox(label="Remove Silences", info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.", value=True)
    
    audio_output = gr.Audio(label="Synthesized Audio")
    # spectrogram_output = gr.Image(label="Spectrogram")

    generate_btn.click(infer, inputs=[ref_audio_input, ref_text_input, gen_text_input, model_choice, remove_silence], outputs=[audio_output])
    gr.Markdown("Unofficial demo by [mrfakename](https://x.com/realmrfakename)")
    

app.queue().launch()