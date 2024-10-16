# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path

import os
import re
import torch
import torchaudio
import numpy as np
import tempfile
from einops import rearrange
from ema_pytorch import EMA
from vocos import Vocos
from pydub import AudioSegment
from model import CFM, UNetT, DiT, MMDiT
from cached_path import cached_path
from model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
    save_spectrogram,
)
from transformers import pipeline
import librosa

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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


class Predictor(BasePredictor):
    def load_model(exp_name, model_cls, model_cfg, ckpt_step):
        checkpoint = torch.load(str(cached_path(f"hf://SWivid/F5-TTS/{exp_name}/model_{ckpt_step}.pt")), map_location=device)
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

        return ema_model, model
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        print("Loading Whisper model...")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device=device,
        )
        print("Loading F5-TTS model...")

        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        self.F5TTS_ema_model, self.F5TTS_base_model = self.load_model("F5TTS_Base", DiT, F5TTS_model_cfg, 1200000)


    def predict(
        self,
        gen_text: str = Input(description="Text to generate"),
        ref_audio_orig: Path = Input(description="Reference audio"),
        remove_silence: bool = Input(description="Remove silences", default=True),
    ) -> Path:
        """Run a single prediction on the model"""
        model_choice = "F5-TTS"
        print(gen_text)
        if len(gen_text) > 200:
            raise gr.Error("Please keep your text under 200 chars.")
        gr.Info("Converting audio...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            aseg = AudioSegment.from_file(ref_audio_orig)
            audio_duration = len(aseg)
            if audio_duration > 15000:
                gr.Warning("Audio is over 15s, clipping to only first 15s.")
                aseg = aseg[:15000]
            aseg.export(f.name, format="wav")
            ref_audio = f.name
        ema_model = self.F5TTS_ema_model
        base_model = self.F5TTS_base_model

        if not ref_text.strip():
            gr.Info("No reference text provided, transcribing reference audio...")
            ref_text = outputs = self.pipe(
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

        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(device)

        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        # Calculate duration
        ref_audio_len = audio.shape[-1] // hop_length
        # if fix_duration is not None:
        #     duration = int(fix_duration * target_sample_rate / hop_length)
        # else:
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text) + len(re.findall(zh_pause_punc, ref_text))
        gen_text_len = len(gen_text) + len(re.findall(zh_pause_punc, gen_text))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        gr.Info(f"Generating audio using F5-TTS")
        with torch.inference_mode():
            generated, _ = base_model.sample(
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
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # wav -> numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()

        if remove_silence:
            gr.Info("Removing audio silences... This may take a moment")
            non_silent_intervals = librosa.effects.split(generated_wave, top_db=30)
            non_silent_wave = np.array([])
            for interval in non_silent_intervals:
                start, end = interval
                non_silent_wave = np.concatenate([non_silent_wave, generated_wave[start:end]])
            generated_wave = non_silent_wave


        # spectogram
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
            torchaudio.save(wav_path, torch.tensor(generated_wave), target_sample_rate)

        return wav_path