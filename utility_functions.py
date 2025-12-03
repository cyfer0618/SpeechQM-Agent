
import torch
import librosa
import torchaudio.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer
import numpy as np
from dataclasses import dataclass
import pandas as pd
import logging
import torchaudio
import os
import fasttext
import json
from my_secrets import HF_TOKEN
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pydub import AudioSegment
import soundfile as sf
import csv
import re
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from ai4bharat.transliteration import XlitEngine
from typing import List, Tuple

def language_identification_indiclid(transcription: str) -> List[Tuple[str, str, float, str]]:
    logging.info(f"Processing transcription for IndicLID: {transcription[:50]}...")
    try:
        if not transcription or transcription.strip() == "":
            return [(transcription, "Unknown", 0.0, "IndicLID")]
        
        indiclid_model = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
        results = indiclid_model.batch_predict([transcription], batch_size=1)
        
        if not results or not isinstance(results, list):
            raise ValueError(f"Invalid result format from IndicLID: {results}")
        
        output = []
        for text, lang_code, confidence, model_used in results:
            if not isinstance(confidence, (int, float)):
                confidence = 0.0
            output.append((text, lang_code, float(confidence), model_used))
        
        return output if output else [(transcription, "Unknown", 0.0, "IndicLID")]
    
    except Exception as e:
        logging.warning(f"IndicLID failed for transcription '{transcription[:50]}...': {e}")
        return [(transcription, "Error", 0.0, "IndicLID")]

def transliterate_file(file_path: str, lang_code: str) -> str:
    try:
        import numpy as np
        if not hasattr(np, 'float'):
             np.float = float
        supported_langs = {'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'sd', 'si', 'ta', 'te', 'ur'}
        if lang_code not in supported_langs:
            raise ValueError(f"Unsupported language code: {lang_code}. Supported codes: {', '.join(supported_langs)}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        df = pd.read_csv(file_path)
        if "ground_truth" not in df.columns:
            raise ValueError("Column 'ground_truth' not found in the CSV file")
        engine = XlitEngine(beam_width=10, src_script_type="roman")
        def transliterate(text):
            if pd.isna(text) or str(text).strip() == "":
                return ""
            try:
                result = engine.translit_sentence(str(text), lang_code)
                if isinstance(result, dict) and lang_code in result:
                    return result[lang_code]
                else:
                    return result
            except Exception as e:
                print(f"Error transliterating '{text}': {e}")
                return text  
        df["ground_truth_transliterated"] = df["ground_truth"].apply(transliterate)
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_transliterated.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        return f"Output saved at: {output_path}"
    except Exception as e:
        print(f"Error in transliterate_file: {e}")
        return f"Error: {e}"

def transcribe_audio(audio_path, source_lang):
    model_ids = {
        "Hindi": "ai4bharat/indicwav2vec-hindi",
        "Tamil": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
        "Sanskrit": "addy88/wav2vec2-sanskrit-stt",
        "Marathi": "ai4bharat/indicwav2vec-marathi",
        "Telugu": "ai4bharat/indicwav2vec-telugu",
    }
    if source_lang not in model_ids:
        raise ValueError(f"Unsupported language for ai4bharat/indicwav2vec2: {source_lang}")
    model_id = model_ids[source_lang]
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    print(f"Transcribing {audio_path} using {model_id}")
    audio_array, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        emission = torch.nn.functional.log_softmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    transcript_words = transcription[0].strip().split()
    print("TRANSCRIPT:", transcript_words)
    return " ".join(transcript_words)


def transcribe_folder_to_csv(folder_path: str, source_language: str):
    parent = os.path.dirname(folder_path.rstrip("/"))
    output_path = os.path.join(parent, 'results', "indicconf_hypothesis.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        logging.info(f"Skipping transcription: File already exists at {output_path}")
        return output_path

    results = []
    for filename in os.listdir(folder_path):
        audio_path = os.path.join(folder_path, filename)
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logging.warning(f"Skipping {filename}: File size is 0 bytes")
            continue

        delete_after = False 

        if filename.lower().endswith(".mp3"):
            try:
                wav_filename = os.path.splitext(filename)[0] + ".wav"
                wav_path = os.path.join(folder_path, wav_filename)
                audio = AudioSegment.from_mp3(audio_path)
                audio.export(wav_path, format="wav")
                logging.info(f"Converted {filename} to {wav_filename}")
                audio_path = wav_path
                transcribe_filename = filename
                delete_after = True 
            except Exception as e:
                logging.error(f"Failed to convert {filename} to .wav: {e}")
                continue

        elif not filename.lower().endswith(".wav"):
            logging.warning(f"Skipping {filename}: Not a .wav or .mp3 file")
            continue
        else:
            transcribe_filename = filename

        try:
            hypothesis = transcribe_audio(audio_path, source_language)
            results.append({
                "Filename": transcribe_filename,
                "Indiconformer_Hypothesis": hypothesis
            })
            logging.info(f"Transcribed {transcribe_filename}: {hypothesis[:50]}...")
        except Exception as e:
            logging.error(f"Failed to transcribe {transcribe_filename}: {e}")
        finally:
            if delete_after:
                try:
                    os.remove(audio_path)
                    logging.info(f"Deleted temporary file: {audio_path}")
                except Exception as e:
                    logging.error(f"Failed to delete temporary .wav file {audio_path}: {e}")

    if not results:
        logging.error("No valid transcriptions generated")
        return "Error: No valid transcriptions"

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logging.info(f"Transcriptions saved to {output_path}")
    return output_path


token =  HF_TOKEN
model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
    "min_duration_on": 0.0,
    "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)
silent_list = []

def perform_vad(audio_file):
    vad_result = pipeline(audio_file)
    return vad_result

def get_total_silence_time(audio_file_path):
    vad_result = perform_vad(audio_file_path)
    audio, sr = librosa.load(audio_file_path, sr=16000)
    total_audio_duration = len(audio) / sr
    total_silence = 0.0
    last_end = 0.0
    for segment in vad_result.itersegments():
        start = segment.start
        end = segment.end
        if start > last_end:
            silence_duration = start - last_end
            total_silence += silence_duration
        last_end = end
    if last_end < total_audio_duration:
        total_silence += total_audio_duration - last_end
    return total_silence


def process_folder_vad(audio_folder: str):
    parent  = os.path.dirname(audio_folder.rstrip("/"))
    output_path = os.path.join(parent, 'results', "vad_silence_stats.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        logging.info(f"Skipping VAD processing: File already exists at {output_path}")
        return f"File already exists: {output_path}"
    
    
    results = []
    for filename in os.listdir(audio_folder):
        if filename.lower().endswith((".wav", ".flac", ".mp3")):
            audio_path = os.path.join(audio_folder, filename)
            try:
                silence = get_total_silence_time(audio_path)
                results.append({
                    "Filename": filename,
                    "Total Silence (s)": round(silence, 2)
                })
                print(f"Processed: {filename}, Silence: {round(silence, 2)}s")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logging.info(f"VAD results saved to {output_path}")
    return f"CSV saved at: {output_path}"


def save_num_speakers(folder_path: str, model_token=HF_TOKEN) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=model_token)
        pipeline.to(device)
    except Exception as e:
        return f"Failed to load diarization pipeline: {e}"
    parent = os.path.dirname(folder_path.rstrip("/"))
    output_csv = os.path.join(parent, 'results', "num_speakers.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if os.path.exists(output_csv):
        logging.info(f"Skipping speaker number calculation: File already exists at {output_csv}")
        return output_csv
    
    try:
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File Name", "Number of Speakers", "Speaker Durations"])
            for filename in sorted(os.listdir(folder_path)):
                if filename == "num_speakers.csv":
                    continue
                if filename.endswith((".wav", ".mp3")):
                    audio_path = os.path.join(folder_path, filename)
                    try:
                        diarization = pipeline(audio_path)
                        unique_speakers = set()
                        speaker_durations = {}
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            unique_speakers.add(speaker)
                            duration = turn.end - turn.start
                            duration_hours = duration / 3600
                            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration_hours
                        speaker_durations_json = json.dumps({k: round(v, 4) for k, v in speaker_durations.items()})
                        writer.writerow([filename, len(unique_speakers), speaker_durations_json])
                        print(f"Processed {filename}: {len(unique_speakers)} unique speakers, Durations: {speaker_durations_json}")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        writer.writerow([filename, "Error", f"Error: {e}"])
        return f"CSV saved at: {output_csv}"
    except Exception as e:
        return f"Error writing to CSV: {e}"

def transcript_quality(transcript):
    words = transcript.strip().split()
    repeated = len(set(words)) < len(words) * 0.7
    chars = set(transcript)
    if len(words) > 3 and not repeated:
        return "passed"
    else:
        return "failed"

def force_alignment_and_ctc_score(speech_file, given_transcript, model_name="Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device in force_alignment_and_ctc_score: {device}")
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        wav, sr = librosa.load(speech_file, sr=16000)
        input_values = processor(wav, return_tensors="pt", sampling_rate=16000).input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
            emissions = torch.log_softmax(logits, dim=-1).cpu()
        token_ids = processor.tokenizer.encode(given_transcript, add_special_tokens=False)
        def get_trellis(emission, token_ids, blank_id=0):
            num_frame = emission.size(0)
            num_tokens = len(token_ids)
            trellis = torch.zeros((num_frame, num_tokens))
            trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
            trellis[0, 1:] = -float("inf")
            trellis[-num_tokens + 1:, 0] = float("inf")
            for t in range(num_frame - 1):
                trellis[t + 1, 1:] = torch.maximum(
                    trellis[t, 1:] + emission[t, blank_id],
                    trellis[t, :-1] + emission[t, token_ids[1:]],
                )
            return trellis
        trellis = get_trellis(emissions[0], token_ids)
        @dataclass
        class Point:
            token_index: int
            time_index: int
            score: float
        def backtrack(trellis, emission, token_ids, blank_id=0):
            t, j = trellis.size(0) - 1, trellis.size(1) - 1
            path = [Point(j, t, emission[t, blank_id].exp().item())]
            while j > 0 and t > 0:
                p_stay = emission[t - 1, blank_id]
                p_change = emission[t - 1, token_ids[j]]
                stayed = trellis[t - 1, j] + p_stay
                changed = trellis[t - 1, j - 1] + p_change
                t -= 1
                if changed > stayed:
                    j -= 1
                prob = (p_change if changed > stayed else p_stay).exp().item()
                path.append(Point(j, t, prob))
            while t > 0:
                prob = emission[t - 1, blank_id].exp().item()
                path.append(Point(j, t - 1, prob))
                t -= 1
            return path[::-1]
        path = backtrack(trellis, emissions[0], token_ids)
        @dataclass
        class Segment:
            label: str
            start: int
            end: int
            score: float
        def merge_repeats(path):
            i1, i2 = 0, 0
            segments = []
            while i1 < len(path):
                while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                    i2 += 1
                avg_score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
                segments.append(Segment(
                    processor.tokenizer.convert_ids_to_tokens(token_ids[path[i1].token_index]),
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    avg_score
                ))
                i1 = i2
            return segments
        segments = merge_repeats(path)
        waveform = torch.tensor(wav).unsqueeze(0)
        sample_rate = sr
        ratio = waveform.size(1) / trellis.size(0)
        fa = []
        for i, seg in enumerate(segments):
            if seg.label not in ["|", "<unk>", "<pad>"]:
                x0 = int(ratio * seg.start)
                x1 = int(ratio * seg.end)
                start_time = x0 / sample_rate
                end_time = x1 / sample_rate
                fa.append([seg.label, f"{start_time:.2f}", f"{end_time:.2f}", f"{seg.score:.2f}"])
        filtered_segments = [seg for seg in segments if seg.label not in ["|", "<unk>", "<pad>"]]
        total_score = sum(seg.score for seg in filtered_segments)
        average_ctc_score = total_score / len(filtered_segments) if filtered_segments else 0
        return fa, average_ctc_score
    except Exception as e:
        logging.error(f"Error in forced alignment for {speech_file}: {e}")
        return [], 0

def process_audio_directory(audio_dir, transcript_csv, output_csv_path=None):
    try:
        df = pd.read_csv(transcript_csv)
        results = []
        for idx, row in df.iterrows():
            filename = row['filename']
            transcript = str(row['ground_truth'])
            audio_path = os.path.join(audio_dir, filename)
            if not os.path.exists(audio_path):
                logging.warning(f"Audio file not found: {audio_path}")
                results.append({
                    "filename": filename,
                    "label": "Error",
                    "start_time": "0.00",
                    "end_time": "0.00",
                    "score": "0.00",
                    "average_ctc_score": "0.00"
                })
                continue
            logging.info(f"Processing: {filename}")
            aligned_segments, avg_score = force_alignment_and_ctc_score(audio_path, transcript)
            if not aligned_segments:
                results.append({
                    "filename": filename,
                    "label": "Error",
                    "start_time": "0.00",
                    "end_time": "0.00",
                    "score": "0.00",
                    "average_ctc_score": f"{avg_score:.2f}"
                })
                continue
            for seg in aligned_segments:
                results.append({
                    "filename": filename,
                    "label": seg[0],
                    "start_time": seg[1],
                    "end_time": seg[2],
                    "score": seg[3],
                    "average_ctc_score": f"{avg_score:.2f}"
                })
        if output_csv_path:
            keys = results[0].keys() if results else ["filename", "label", "start_time", "end_time", "score", "average_ctc_score"]
            with open(output_csv_path, "w", newline='', encoding="utf-8") as out_file:
                writer = csv.DictWriter(out_file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            logging.info(f"Results saved to {output_csv_path}")
        return results
    except Exception as e:
        logging.error(f"Error in process_audio_directory: {e}")
        return []

def is_upsampled_from_8k_v2(audio_path, threshold_ratio=0.02):
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Encountered an error loading file {audio_path}")
        print(f"ERROR: {e}")
        return False
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    spectrum = np.abs(np.fft.rfft(waveform.numpy()[0]))
    freqs = np.fft.rfftfreq(len(waveform[0]), 1 / sr)
    total_energy = np.sum(spectrum)
    high_freq_band = (freqs >= 4000) & (freqs <= 8000)
    high_freq_energy = np.sum(spectrum[high_freq_band])
    ratio = high_freq_energy / total_energy
    return (ratio < threshold_ratio).item()

def check_upsampling_folder(folder_path: str):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".wav", ".mp3")):
            audio_path = os.path.join(folder_path, filename)
            try:
                is_upsampled = is_upsampled_from_8k_v2(audio_path)
                results.append({
                    "Filename": filename,
                    "Is_Upsampled": is_upsampled
                })
                print(f"Processed: {filename}, Upsampled: {is_upsampled}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                results.append({
                    "Filename": filename,
                    "Is_Upsampled": "Error"
                })
    df = pd.DataFrame(results)
    parent = os.path.dirname(folder_path.rstrip("/"))
    os.makedirs(os.path.join(parent, 'results'), exist_ok=True)
    output_path = os.path.join(parent, 'results', "upsampling_check.csv")
    df.to_csv(output_path, index=False)
    return f"CSV saved at: {output_path}"

class IndicBERT_Data(Dataset):
    def __init__(self, indices, X):
        self.size = len(X)
        self.x = X
        self.i = indices
    def __len__(self):
        return (self.size)
    def __getitem__(self, idx):
        text = self.x[idx]
        index = self.i[idx]
        return tuple([index, text])

class IndicLID():
    def __init__(self, input_threshold=0.5, roman_lid_threshold=0.6):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        absolute_path = "/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/trash"
        self.IndicLID_FTN_path = os.path.join(absolute_path, 'indiclid-ftn/model_baseline_roman.bin')
        self.IndicLID_FTR_path = os.path.join(absolute_path, 'indiclid-ftr/model_baseline_roman.bin')
        self.IndicLID_BERT_path = os.path.join(absolute_path, 'indiclid-bert/basline_nn_simple.pt')
        self.IndicLID_FTN = fasttext.load_model(self.IndicLID_FTN_path)
        self.IndicLID_FTR = fasttext.load_model(self.IndicLID_FTR_path)
        self.IndicLID_BERT = torch.load(self.IndicLID_BERT_path, map_location=self.device)
        self.IndicLID_BERT.eval()
        self.IndicLID_BERT_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
        self.input_threshold = input_threshold
        self.model_threshold = roman_lid_threshold
        self.classes = 47
        self.IndicLID_lang_code_dict = {
            'asm_Latn': 0, 'ben_Latn': 1, 'brx_Latn': 2, 'guj_Latn': 3, 'hin_Latn': 4, 'kan_Latn': 5,
            'kas_Latn': 6, 'kok_Latn': 7, 'mai_Latn': 8, 'mal_Latn': 9, 'mni_Latn': 10, 'mar_Latn': 11,
            'nep_Latn': 12, 'ori_Latn': 13, 'pan_Latn': 14, 'san_Latn': 15, 'snd_Latn': 16, 'tam_Latn': 17,
            'tel_Latn': 18, 'urd_Latn': 19, 'eng_Latn': 20, 'other': 21, 'asm_Beng': 22, 'ben_Beng': 23,
            'brx_Deva': 24, 'doi_Deva': 25, 'guj_Gujr': 26, 'hin_Deva': 27, 'kan_Knda': 28, 'kas_Arab': 29,
            'kas_Deva': 30, 'kok_Deva': 31, 'mai_Deva': 32, 'mal_Mlym': 33, 'mni_Beng': 34, 'mni_Meti': 35,
            'mar_Deva': 36, 'nep_Deva': 37, 'ori_Orya': 38, 'pan_Guru': 39, 'san_Deva': 40, 'sat_Olch': 41,
            'snd_Arab': 42, 'tam_Taml': 43, 'tel_Telu': 44, 'urd_Arab': 45
        }
        self.IndicLID_lang_code_dict_reverse = {v: k for k, v in self.IndicLID_lang_code_dict.items()}
    def pre_process(self, input):
        return input
    def char_percent_check(self, input):
        input_len = len(list(input))
        special_char_pattern = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        special_char_matches = special_char_pattern.findall(input)
        special_chars = len(special_char_matches)
        spaces = len(re.findall('\s', input))
        newlines = len(re.findall('\n', input))
        total_chars = input_len - (special_chars + spaces + newlines)
        en_pattern = re.compile('[a-zA-Z0-9]')
        en_matches = en_pattern.findall(input)
        en_chars = len(en_matches)
        if total_chars == 0:
            return 0
        return (en_chars/total_chars)
    def native_inference(self, input_list, output_dict):
        if not input_list:
            return output_dict
        input_texts = [line[1] for line in input_list]
        IndicLID_FTN_predictions = self.IndicLID_FTN.predict(input_texts)
        for input, pred_label, pred_score in zip(input_list, IndicLID_FTN_predictions[0], IndicLID_FTN_predictions[1]):
            output_dict[input[0]] = (input[1], pred_label[0][9:], pred_score[0], 'IndicLID-FTN')
        return output_dict
    def roman_inference(self, input_list, output_dict, batch_size):
        if not input_list:
            return output_dict
        input_texts = [line[1] for line in input_list]
        IndicLID_FTR_predictions = self.IndicLID_FTR.predict(input_texts)
        IndicLID_BERT_inputs = []
        for input, pred_label, pred_score in zip(input_list, IndicLID_FTR_predictions[0], IndicLID_FTR_predictions[1]):
            if pred_label[0][9:] == 'eng_Latn' and pred_score[0] > self.model_threshold:
                output_dict[input[0]] = (input[1], pred_label[0][9:], pred_score[0], 'IndicLID-FTR')
            else:
                IndicLID_BERT_inputs.append(input)
        output_dict = self.IndicBERT_roman_inference(IndicLID_BERT_inputs, output_dict, batch_size)
        return output_dict
    def IndicBERT_roman_inference(self, IndicLID_BERT_inputs, output_dict, batch_size):
        if not IndicLID_BERT_inputs:
            return output_dict
        df = pd.DataFrame(IndicLID_BERT_inputs)
        dataloader = self.get_dataloaders(df.iloc[:,0], df.iloc[:,1], batch_size)
        with torch.no_grad():
            for data in dataloader:
                batch_indices = data[0]
                batch_inputs = data[1]
                word_embeddings = self.IndicLID_BERT_tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
                word_embeddings = word_embeddings.to(self.device)
                batch_outputs = self.IndicLID_BERT(word_embeddings['input_ids'], 
                            token_type_ids=word_embeddings['token_type_ids'], 
                            attention_mask=word_embeddings['attention_mask'])
                _, batch_predicted = torch.max(batch_outputs.logits, 1)
                for index, input, pred_label, logit in zip(batch_indices, batch_inputs, batch_predicted, batch_outputs.logits):
                    output_dict[index] = (input, self.IndicLID_lang_code_dict_reverse[pred_label.item()], 
                                        logit[pred_label.item()].item(), 'IndicLID-BERT')
        return output_dict
    def post_process(self, output_dict):
        results = []
        keys = list(output_dict.keys())
        keys.sort()
        for index in keys:
            results.append(output_dict[index])
        return results
    def get_dataloaders(self, indices, input_texts, batch_size):
        data_obj = IndicBERT_Data(indices, input_texts)
        dl = torch.utils.data.DataLoader(data_obj, batch_size=batch_size, shuffle=False)
        return dl
    def predict(self, input):
        input_list = [input,]
        return self.batch_predict(input_list, 1)
    def batch_predict(self, input_list, batch_size):
        output_dict = {}
        roman_inputs = []
        native_inputs = []
        for index, input in enumerate(input_list):
            if self.char_percent_check(input) > self.input_threshold:
                roman_inputs.append((index, input))
            else:
                native_inputs.append((index, input))
        output_dict = self.native_inference(native_inputs, output_dict)
        output_dict = self.roman_inference(roman_inputs, output_dict, batch_size)
        results = self.post_process(output_dict)
        return results

def get_required_inputs(task_id: int) -> list[str]:
    input_requirements = {
        1: ["audio_dir"],  
        2: ["audio_dir"],  
        3: ["ground_truth_csv"], 
        4: ["ground_truth_csv"], 
        5: ["ground_truth_csv"],  
        6: ["ground_truth_csv"],  
        7: ["audio_dir"],  
        8: ["audio_dir"],
        9: ["audio_dir"],  
        10: ["audio_dir", "ground_truth_csv"],  
        11: ["audio_dir"],  
        12: ["audio_dir"], 
        13: ["audio_dir"],  
        14: ["audio_dir", "ground_truth_csv"], 
        15: ["audio_dir"], 
        16: ["audio_dir"], 
        17: ["audio_dir"], 
        18: ["ground_truth_csv", "lang_code"], 
        19: ["audio_dir"],  
        20: ["audio_dir"], 
        21: ["audio_dir"],  
        22: ["audio_dir"], 
        23: ["audio_dir"],
        24: ["audio_dir"],
    }
    return input_requirements.get(task_id, [])
