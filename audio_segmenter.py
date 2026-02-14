# audio_segmenter.py

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
import soundfile as sf


class AudioSegmenter:
    """
    Audio segmentation engine that detects speech moments and groups them into fluxes.
    """
    
    def __init__(
        self,
        window_ms: float = 50.0,
        hop_ms: float = 25.0,
        noise_percentile: float = 25.0,
        speech_margin_db: float = 8.0,
        min_segment_duration: float = 0.15,
        short_gap: float = 0.8,
        long_gap: float = 2.5,
    ):
        self.window_ms = window_ms
        self.hop_ms = hop_ms
        self.noise_percentile = noise_percentile
        self.speech_margin_db = speech_margin_db
        self.min_segment_duration = min_segment_duration
        self.short_gap = short_gap
        self.long_gap = long_gap
    
    def ffmpeg_extract_mono_wav(self, input_media: Path, out_wav: Path) -> None:
        """Extract mono WAV at 44.1kHz for analysis."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_media),
            "-vn",
            "-ac", "1",
            "-ar", "44100",
            "-f", "wav",
            str(out_wav),
        ]
        print("Extracting mono WAV...")
        subprocess.run(cmd, check=True, capture_output=True)
    
    def compute_rms_db(self, wav_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
        """Compute RMS energy in dB over sliding windows."""
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        window_samples = int(sr * self.window_ms / 1000.0)
        hop_samples = int(sr * self.hop_ms / 1000.0)
        
        if window_samples <= 0 or hop_samples <= 0:
            raise ValueError("Invalid window/hop configuration")
        
        frames = []
        times = []
        i = 0
        while i + window_samples <= len(data):
            seg = data[i : i + window_samples]
            frames.append(seg)
            center_time = (i + window_samples / 2) / sr
            times.append(center_time)
            i += hop_samples
        
        frames = np.stack(frames, axis=0)
        times = np.array(times)
        
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        rms_db = 20 * np.log10(rms + 1e-12)
        
        return rms_db, times, sr
    
    def label_activity(self, rms_db: np.ndarray) -> Tuple[np.ndarray, float]:
        """Label frames as speech or silence based on adaptive threshold."""
        noise_floor = np.percentile(rms_db, self.noise_percentile)
        thresh = noise_floor + self.speech_margin_db
        active = rms_db > thresh
        return active, float(thresh)
    
    def group_segments(
        self,
        active_mask: np.ndarray,
        times: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Group consecutive active/inactive frames into segments."""
        speech = []
        silence = []
        
        def flush_segment(seg_type: str, start_idx: int, end_idx: int):
            if start_idx is None:
                return
            t_start = float(times[start_idx])
            t_end = float(times[end_idx])
            dur = t_end - t_start
            if dur < self.min_segment_duration:
                return
            seg = {
                "start": t_start,
                "end": t_end,
                "duration": dur,
                "type": seg_type,
            }
            if seg_type == "speech":
                speech.append(seg)
            else:
                silence.append(seg)
        
        current_type = None
        start_idx = None
        
        for i, is_active in enumerate(active_mask):
            t = "speech" if is_active else "silence"
            if current_type is None:
                current_type = t
                start_idx = i
            elif t != current_type:
                flush_segment(current_type, start_idx, i - 1)
                current_type = t
                start_idx = i
        
        if current_type is not None:
            flush_segment(current_type, start_idx, len(active_mask) - 1)
        
        return speech, silence
    
    def classify_intensity(
        self,
        rms_db: np.ndarray,
        times: np.ndarray,
        seg: Dict[str, Any]
    ) -> str:
        """Classify segment intensity based on mean dB level."""
        mask = (times >= seg["start"]) & (times <= seg["end"])
        if not np.any(mask):
            seg["mean_db"] = None
            seg["peak_db"] = None
            return "unknown"
        
        vals = rms_db[mask]
        mean_db = float(vals.mean())
        seg["mean_db"] = mean_db
        seg["peak_db"] = float(vals.max())
        
        if mean_db < -30:
            return "very_soft"
        elif mean_db < -24:
            return "soft"
        elif mean_db < -18:
            return "normal"
        elif mean_db < -12:
            return "loud"
        else:
            return "very_loud"
    
    def build_moments_and_fluxes(
        self,
        speech_segments: List[Dict[str, Any]],
        silence_segments: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build moments and fluxes from speech segments."""
        speech_segments = sorted(speech_segments, key=lambda s: s["start"])
        silence_segments = sorted(silence_segments, key=lambda s: s["start"])
        
        sil_idx = 0
        fluxes: List[Dict[str, Any]] = []
        moments: List[Dict[str, Any]] = []
        
        def new_flux_id(i: int) -> str:
            return f"flux_{i:03d}"
        
        flux_i = 1
        current_flux = {"id": new_flux_id(flux_i), "moments": [], "start": None, "end": None}
        
        def find_max_silence_between(t1: float, t2: float) -> float:
            nonlocal sil_idx
            max_gap = 0.0
            while sil_idx < len(silence_segments) and silence_segments[sil_idx]["end"] <= t1:
                sil_idx += 1
            j = sil_idx
            while j < len(silence_segments) and silence_segments[j]["start"] < t2:
                gap = silence_segments[j]["duration"]
                max_gap = max(max_gap, gap)
                j += 1
            return max_gap
        
        for idx, seg in enumerate(speech_segments, start=1):
            mid = f"m_{idx:03d}"
            moment = {
                "id": mid,
                "index": idx - 1,
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "intensity": seg.get("intensity", "unknown"),
                "mean_db": seg.get("mean_db"),
                "peak_db": seg.get("peak_db"),
            }
            moments.append(moment)
            
            if idx > 1:
                prev = speech_segments[idx - 2]
                gap_len = find_max_silence_between(prev["end"], seg["start"])
                if gap_len > self.long_gap:
                    if current_flux["moments"]:
                        current_flux["end"] = prev["end"]
                        current_flux["duration"] = current_flux["end"] - current_flux["start"]
                        fluxes.append(current_flux)
                    flux_i += 1
                    current_flux = {"id": new_flux_id(flux_i), "moments": [], "start": None, "end": None}
                elif gap_len > self.short_gap:
                    current_flux["has_pause"] = True
            
            if current_flux["start"] is None:
                current_flux["start"] = seg["start"]
            current_flux["moments"].append(mid)
        
        if current_flux["moments"]:
            current_flux["end"] = speech_segments[-1]["end"]
            current_flux["duration"] = current_flux["end"] - current_flux["start"]
            fluxes.append(current_flux)
        
        return moments, fluxes
    
    def segment(self, input_media: Path, keep_temp_wav: bool = False) -> Dict[str, Any]:
        """Main segmentation pipeline."""
        input_media = Path(input_media).expanduser().resolve()
        
        if keep_temp_wav:
            tmp_wav = input_media.with_name(input_media.stem + "_seg_tmp.wav")
        else:
            tmp_dir = tempfile.mkdtemp()
            tmp_wav = Path(tmp_dir) / "temp_audio.wav"
        
        try:
            self.ffmpeg_extract_mono_wav(input_media, tmp_wav)
            rms_db, times, sr = self.compute_rms_db(tmp_wav)
            active_mask, thresh = self.label_activity(rms_db)
            speech, silence = self.group_segments(active_mask, times)
            
            for seg in speech:
                seg["intensity"] = self.classify_intensity(rms_db, times, seg)
            
            moments, fluxes = self.build_moments_and_fluxes(speech, silence)
            
            total_speech = sum(s["duration"] for s in speech)
            total_silence = sum(s["duration"] for s in silence)
            total_duration = times[-1] if len(times) > 0 else 0
            
            return {
                "input_media": str(input_media),
                "sample_rate": sr,
                "total_duration": total_duration,
                "total_speech": total_speech,
                "total_silence": total_silence,
                "speech_ratio": total_speech / max(total_duration, 0.001),
                "threshold_db": thresh,
                "speech_segments": speech,
                "silence_segments": silence,
                "moments": moments,
                "fluxes": fluxes,
                "moment_count": len(moments),
                "flux_count": len(fluxes),
            }
        finally:
            if not keep_temp_wav and tmp_wav.exists():
                tmp_wav.unlink()
                try:
                    tmp_wav.parent.rmdir()
                except:
                    pass
    
    def save_json(self, result: Dict[str, Any], output_path: Path) -> Path:
        """Save segmentation result to JSON."""
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return output_path
    
    @staticmethod
    def load_json(json_path: Path) -> Dict[str, Any]:
        """Load segmentation result from JSON."""
        with Path(json_path).open("r", encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_segmenter.py /path/to/video_or_audio")
        sys.exit(1)
    
    media_path = Path(sys.argv[1]).expanduser().resolve()
    if not media_path.exists():
        print(f"File not found: {media_path}")
        sys.exit(1)
    
    segmenter = AudioSegmenter()
    result = segmenter.segment(media_path)
    
    out_json = media_path.with_name(media_path.stem + "_segments.json")
    segmenter.save_json(result, out_json)
    
    print(f"\n✓ Segmentation complete!")
    print(f"  Moments: {result['moment_count']}")
    print(f"  Fluxes: {result['flux_count']}")
    print(f"  Speech ratio: {result['speech_ratio']*100:.1f}%")
    print(f"  JSON: {out_json}")