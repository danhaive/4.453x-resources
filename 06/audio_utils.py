import numpy as np
import subprocess as sp
import os
DEVNULL = open(os.devnull, 'w')

def ffmpeg_save_audio(filename, y, sr=44100):
    # should allow bitrate argument
    # should allow stereo output
    pipe = sp.Popen([
        'ffmpeg',
        '-y', # (optional) means overwrite the output file if it already exists.
        '-f', 's16le', # means 16bit input
        '-acodec', 'pcm_s16le', # means raw 16bit input
        '-ar', str(sr), # the input will have 44100 Hz
        '-ac','1', # the input will have 1 channels (mono)
        '-i', '-', # means that the input will arrive from the pipe
        '-vn', # means 'don't expect any video input'
        filename],
        stdin=sp.PIPE, stdout=DEVNULL, stderr=DEVNULL, bufsize=4096, close_fds=True)
    y16 = (y * np.iinfo(np.int16).max).astype(np.int16)
    pipe.stdin.write(y16.tostring())
    pipe.stdin.close()
    pipe.wait()
    

def convert_bit_depth(y, in_type, out_type, normalize=False):
    in_type = np.dtype(in_type).type
    out_type = np.dtype(out_type).type
    
    if normalize:
        peak = np.abs(y).max()
        if peak == 0:
            normalize = False
            
    if issubclass(in_type, np.floating):
        if normalize:
            y /= peak
        if issubclass(out_type, np.integer):
            y *= np.iinfo(out_type).max
        y = y.astype(out_type)
    elif issubclass(in_type, np.integer):
        if issubclass(out_type, np.floating):
            y = y.astype(out_type)
            if normalize:
                y /= peak
        elif issubclass(out_type, np.integer):
            in_max = peak if normalize else np.iinfo(in_type).max
            out_max = np.iinfo(out_type).max
            if out_max > in_max:
                y = y.astype(out_type)
                y *= (out_max / in_max)
            elif out_max < in_max:
                y /= (in_max / out_max)
                y = y.astype(out_type)
    return y

# load_audio can not detect the input type
def ffmpeg_load_audio(filename, sr=44100, mono=False, normalize=True, in_type=np.int16, out_type=np.float32):
    in_type = np.dtype(in_type).type
    out_type = np.dtype(out_type).type
    channels = 1 if mono else 2
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]
    command = [
        'ffmpeg',
        '-i', filename,
        '-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-']
    p = sp.Popen(command, stdout=sp.PIPE, stderr=DEVNULL, bufsize=4096, close_fds=True)
    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.fromstring(raw, dtype=in_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()

    if audio.size == 0:
        return audio.astype(out_type), sr
    
    audio = convert_bit_depth(audio, in_type, out_type, normalize)

    return audio, sr