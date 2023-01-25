import pydub 
import numpy as np
import math
def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

def split(x,size,channels):
    """spilt numpy to specific size"""
    data_list=[]
    i=0
    total=(x.shape[0]/size)
    while(x.shape[0]>=size):    
        data,x=np.vsplit(x, [size])
        data_list.append(data)
        i=i+1
        if(i%10==0):
            print("Process =",i*100/total,"%")
    data=np.array(data_list)
    return data
    
audio_file = 'piano.mp3'

sr, x = read(audio_file)
print('herre')
print(sr,x.shape)

data=split(x,480000,2)
write("test.mp3",sr,data[0])
print(data.shape)
np.save("piano data.npy",data)
