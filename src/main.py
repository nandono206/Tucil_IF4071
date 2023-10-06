from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os 

folder_path = './dictionary/'
template = []
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through the files
    for file_name in files:
        # You can process each file here, for example, print the file name
        print(file_name)
        (rate,sig) = wav.read(folder_path + file_name)
        print(sig.shape )
        mfcc_feat = mfcc(sig,rate,nfilt = 39,numcep = 39,nfft =2048)
        # fbank_feat = logfbank(sig,rate)

        # print(fbank_feat[1:3,:])
        print(mfcc_feat.shape)
        template.append([file_name[:-4],mfcc_feat])
        print(mfcc_feat.shape)
