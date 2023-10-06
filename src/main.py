from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("./dictionary/pemrosesan.wav")
mfcc_feat = mfcc(sig,rate, nfft=1500, numcep=39)
# fbank_feat = logfbank(sig,rate,  nfft=1500)

print(mfcc_feat.shape)