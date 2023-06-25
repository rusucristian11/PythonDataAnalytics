
modelpath   = "models/" 
source_train   = r"C:/AUDIO-PROCESSED/DDK-Analysis/"
# source_test = r"test/" 

# Parameters for the extraction of MFCCs coefficiants (i have to test with different parameters for a better accuracy)

sr = 16000        # Sampling frequency
wst = 0.030       # Window size (seconds)
fpt = 0.015      # Overlaping Frame period (seconds) 
nfft = round(wst*sr)      # Window size (samples)
# nfft = 512 #default value
fp = round(fpt*sr)        # Frame period (samples)
nbands = 20    # Number of filters in the filterbank (40 default)
ncomp =  20    # Number of MFCC components
#####################
fmin = 0
fmax = None
n_mfcc = 13 # MFCC is a very compressible representation, often using just 20 or 13 coefficients instead of 32-64 bands in Mel spectrogram
n_mel = 40
frame_length = 256  # or any other value divisible by 2
frame_step = 129  # or any other desired value
n_features = 257
sample_rate = 1600

# GMM
# Params for making the GMM models
n_components = 10
max_iter = 100
covariance_type='diag'
n_init = 3


