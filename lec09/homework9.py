import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    #raise RuntimeError("You need to change this part")
    framelength = int(0.025*Fs)
    frameskip = int(0.01*Fs)
    frames = np.array([ waveform[m:m+framelength] for m in range(0, len(waveform)-framelength, frameskip) ])
    energy = np.sum(np.square(frames), axis=1)
    VAD = [ 1 if energy[m]>0.1*max(energy) else 0 for m in range(len(energy)) ]
    start_times = [ m*frameskip for m in range(len(VAD)) if VAD[m-1]==0 and VAD[m]==1 ]
    end_times = [ m*frameskip for m in range(len(VAD)) if VAD[m-1]==1 and VAD[m]==0 ]
    segments = [ waveform[start_times[k]:end_times[k]] for k in range(len(start_times)) ]
    return segments

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    #raise RuntimeError("You need to change this part")
    N = int(0.004*Fs)
    S = int(0.002*Fs)
    models = []
    for k in range(len(segments)):
        frames = np.array([ segments[k][m+1:m+N+1]-segments[k][m:m+N] for m in range(0, len(segments[k])-N, S) ])
        mstft = np.abs(np.fft.fft(frames, axis=1))
        sgram = 20*np.log10( np.maximum(0.001*np.amax(mstft), mstft) )
        models.append( np.average( sgram[:, 0:int(N/2)], axis=0 ) )
    return models

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    #raise RuntimeError("You need to change this part")
    segments = VAD(testspeech, Fs)
    testspectra = segments_to_models(segments, Fs)
    Y = len(models)
    K = len(testspectra)
    sims = np.zeros((Y,K))
    for y in range(Y):
        for k in range(K):
            x = testspectra[k]
            m = models[y]
            sims[y,k] = np.dot(x,m) / (np.sqrt(np.sum(np.square(m)))*np.sqrt(np.sum(np.square(x))))
    test_outputs = []
    for k in range(len(testspectra)):
        test_outputs.append( labels[np.argmax(sims[:,k])] )
    return sims, test_outputs
    #return 1, []
        

