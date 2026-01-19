import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    #raise RuntimeError("You need to write this part!")
    nframes = int((len(speech)-frame_length)/frame_skip)
    frames = np.array([speech[m*frame_skip:m*frame_skip+frame_length] for m in range(nframes)])
    A = librosa.lpc(frames, order=order)
    excitation = np.zeros((nframes,frame_length))
    for m in range(nframes):
        for n in range(order,frame_length):
            for k in range(0,order+1):
                excitation[m,n] += A[m,k]*frames[m,n-k]
    return A, excitation

def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    #raise RuntimeError("You need to write this part!")
    synthesis = e
    for n in range(len(synthesis)):
        frame = int(n/frame_skip)
        for k in range(1,min(n,11)):
            synthesis[n] -= A[frame,k] * synthesis[n-k]
    return synthesis

def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    #raise RuntimeError("You need to write this part!")
    nframes, frame_length = excitation.shape
    gain = np.zeros(nframes)
    for m in range(nframes):
        gain[m] = np.sqrt(np.average(np.square(excitation[m,:])))
    e_robot = np.zeros(nframes*frame_skip)
    n = 0
    while n < len(e_robot):
        e_robot[n] = gain[int(n/frame_skip)]
        n += T0
    return gain, e_robot
    
