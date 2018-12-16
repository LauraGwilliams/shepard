function randLanguageW(iBlock, SubjectName,fh)


rate_training_function(SubjectName,fh,iBlock);
Infolder='WAVS/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Main instructions for the experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[listen_sound, Fs]=audioread([Infolder 'stimulus.wav']);
listen_sound=4*listen_sound;
tiempo_L=size(listen_sound,1)/Fs;
tiempo_L=70;
KbReleaseWait;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Main instructions for the experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

texto=sprintf(['SYNCHRONIZATION TASK\n\nYou will be listening to syllables during 70 seconds and at the same time '...
    'you have to WHISPER "Tah tah tah...".\n'...
    'Your goal is to synchronize your whisper to the external audio. In other words, you have to whisper '...
    '?tah tah tah?? at the same rhythm as the syllables you are listening to.\n'... 
    'Please, do not change the audio volume and fix your eyes on the red dot.\n\n'...  
    'Press Space Bar to continue.\n']);

instruccion(fh, texto, ' ');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Language presentation
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileOut=([SubjectName  '/output_' num2str(iBlock) '.wav']);

figure(fh)
%%%% SET THE CROSS
h=plot(1,2,'r.','MarkerSize',100);
axis(gca,'off');
drawnow;
%%%%%


% Parform low-level initialization of the sound driver:
InitializePsychSound(1);
% Provide some debug output:
PsychPortAudio('Verbosity', 10);

% Open the default audio device [], with mode 2 (== Only audio capture),
% and a required latencyclass of 2 == low-latency mode, as well as
% a Fsuency of Fs Hz and 2 sound channels for stereo capture.
% This returns a handle to the audio device:
painput = PsychPortAudio('Open', [], 2, 2, Fs, 1);
% Preallocate an internal audio recording  buffer with a capacity of :
PsychPortAudio('GetAudioData', painput, tiempo_L);

% Open default audio device [] for playback (mode 1), low latency (2), Fs Hz,
% stereo output
paoutput = PsychPortAudio('Open', [], 1, 2, Fs, 1);
PsychPortAudio('FillBuffer', paoutput, listen_sound');


% Start audio capture
painputstart = PsychPortAudio('Start', painput, 0, 0, 1);
% Start the playback engine
playbackstart = PsychPortAudio('Start', paoutput, 0, 0, 1);
delay(iBlock)=playbackstart-painputstart;

%tiempo_L=15;
WaitSecs(tiempo_L-delay(iBlock)-0.01);
%WaitSecs(4);
delete(h);
drawnow;
[audiodata, ~, ~] = PsychPortAudio('GetAudioData', painput);
PsychPortAudio('Stop', painput, 1);
PsychPortAudio('Stop', paoutput, 1);
PsychPortAudio('Close');

audiowrite(fileOut, audiodata, Fs);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clf

end



