function rate_training_function(subject, fh,iBlocks)

% Wait for release of all keys on keyboard:
tiempo=10;
%tiempo=2;
%%%%%%%%%%%%%%
%%%%% Training the rythm
%%%%%%%%%%%%%
for i=1:1
    name_output=[subject '/train_' num2str(iBlocks) '.wav']
    [listen_sound, Fs]=audioread(['WAVS/example.wav']);
    listen_sound=listen_sound';
    
    texto=sprintf(['RATE TRAINING STEP\n\n'...
        'Once you press the space bar you will start hearing a string of sounds. Just pay attention to it.'...
        '\n\nPress the Space Bar to continue']);
    instruccion(fh, texto, ' ');
    %%%%%%%
    optext=sprintf('Do not whisper, just pay attention.');
    figure(fh)
    htext_ca  = uicontrol('Style','text','FontName','Arial', 'FontWeight', 'bold', 'String',optext,'Units','normalized', ...
        'BackgroundColor','w','ForegroundColor',[0. 0. 0.],'Position',[0.1, 0.5, 0.8, 0.3],'FontSize',36,'HorizontalAlignment', 'center');
    drawnow
    WaitSecs(0.5);
    %%%%%%%
    
    % Perform low-level initialization of the sound driver:
    InitializePsychSound(1);
    % Provide some debug output:
    PsychPortAudio('Verbosity', 10);
    % Open the default audio device [], with mode 2 (== Only audio capture),
    % and a required latencyclass of 2 == low-latency mode, as well as
    % a Fsuency of Fs Hz and 2 sound channels for stereo capture.
    % This returns a handle to the audio device:
    % Open default audio device [] for playback (mode 1), low latency (2), Fs Hz,
    % stereo output
    paoutput = PsychPortAudio('Open', [], 1, 2, Fs, 1);
    PsychPortAudio('FillBuffer', paoutput, listen_sound);
    playbackstart = PsychPortAudio('Start', paoutput, 0, 0, 1);
    WaitSecs(tiempo);
    PsychPortAudio('Stop', paoutput, 1);
    PsychPortAudio('Close');
    
    %%%%%
    delete(htext_ca);
    drawnow;
    %%%%%
    
    texto=sprintf(['After pressing the space bar start whispering "Tah Tah Tah..." with the same rhythm of the previous audio. '...
        'Remember you are supposed to WHISPER, do not speak loud.\n\n'...
        'Press Space Bar to continue']);
    instruccion(fh, texto, ' ');
    optext=sprintf('RECORDING, KEEP WHISPERING');
    figure(fh)
    htext_ca  = uicontrol('Style','text','FontName','Arial', 'FontWeight', 'bold', 'String',optext,'Units','normalized', ...
        'BackgroundColor','w','ForegroundColor',[0. 0. 0.],'Position',[0.1, 0.5, 0.8, 0.3],'FontSize',36,'HorizontalAlignment', 'center');
    drawnow
    % Perform low-level initialization of the sound driver:
    InitializePsychSound(1);
    % Provide some debug output:
    PsychPortAudio('Verbosity', 10);
    % Open the default audio device [], with mode 2 (== Only audio capture),
    % and a required latencyclass of 2 == low-latency mode, as well as
    % a Fsuency of Fs Hz and 2 sound channels for stereo capture.
    % This returns a handle to the audio device:
    painput = PsychPortAudio('Open', [], 2, 2, Fs, 1);
    % Preallocate an internal audio recording  buffer with a capacity of :
    PsychPortAudio('GetAudioData', painput, tiempo);
    
    % Start audio capture
    painputstart = PsychPortAudio('Start', painput, 0, 0, 1);
    uiwait(fh, tiempo-0.1);
    [audiodata, offset, overrun] = PsychPortAudio('GetAudioData', painput);
    PsychPortAudio('Stop', painput, 1);
    PsychPortAudio('Close');
    
    audiowrite(name_output,audiodata, Fs);
    delete(htext_ca);
    drawnow;
    WaitSecs(0.5);
end
texto=sprintf(['Great! You already know how to pronounce the "tahs". Lets move to the next step.\n\n'...
    'Press Space Bar to continue']);
instruccion(fh, texto, ' ');
WaitSecs(0.5);



