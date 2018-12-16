function equalize_noise(fh,listen_sound, Fs)

flag_1=0;
flag_2=0;

texto=sprintf(['Set the volume of your computer in the middle of the range.\n\n'...
    'Press the Space Bar to continue']);
instruccion(fh, texto, ' ');

texto=sprintf(['VOLUMEN ADJUSTMENT STEP.\n As soon as you press the space bar you will start hearing a sound.'...
    'Then, increase the volume of your computer as much as possible without feeling uncomfortable.\n'...
    'If the volume is not loud enough you will NOT be able to perform the task correctly.\n\n'...
    'Press the space bar to start.']);
instruccion(fh, texto, ' ');

while flag_2==0
    % Perform low-level initialization of the sound driver:
    InitializePsychSound(1);
    % Provide some debug output:
    PsychPortAudio('Verbosity', 10);
    
    % Open default audio device [] for playback (mode 1), low latency (2), Fs Hz,
    % stereo output
    paoutput = PsychPortAudio('Open', [], 1, 2, Fs, 1);
    PsychPortAudio('FillBuffer', paoutput, listen_sound);
    
    % Start the playback engine
    playbackstart = PsychPortAudio('Start', paoutput, 0, 0, 1);
    
    optext=sprintf('Increase the volume and press spaceBar when you feel it is loud enough.');
    figure(fh)
    htext_ca  = uicontrol('Style','text','FontName','Arial', 'FontWeight', 'bold', 'String',optext,'Units','normalized', ...
        'BackgroundColor','w','ForegroundColor',[0. 0. 0.],'Position',[0.1, 0.5, 0.8, 0.3],'FontSize',36,'HorizontalAlignment', 'center');
    uiwait(fh);
    ch = get(fh, 'Userdata');
    
    if ch==32
        PsychPortAudio('Stop', paoutput, 1);
        PsychPortAudio('Close');
        flag_2=1;
        delete(htext_ca);
        drawnow;
        WaitSecs(0.5);
    end
end

end
