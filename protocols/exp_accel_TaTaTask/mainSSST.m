%%%% Screen preferences 1440x900
clear all
close all

% Subject parameters
subject= inputdlg('Name');
subject{1}=['datos/s_' subject{1}];
mkdir(subject{1});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% The subjects fix the intensity as high as possible
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Infolder=['WAVS/'];
[listen_sound, Fs]=audioread([Infolder 'volume.wav']);
listen_sound=4*listen_sound;
tiempo_L=size(listen_sound,1)/Fs;
fh=getkeyn;
equalize_noise(fh, listen_sound', Fs);
close all
fh=getkeyn;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% TWO random blocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
flag=1;
for iBlock=1:2
    randLanguageW( iBlock, subject{1},fh);
    if iBlock==1
        texto=sprintf(['STOP WHISPERING! You finished the first block.\n\n'...
            'Press Space Bar to complete the second one.']);
        instruccion(fh, texto, ' ');
        close all
        fh=getkeyn;
    end
end

texto=sprintf('YOU FINISHED THE EXPERIMENT, THANKS!');
instruccion(fh, texto, 'f');
close all;