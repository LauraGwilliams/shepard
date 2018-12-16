clear all
close all

root_folder='~/Desktop/analisis/';
visualizar=1;

data_folder=[root_folder 'raw_data/'];
subjects=dir([data_folder 's*']);
subjects={subjects.name};
folderOut=[root_folder 'Results/'];
mkdir(folderOut)


fs_new=100;
freqFilt=[3.3 5.7];
envelopeFile='envelope_stimulus';

%Declare here the specific name of the subject you would like to analize,
%in case you do not want to reanalize them all
%subjects={'s_mm17'}

for iSub=1:length(subjects)
    synch=[nan nan];
    subject=subjects{iSub}
    
    for iBlock=1:2
        file=['output_' num2str(iBlock)];
        
        %%%%% EXTRACTS THE SPEECH
        files=dir([data_folder subject '/' file '.wav']);
        if length(files)==1
            
            file_name=[data_folder subject '/' files.name];
            [signal_1, Fs]=audioread(file_name);
            signal_1=signal_1(:,1);
            signal_1=signal_1-mean(signal_1);
            hi=hilbert(signal_1);
            envelope_speech=abs(hi);
            
            %%% Smooths and shows the envelope with the original signal
            n_average=0.01*Fs;
            coeff= ones(1, n_average)/n_average;
            envelope_speech= filtfilt(coeff, 1, envelope_speech);
            envelope_speech=resample(envelope_speech, fs_new,Fs);
            envelope_speech=detrend(envelope_speech);
            
            %%%%%%%%%%%%%%
            %%% Computing the spectral content of the envelope from the speech
            %%%%%%%%%%%%%%
            envelope_speech=envelope_speech-mean(envelope_speech);
            L=size(envelope_speech,1);
            NFFT = 2^nextpow2(L); % Next power of 2 from length of y
            Y = fft(envelope_speech,NFFT)/L;
            f = 100/2*linspace(0,1,NFFT/2+1);
            %%% Frequencies below 10 Hz
            f=f(f<10);
            Y=Y(1:size(f,2));
            amp_fft= abs(Y).^2;
            
            %%% JUST IF VISUALIZE =1
            if visualizar==1
                
                figure('name', file)
                subplot(2,2,1)
                hold on
                time=(1:length(signal_1))./Fs;
                plot(time, signal_1, 'k');
                time=(1:length(envelope_speech))./100;
                plot(time,envelope_speech, 'r', 'LineWidth',2);
                hold off
                xlim([20 30])
    
                df=f(3)-f(2);
                n_average=round(0.2/df);
                coeff= ones(1, n_average)/n_average;
                smooth_fft= filter(coeff, 1, amp_fft);
                smooth_fft=smooth_fft/max(smooth_fft);
                subplot(2,2,2)
                plot(f, smooth_fft);
                clear time
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%
            %%%%% NOW COMPUTES THE PLV
            %%%%%%%%%%%%%%%%%%%%%%%%
            T=5;
            shift=2;
            %%% Loads the envelope of the heard language
            load([root_folder '/raw_data/AudioStim/' envelopeFile '.mat']);
            envelope_heard=detrend(envelope);
            clear envelope;
            envelope_heard=freqfiltbp(envelope_heard',freqFilt,fs_new,1,2);
            envelope_speech_fil=freqfiltbp(envelope_speech',freqFilt,fs_new,1,2);
            %envelope_heard=envelope_heard(1:size(envelope_speech_fil,2));
            
            temp_1=hilbert(envelope_heard);
            temp_2=hilbert(envelope_speech_fil);
            
            phi_2=(angle(temp_2));
            phi_1=angle(temp_1);
            
            tmp=min(length(phi_1),length(phi_2));
            %tmp=10*100;
            phase_diff=wrapToPi(phi_1(1:tmp)-phi_2(1:tmp));
            
            %%% Calculates the PLV
            nT=round(fs_new*T);
            nshift=round(100*shift);
            n_ant=1;
            i=1;
            while (n_ant+nT)<length(phase_diff)
                PLV(i)=abs(sum(exp(1i*phase_diff(n_ant:n_ant+nT))))/nT;
                time(i)=i*shift+T/2;
                n_ant=n_ant+nshift;
                i=i+1;
            end
            synch(iBlock)=mean(PLV);
            
            
            if visualizar==1
                subplot(2,2,[3 4])
                plot(time, PLV);
                title(['PLV:' num2str(synch(iBlock))], 'FontSize', 16);
%                 subplot(2,2,1)
%                 hist(wrapToPi(phase_diff))
%                 xlim([-pi pi]);
%                 title([subject(3:end) ' mPHI=' num2str(mean(phase_diff),'%.2f')], 'FontSize', 18);
                waitforbuttonpress
                close all
            end

            clear PLV
        end
        clear file
    end
    save([folderOut subject '.mat'],'synch');
end


