%%% Three steps to analysing the coupling between the spoken syllables and
%%% the listened ones
clear all
close all
data_folder='~/Desktop/analisis/raw_data/AudioStim/';

visualizar=1;
fileName='stimulus';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% STEP 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file_name=[data_folder fileName '.wav'];
[signal_1, Fs]=audioread(file_name);
signal_1=signal_1-mean(signal_1);

hi=hilbert(signal_1);
envelope=abs(hi);

%%% Smooths and
n_average=0.01*Fs;
coeff= ones(1, n_average)/n_average;
envelope= filtfilt(coeff, 1, envelope);
envelope=resample(envelope, 100,Fs);

%%% Solo si lo quiero ver shows the envelope with the original signal
if visualizar==1
    subplot(1,2,1)
    hold on
    time=(1:length(signal_1))./Fs;
    plot(time, signal_1, 'k');
    time=(1:length(envelope))./100;
    plot(time, envelope, 'r', 'LineWidth',2);
    hold off
    xlim([20 30])
    
    %%%%%%%%%%%%%%
    %%% Computing the spectral content of the envelope from the speech
    %%%%%%%%%%%%%%
    envelope=envelope-mean(envelope);
    L=size(envelope,1);
    NFFT = 2^nextpow2(L); % Next power of 2 from length of y
    Y = fft(envelope,NFFT)/L;
    f = 100/2*linspace(0,1,NFFT/2+1);
    %%% Frequencies below 10 Hz
    f=f(f<10);
    Y=Y(1:size(f,2));
    amp_fft= abs(Y).^2;
    
    df=f(3)-f(2);
    n_average=round(0.2/df);
    coeff= ones(1, n_average)/n_average;
    smooth_fft= filter(coeff, 1, amp_fft);
    smooth_fft=smooth_fft/max(smooth_fft);
    
    subplot(1,2,2)
    plot(f, smooth_fft,'k', 'LineWidth', 3);
    set(gca,'FontSize',18);
    
    
    save([data_folder 'envelope_' fileName '.mat'], 'envelope');
end