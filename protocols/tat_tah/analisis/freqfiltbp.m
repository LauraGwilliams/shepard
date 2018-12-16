function [ dataOut,freqfilt,f ] = freqfiltbp(dataIn,cutoffs,Fs,numdev,dim)
% freqfiltbp filters the signal by transforming to the frequency domain and
% applying a gaussian of a certain width.
%   Makes a gaussian frequency filter
%   dataIn = time-series data to be filtered
%   cutoffs = lower and upper limit of filter
%   Fs = sampling rate
%   numdev = # of standard deviations each cutoff should be from the center
%           frequency
%   dim = dimension to act along
%   
%   dataOut = filtered data
%   freqfilt = frequency filter used
%   f = frequency axis to use to plot freqfilt


%   get frequency axis
f = (1:size(dataIn,dim)) - 1;
f = f./length(f);
f = f.*Fs;

% build frequency filter
if ~mod(length(dataIn),2)
    freqfilt = normpdf(f(1:length(f)/2),mean(cutoffs),diff(cutoffs)/(2*numdev));
    freqfilt = freqfilt([1:length(f)/2 length(f)/2:-1:1]);
else
    freqfilt = normpdf(f(1:round(length(f)/2)),mean(cutoffs),diff(cutoffs)/(2*numdev));
    freqfilt = freqfilt([1:end end-1:-1:1]);
end
% fit size to the matrix of dataIn
freqfilt = shiftdim(freqfilt,2-dim);
siz = size(dataIn);
siz(dim) = 1;
freqfilt = repmat(freqfilt,siz);

% apply filter to frequency data
fqdata = fft(dataIn,[],dim);
fqfiltdata = fqdata.*freqfilt;

% convert back to time signal
dataOut = ifft(fqfiltdata,[],dim,'symmetric');
end