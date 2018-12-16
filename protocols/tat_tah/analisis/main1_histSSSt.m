clear all
close all
folderIn=['Results/'];
subjects=dir([folderIn '*.mat']);
subjects={subjects.name};
addpath('extraScripts/')
nBins=9;
     
for iSub=1:length(subjects)
        load([folderIn subjects{iSub}]);
        names{iSub}=subjects{iSub}(1:end-4)
        plv(iSub,:)=synch;
end

%%%%% Keep just AV
plvMean=mean(plv)
histogram(plvMean,nBins)
set(gca, 'FontSize',20)
xlim([min(plvMean)-0.01 max(plvMean)+0.01]);




