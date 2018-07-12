import os

folder = '/Volumes/Server/MORPHLAB/Projects/shepard/tones_95dB'

pathiter = (os.path.join(root, filename)
            for root, _, filenames in os.walk(folder)
                for filename in filenames)
for path in pathiter:
    newname =  path.replace('D', 'Eb')
    if newname != path:
        os.rename(path,newname)
