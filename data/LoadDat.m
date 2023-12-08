%% This script takes some data set files in generic binary data format and reads the fragments
%   File fragments are written in workspace as variable Dataset

%% Initialization
clc

%% Get the name of the binary files for reading fragments
[FileName,PathName] = uigetfile('a.dat','Dataset in Generic Binary File Format','MultiSelect', 'on');
if isequal(FileName,0)
    return;
end

if ischar(FileName)
    FileName = {FileName};
end

%% Read fragments into Dataset
%   Dataset is a 2xN cell array.
%       Dataset{1,i}: Corresponding filename
%       Dataset{2,i}: A structure with the following field:
%           Fragments: A cell vector; The content of each individual cell is a fragment
%               All fragments of Dataset{2,i} are taken from a single file

N = length(FileName);
Dataset = cell(2,N);

for j=1:N
    
    % Open file
    fileID = fopen([PathName FileName{j}],'r');
    str = FileName{j};
    str(strfind(lower(str), '.dat'):end) = [];
    Dataset{1,j} = str;
    
    % Length of file
    fseek(fileID, 0, 'eof');
    FileLength = ftell(fileID);
    fseek(fileID, 0, 'bof');
    
    % Read file fragments
    cnt = 0;
    L = 0;
    while ~feof(fileID)
        
        % Read fields
        FileID = fread(fileID,1,'uint64=>double',0,'b');
        FragmentID = fread(fileID,1,'uint64=>double',0,'b');
        L0 = fread(fileID,1,'uint64=>double',0,'b');
        Frg = fread(fileID,L0,'uint8=>double',0,'b');
        cnt = cnt+3*8+L0;
        
        % Fill Dataset
        Dataset{2,j}(FileID).Fragments{FragmentID} = Frg';
        L = L+1;
        
        % Show Progress
        %fprintf('Progress: %d%% \n',round(100*cnt/FileLength));
        
        % Break loop
        if cnt>=FileLength
            break;
        end
    end
    
    % Close file
    fclose(fileID);
    
    fprintf('File %d from %d: Total number of fragments: %d \n',j,N,L);
end
