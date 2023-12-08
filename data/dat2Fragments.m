
%
% This m file is used to convert 1024-byte fragments stored in dat file format to csv (comma-separated integer byte values) 512-byte fragments.
% It was designed to be used for datasets:
% [2]	A. Khodadadi and M. Teimouri, “Dataset for file fragment classification of audio file formats,” BMC Res. Notes, vol. 12, no. 1, p. 819, Dec. 2019, doi: 10.1186/s13104-019-4856-1.
% [3]	N. Sadeghi, M. Fahiminia, and M. Teimouri, “Dataset for file fragment classification of video file formats,” BMC Res. Notes, vol. 13, no. 1, p. 213, Apr. 2020, doi: 10.1186/s13104-020-05037-x.
% [4]	F. Mansouri Hanis and M. Teimouri, “Dataset for file fragment classification of textual file formats,” BMC Res. Notes, vol. 12, no. 1, p. 801, Dec. 2019, doi: 10.1186/s13104-019-4837-4.
% [5]	R. Fakouri and M. Teimouri, “Dataset for file fragment classification of image file formats,” BMC Res. Notes, vol. 12, no. 1, p. 774, Nov. 2019, doi: 10.1186/s13104-019-4812-0.
% Those datasets shuld first be imported using the LoadDat.m script (stored here for backup and originall available together with the four referenced datasets)
%
% Output files of this script are:
% fragments_single_512b (fragments)
% classes_single_512b (classes)
%

%LoadDat.m

x = size(Dataset);
dSets = x(1,2);
for j = 1:dSets
	class = Dataset{1,j}
	dataSetDimenstions = size(Dataset{2,j});
	dataSetRows = dataSetDimenstions(2);
	for i = 1:dataSetRows

		% Fragments are stored as 5x1024 fragments
		frags_1024_matrix = cell2mat(reshape(Dataset{2,j}(i).Fragments,5,1));
		
		% Convert 5x1024 fragments to 5x512 fragments
		r = randi([1 512],1,5);
		for k = 1:5
			frags_512_single(k,:) = frags_1024_matrix(k,r(k):r(k)+511);
		end
		class_vector_single = cell2mat(repmat({class(1,1:4)}, 5, 1));
		
		% Write to file
		dlmwrite('fragments_single_512b.csv',frags_512_single,'-append');
		dlmwrite('classes_single_512b.csv',class_vector_single,'-append');

		
	end
end