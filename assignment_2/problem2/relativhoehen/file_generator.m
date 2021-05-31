Int =  0
index = 0
HightElements = 0
fileID = fopen('93_Scheuereck_111_umring_12_A_Laub_Prakrikum2_1.ascii','w');
fprintf(fileID, 'Seg Int Puls MidToFirst\n\n');
for k = 1:100
	% Create a mat filename, and load it into a structure called matData
	matFileName = sprintf('93_Scheuereck_111_umring_12_LAUB_Seg9%d.ascii', k)
    if exist(matFileName, 'file')
        
            fid = importdata(matFileName)
            
            MaxTreeHight = max(fid.data(:,3))
            SegmentHight = MaxTreeHight/10
                    
            DataOfArray = fid.data(:,3:6)
            
            DataOfArray = DataOfArray(DataOfArray(:,1)>(MaxTreeHight - SegmentHight*2),:)
            
            
            %SinglePointsArray =  DataOfArray(DataOfArray(:,4)==0,:)
            FirstPointArray = DataOfArray(DataOfArray(:,4)==1,:)
            MiddlePointArray = DataOfArray(DataOfArray(:,4)==2,:)
            %LastPointArray = DataOfArray(DataOfArray(:,4)==3,:)
            
            %MeanSinglePoint = mean(SinglePointsArray(:,3))
            MeanIntFirstPoint = mean(FirstPointArray(:,2))
            MeanPulsFirstPoint = mean(FirstPointArray(:,3))
            
            NumOfFirstPoints = size(FirstPointArray ,1)
            NumOfMiddlePoints = size(MiddlePointArray,1)
            
            MeanMiddleToFirst = (NumOfMiddlePoints/NumOfFirstPoints)
            
                       
            Segment = sprintf('09%d', k)
            % fprintf(fileID,'%s %f %f %f\n',Segment,MeanIntFirstPoint,MeanPulsFirstPoint, MeanMiddleToFirst );
    
	end
end
fclose(fileID);
Int =  0
index = 0
HightElements = 0
fileID = fopen('93_Scheuereck_111_umring_12_A_Nadel_Prakrikum2_1.ascii','w');
% fprintf(fileID, 'Seg Int Puls MidToFirst\n\n');
for k = 1:100
	% Create a mat filename, and load it into a structure called matData
	matFileName = sprintf('93_Scheuereck_111_umring_12_NADEL_Seg9%d.ascii', k)
    if exist(matFileName, 'file')
    fid = importdata(matFileName)

            fid = importdata(matFileName)
            
            MaxTreeHight = max(fid.data(:,3))
            SegmentHight = MaxTreeHight/10
                    
            DataOfArray = fid.data(:,3:6)
            
            DataOfArray = DataOfArray(DataOfArray(:,1) > (MaxTreeHight - SegmentHight*2),:)
            
            %SinglePointsArray =  DataOfArray(DataOfArray(:,4)==0,:)
            FirstPointArray = DataOfArray(DataOfArray(:,4)==1,:)
            MiddlePointArray = DataOfArray(DataOfArray(:,4)==2,:)
            %LastPointArray = DataOfArray(DataOfArray(:,4)==3,:)
            
            %MeanSinglePoint = mean(SinglePointsArray(:,3))
            MeanIntFirstPoint = mean(FirstPointArray(:,2))
            MeanPulsFirstPoint = mean(FirstPointArray(:,3))
            
            NumOfFirstPoints = size(FirstPointArray ,1)
            NumOfMiddlePoints = size(MiddlePointArray,1)
            
            MeanMiddleToFirst = (NumOfMiddlePoints/NumOfFirstPoints)
            
                       
            Segment = sprintf('09%d', k)
            % fprintf(fileID,'%s %f %f %f\n',Segment,MeanIntFirstPoint, MeanPulsFirstPoint, MeanMiddleToFirst );
    
	end
end
fclose(fileID);
