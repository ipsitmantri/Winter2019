function [binary_mask,color_mask]=Untitled2(filename)
im_file=strcat('C:\Users\mmkip\Desktop\MONUSEG\MoNuSegTrainingData\MoNuSeg Training Data\Tissue Images\',filename,'.png');

xml_file=strcat('C:\Users\mmkip\Desktop\MONUSEG\MoNuSegTrainingData\MoNuSeg Training Data\Annotations\',filename,'.xml'); 
 
xDoc = xmlread(xml_file);
Regions=xDoc.getElementsByTagName('Region'); % get a list of all the region tags
for regioni = 0:Regions.getLength-1
    Region=Regions.item(regioni);  % for each region tag
 
    %get a list of all the vertexes (which are in order)
    verticies=Region.getElementsByTagName('Vertex');
    xy{regioni+1}=zeros(verticies.getLength-1,2); %allocate space for them
    for vertexi = 0:verticies.getLength-1 %iterate through all verticies
        %get the x value of that vertex
        x=str2double(verticies.item(vertexi).getAttribute('X'));
       
        %get the y value of that vertex
        y=str2double(verticies.item(vertexi).getAttribute('Y'));
        xy{regioni+1}(vertexi+1,:)=[x,y]; % finally save them into the array
    end
   
end
im_info=imfinfo(im_file);

 
nrow=im_info.Height;
ncol=im_info.Width;
binary_mask=zeros(nrow,ncol); %pre-allocate a mask
color_mask = zeros(nrow,ncol,3);
%mask_final = [];
for zz=1:length(xy) %for each region
    fprintf('Processing object # %d \n',zz);
    smaller_x=xy{zz}(:,1); 
    smaller_y=xy{zz}(:,2);
   
    %make a mask and add it to the current mask
    %this addition makes it obvious when more than 1 layer overlap each
    %other, can be changed to simply an OR depending on application.
    polygon = poly2mask(smaller_x,smaller_y,nrow,ncol);
    binary_mask=binary_mask+zz*(1-min(1,binary_mask)).*polygon;%
    color_mask = color_mask + cat(3, rand*polygon, rand*polygon,rand*polygon);
    %binary mask for all objects
    %imshow(ditance_transform)
end

figure;imshow(binary_mask)
figure;imshow(color_mask)

