function [] = visLogisticRegression(beta,x,y, label, correct)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% visualize the model
% prepare a grid of points to evaluate the model at

% visualize the data
figure; hold on;
h1 = scatter(x(label==0),y(label==0),50,'k','filled');  % black dots for 0
h2 = scatter(x(label==1),y(label==1),50,'w','filled');  % white dots for 1
set([h1 h2], 'MarkerEdgeColor' ,[.5 .5 .5]);        
% outline dots in gray
legend([h1 h2],{'y==0'  'y==1'},'Location','NorthEastOutside');
xlabel('x');
ylabel('y');

ax = axis;
xvals = linspace(ax(1),ax(2),100);
yvals = linspace(ax(3),ax(4),100);
[xx,yy] = meshgrid(xvals,yvals);
% construct regressor matrix
X = [xx(:) yy(:)];
%X(:,end+1) = 1;

Y_Test = glmval(beta,X,'logit');

% evaluate model at the points (but don't perform the final thresholding)
outputimage = reshape(Y_Test,[length(yvals) length(xvals)]);
% visualize the image (the default coordinate system for images
% is 1:N where N is the number of pixels along each dimension.
% we have to move the image to the proper position; we
% accomplish this by setting XData and YData.)
h3 = imagesc(outputimage,[0 1]);  
% the range of the logistic function is 0 to 1
set(h3,'XData',xvals,'YData',yvals);
colormap(hot);
colorbar;
% visualize the decision boundary associated with the model
% by computing the 0.5-contour of the image
[c4,h4] = contour(xvals,yvals,outputimage,[.5 .5]);
set(h4,'LineWidth',3,'LineColor',[0 0 1]);  
% make the line thick and blue
% send the image to the bottom so that we can see the data points
uistack(h3,'bottom');
% send the contour to the top
uistack(h4,'top');
% restore the original axis range
axis(ax);
% report the accuracy of the model in the title
title(sprintf( 'Classification accuracy is %.1f%%',correct));
end

