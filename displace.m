% % %  clear all; 
% % %  [x,y] = meshgrid(1:128,1:128);
% % %  [m,n] = meshgrid(-64:63,-64:63);
% % %  m = m/64*pi; n = n/64*pi;
% % % g1 = fspecial('gaussian',128, 5);
% % % g1 = g1./max(g1(:));
% % % 
% % % g2 = fspecial('gaussian',128, 5);
% % % g2 = g2./max(g2(:));gthis = g2; 
% % % 
% % % gthisout = interp2([gthis gthis gthis;...
% % %      gthis gthis gthis;...
% % %      gthis gthis gthis],142+x,142+y,'bilinear');
% % %  g2 = gthisout;
% % % subplot(121); contour(g1);colorbar;
% % % hold on; contour(g2,'r');hold off;
clear all;
[x,y] = meshgrid(1:32,1:32);
[m,n] = meshgrid(-16:15,-16:15);
m = m/16*pi; n = n/16*pi;
load('../weather/ThreeModeData.mat');
N = size(soltsav,3);
dat = zeros(N-1, 324);
%j = 600;
for j = 1:N-1
g1 = soltsav(:, :,j);
g2 = soltsav(:, :, j+1);

%input
display(j);
gconf = zeros(32);

gthis = g1;
lap = (abs(m)+abs(n)).^0.25; % Sparsity
glap = -exp(-(lap/pi*4))*(4/pi)^2;
qx = zeros(32); qy = qx;
for i = 1:500
 %display(i);
 gthisout = interp2([gthis gthis gthis;...
     gthis gthis gthis;...
     gthis gthis gthis],32+x-qx,32+y-qy,'bilinear');
 gthisout(isnan(gthisout)) = gthis(isnan(gthisout));
 [gx,gy]=gradient(gthisout);
 qx = qx+(ifft2(fft2(gx.*(g2 - gthisout)).*fftshift(glap)));
 qy = qy+(ifft2(fft2(gy.*(g2 - gthisout)).*fftshift(glap)));
 gconf = max(gconf,((abs(gthisout))));
 %display(max(abs(qx(:))));
 %display(max(abs(qy(:))));

 %subplot(131);
 %imagesc(gthisout);subplot(132);imagesc(g1);subplot(133);imagesc(g2);colorbar;drawnow;
 end
gconf = gconf>mean(gconf(:));
 gthisout = interp2([gthis gthis gthis;...
     gthis gthis gthis;...
     gthis gthis gthis],32+x-qx,32+y-qy,'bilinear');
%subplot(121); 
%contour(g1,'k','LineWidth',1);
%hold on;contour(gthisout,'k--','LineWidth',2);
 %contour(g2,'r');hold off;axis('square');
%legend('initial','truth','final');
%subplot(122);
%quiver(gconf.*qx,gconf.*qy);
%plot(curl(gconf,gconf, qx,qy));
%display(gconf);
%contour(gradient(gconf.*qx+ gconf.*qy));
qMag = qx.^2 + qy.^2;
subplot(121);
imagesc(log(qMag));subplot(122); imagesc(gconf.*log(qMag));drawnow;

[featuresX, bX] = extractHOGFeatures(gconf.*log(qMag));
%[featuresY, bY] = extractHOGFeatures(qy);
imagesc(qMag);
dat(j, :) = featuresX;
display(featuresX);
%X = [dat(j, :), dat(j+1, :)];
%displays(size(x));
%plot(b);

%legend('displacement field'); axis('tight');axis('square');drawnow;
end
display(squareform(pdist(dat, dat)), 'cosine');
k=3;
[idx, C, sumd, D] = kmeans(dat, k);
display(D);



  