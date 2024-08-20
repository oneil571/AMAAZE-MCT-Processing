#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 23:41:02 2024

@author: rileywilde
"""


import os 
import numpy as np
import matplotlib.pyplot as plt

from amaazetools.trimesh import __virtual_goniometer__ as vgo
from mayavi import mlab


#where segmented meshes are kept:
path = '/Users/rileywilde/Risa_Stuff/SAMS/teeth_sep22/Default/HecateVertex/NumSegments_5/NumEigs_2/segment';

#originals... just so I can grab the names
name_path = '/Users/rileywilde/Risa_Stuff/ariaDNE_code-master/alignedMeshes2';




ddd = os.listdir(name_path)

ddd(1:2)=[];

figure;

for risayouredrivingmenuts = [1,2]
subplot(1,2,risayouredrivingmenuts)
    

names = {};
for i = 1:length(ddd)
    n = ddd(i).name;
    if length(n)>4
        names{i} = n(1:end-4);
    end
end 


OUTPUT = zeros(length(names),9*5);

for i = 1%:length(names)
    rowi = [];
    
    %figure;
    hold on;
    for j = 1:5
        namei = strcat(names{i},'_seg',string(j),'.ply');
        [v,f]= read_ply(strcat(path,'/',namei));
        t=triangulation(f,v);
        vn= vertexNormal(t);
        [theta,n1,n2,C,vs] = VirtualGoniometer(v,vn);
        n1=n1';
        n2=n2';     
        proj1 = [n1(1:2),0];
        proj1 = proj1/sqrt(sum(proj1.^2)+1e-6);
        proj2 = [n2(1:2),0];
        proj2 = proj2/sqrt(sum(proj2.^2)+1e-6);
        
        av1 = mean(v(C==1,:)); %for segmentation
        av2 = mean(v(C==2,:));
        
        a1 = acosd(sum(n1.*proj1));
        a2 = acosd(sum(n2.*proj2));
        
        av_vn1 = mean(vn(C==1,:));
        av_vn2 = mean(vn(C==2,:));
        
        
        if j==4
            qqqq = [1132,365,829,999,124,216,764,767,141];
            C(qqqq) = 2;
        end
        
        t = triangulation(f,v);
        E = t.edges;
        
        inc2 = C==2;
        
        e2 = E( prod(inc2(E),2)==1,:);
        G = graph(e2(:,1),e2(:,2));
        [b,bsz] = conncomp(G);
        
        
        if size(bsz(bsz>1))>1
            disp(j)
        end
            
        
        
        if av1(3)>av2(3)
            rowi = cat(2,rowi,[theta,a1,a2,n1,n2]);
            svi_color_plot(t,C)
            axis on
            xlabel('x');ylabel('y');zlabel('z');
            %title(strcat(namei,string(i),',',string(j),'reg'))
            %scat([av1;av2],[.2*av_vn1;.2*av_vn2])
            scat([av1;av2],[.2*n1;.2*n2])
            view(90,0)
            
            pause(6)
        else 
            rowi = cat(2,rowi,[theta,a2,a1,n2,n1]);
            svi_color_plot(t,1-C)
            axis on
            xlabel('x');ylabel('y');zlabel('z');
            %title(strcat(namei,string(i),',',string(j),'flip'))
            %scat([av2;av1],[.2*av_vn2;.2*av_vn1])
            scat([av2;av1],[.2*n2;.2*n1])
            view(90,0)
            
            pause(6)
        end
    end
    %camlight
    text(-.5,-.4,-.15,'A',FontSize=150,FontName='Times')
    hold off;
    OUTPUT(i,:) = rowi;
end


end

function [theta,n1,n2,C1,C2] = VGO_RP(P,N,r,ID)
%Virtual goniometer based on projection clustering
%Input:
%   P = nx3 numpy array of vertices of points
%   N = nx3 array of vertex normals
%   r = radius of search
%   ID = integer indices or length(P) logical array of pts to do   
%   Can also use N as face normals, and P as face centroids
%Output:
%   theta = Angle, n*1
%   n1,n2 = n*3 Normal vectors between two patches (theta=angle(n1,n2))
%   C = Clusters (C=1 and C=2 are the two detected clusters)
%RCWO
if nargin < 4 
    ID = 1:length(P);
end

if max(ID) ~= 1  || length(ID)~=size(P,1)
    ind = ID;
else
    ind = find(ID);
end


n1 = zeros(size(P));
n2 = n1;
theta = n1(:,1);
C1 = cell(length(P),1);
C2 = C1;

idx = rangesearch(P,P(ind,:),r);

for i = 1:length(ind)
    
   [theta(ind(i),:),n1(ind(i),:),n2(ind(i),:),C,v] = VirtualGoniometer(P(idx{i},:),N(idx{i},:));
   %what is v?
   C1{i} = idx{i}(C==1);
   C2{i} = idx{i}(C==2);
   
   %{
   %C = RP1D_clustering(N(idx{i},:),100);
   
   P1 = P(idx{i}(C==1),:);
   
   c1 = pca(P1,'Economy',false);
   n1(ind(i),:) = c1(:,3);
   s1 = mean(N(idx{i}(C==1),:));
   if n1(ind(i),:)*s1' < 0
      n1(ind(i),:) = -n1(ind(i),:);
   end

   P2 = P(idx{i}(C==2),:);
   c2 = pca(P2,'Economy',false);
   
   n2(ind(i),:) = c2(:,3);
   
   s2 = mean(N(idx{i}(C==2),:));
   if n2(ind(i),:)*s2' < 0
      n2(ind(i),:) = -n2(ind(i),:);
   end
   
   C1{i} = idx{i}(C==1);
   C2{i} = idx{i}(C==2);
   
   theta(ind(i)) = acos(dot(n1(ind(i),:),n2(ind(i),:),2))*180/pi;
 %}
end

end
%Faster Virtual goniometer
%Input:
%   P = nx3 numpy array of vertices of points in patch
%   N = nx3 array of vertex normals
%   Can also use N as face normals, and P as face centroids
%Output:
%   theta = Angle
%   n1,n2 = Normal vectors between two patches (theta=angle(n1,n2))
%   C = Clusters (C=1 and C=2 are the two detected clusters)
function [theta,n1,n2,C,v] = VirtualGoniometer(P,N)

   [N_pca,D] = eig(N'*N);   %Eigenvalues/eigenvectors of covariance matrix
  
   %N_pca = pca(N);
   
   N1 = N_pca(:,1)';

   %P_pca = pca(P);
   %N2 = P_pca(:,3)';
   N2 = mean(N,1);
   N2 = N2/norm(N2);
   
   try
   v = cross(N1,N2);
   
   catch
       N2
   end
   v = v/norm(v);
   x = sum(N.*v,2);
   [w,m] = withness(x);
   C = zeros(length(x),1);
   C(x>m) = 1;
   C(x<=m) = 2;

   %C = RP1D_clustering(N,100);
   %
   P1 = P(C==1,:);
                   c1 = pca(P1,'Economy',false); %so we always have vector # 3
   n1 = c1(:,3);
   s1 = mean(N(C==1,:),1);
   if dot(n1,s1) < 0
      n1 = -n1;
   end

   P2 = P(C==2,:);
   c2 = pca(P2,'Economy',false);
   n2 = c2(:,3);
   s2 = mean(N(C==2,:),1);
   if dot(n2,s2) < 0
      n2 = -n2;
   end
   theta = acosd(dot(n1,n2));
   
end

function [w,m] = withness(x)

   x = sort(x);
   sigma = std(x);
   n = length(x);
   v = zeros(n,1);
   for i=1:n
      x1 = x(1:i);
      x2 = x((i+1):n);
      m1 = mean(x1);
      m2 = mean(x2);
      v(i) = (sum((x1-m1).^2) + sum((x2-m2).^2))/(sigma^2*n);
   end
   [w,ind] = min(v);
   m = x(ind);

end


function svi_color_plot(T,vol,v)
%plots input VOL volume over surface T triangulation with normalization
Pts = T.Points;
Tri = T.ConnectivityList;

%normalizing:
nvol = vol - min(vol);
nvol = nvol./(max(nvol));
nvol = histeq(nvol);
nvol = nvol*(2/3);
mvol = 1-nvol; 
nvol = max(nvol)-nvol;

%figure
%plotsurf(T);
%trisurf(T);
hold on;
patch('Faces',Tri,'Vertices',Pts,'FaceVertexCData',nvol,'FaceColor','interp','EdgeColor','none');
daspect([1 1 1])
axis tight

if nargin == 3
view(v)
end 

%camlight
axis off

colormap jet
end




function scat(pts,vec,color)

if nargin < 2 % regular
    if size(pts,2) ==2
        scatter(pts(:,1),pts(:,2),'filled')
        xlabel('x')
        ylabel('y')
    else 
        scatter3(pts(:,1),pts(:,2),pts(:,3),'filled')
        xlabel('x')
        ylabel('y')
        zlabel('z')
        daspect([1 1 1])
    end 
else 
    if isempty(vec) % use color
        if size(pts,2) ==2
            scatter(pts(:,1),pts(:,2),10,color,'filled')
            xlabel('x')
            ylabel('y')
        else 
            scatter3(pts(:,1),pts(:,2),pts(:,3),10,color,'filled')
            xlabel('x')
            ylabel('y')
            zlabel('z')
            daspect([1 1 1])
        end  
    else % quiver
        if nargin == 3 %color vectors, oh bloody hell
            hold on;
            c = color./max(color);
            c = c-min(c);
            c = .85*[c,ones(length(c),2)];
            c = hsv2rgb(c);
           
            if size(pts,2) ==2
                for i = 1:length(pts)
                quiver(pts(i,1),pts(i,2),vec(i,1),vec(i,2),'AutoScale','off','Color',c(i,:))
                end
                xlabel('x')
                ylabel('y')
                daspect([1 1 1])
            else 
                for i = 1:length(pts)
                quiver3(pts(i,1),pts(i,2),pts(i,3),vec(i,1),vec(i,2),vec(i,3),'AutoScale','off','Color',c(i,:))
                end
                xlabel('x')
                ylabel('y')
                zlabel('z')
                daspect([1 1 1])
            end 
            
        else 
            if size(pts,2) ==2
                quiver(pts(:,1),pts(:,2),vec(:,1),vec(:,2),'AutoScale','off')
                xlabel('x')
                ylabel('y')
            else 
                quiver3(pts(:,1),pts(:,2),pts(:,3),vec(:,1),vec(:,2),vec(:,3),'AutoScale','off','Color','black','linewidth',3)
                xlabel('x')
                ylabel('y')
                zlabel('z')
                daspect([1 1 1])
            end 
            
        end 
    end 
end 

end
