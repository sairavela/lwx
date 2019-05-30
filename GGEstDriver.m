%%Sai Ravela, (C) 2017
%GGEstDriver
%filename='Counties/Birch Hill.xlsx';
%xlsread(filename, 'E28:E819')
y = [[    0.           567.05561048  5634.01898604  4388.99340126 4787.79668335  4081.62433019  2799.8337528   3558.51895584 622.97112745   813.1699974 ]
 [  567.05561048     0.          2233.72940413  1719.71706018 2234.99190479  1462.82247546  1677.44892504  1841.34946581 48.39350711   111.65114048]
 [ 5634.01898604  2233.72940413     0.           103.73730307 29.42948405   233.26461736   273.14242671    78.38563677 2823.11805381  2344.36869517]
 [ 4388.99340126  1719.71706018   103.73730307     0.           272.46646177 83.5027576   1698.51107122   346.45601777  2031.88364981 1823.58961606]
 [ 4787.79668335  2234.99190479    29.42948405   272.46646177     0. 229.18110677   855.94169192    22.25452761  2635.99911761 2604.43332602]
 [ 4081.62433019  1462.82247546   233.26461736    83.5027576    229.18110677 0.           745.2512391    223.89207897  1714.89517321 1262.51026337]
 [ 2799.8337528   1677.44892504   273.14242671  1698.51107122 855.94169192   745.2512391      0.           586.57945142 1920.08018907  2919.38459323]
 [ 3558.51895584  1841.34946581    78.38563677   346.45601777 22.25452761   223.89207897   586.57945142     0.          2114.77768139 2359.69474534]
 [  622.97112745    48.39350711  2823.11805381  2031.88364981 2635.99911761  1714.89517321  1920.08018907  2114.77768139     0. 27.3313797 ]
 [  813.1699974    111.65114048  2344.36869517  1823.58961606 2604.43332602  1262.51026337  2919.38459323  2359.69474534    27.3313797 0.        ]];
disp(size(y));
%m = horzcat(xlsread(filename, 'D2:D793'),xlsread(filename, 'G2:G793'),xlsread(filename, 'J2:U793'))';%,xlsread(filename, 'M3170:O3961'),xlsread(filename, 'Q3170:S3961'),xlsread(filename, 'U3170:U3961'),xlsread(filename, 'W3170:Y3961'),xlsread(filename, 'AA3170:AA3961')).';
%disp("Size of m is:" + size(m))
%x = -5:5; g = exp(-x.*x/2/10);gg=toeplitz((g));[u,s,v]=svd(gg');
%disp(x)
%disp(size(u)+ " " + size(s))
%samps = m./std(m,[],2);

    S= y;%cov(samps');
    lopt=1100 ;
%disp(size())
GGM = GGMEst(S,lopt);
GGM(abs(GGM) < 1e-4) = 0;
disp(GGM);


%disp(aicbic(mlecov({'Temp Diff' 'NAO' 'EA' 'WP' 'PNA' 'EA/WR' 'SCA' 'POL' 'Expl' 'ENSO' 'AO' 'PDO'}, GGM, 'pdf', pdf)), 12^2);
%disp(aic(tfest(GGM, 12^2)));
G=graph(y,'OmitSelfLoops', 'upper');
%T = minspantree(G,'Type','forest')
%Birch hill changed bc outlier, not sure if I know where it is
x = [-73.7562 -72.5199 -72.125 -71.1137 -71.0589 -71.3489 -70.5134 -71.4128 -72.2164 -71.8023];
y = [42.6526 42.3732 42.632 42.2123 42.3601 42.4604 41.3890 41.8240 42.6441 42.2626];
LWidths = 5*G.Edges.Weight/max(G.Edges.Weight);
%figure(1); plot(lamp,[trc' sam' abs(trc'-sam')]);
%figure(1);plot(T,'LineWidth',5*T.Edges.Weight/max(T.Edges.Weight))
G.Nodes.Name = {'Albany' 'Amherst' 'Birch Hill' 'Blue Hill' 'Boston' 'Concord' 'Edgartown' 'Providence' 'Tully Lake' 'Worcester'}';
figure(1);plot(G, 'XData', x,'YData', y ,'LineWidth',5*G.Edges.Weight/max(G.Edges.Weight))
set(gca,'xtick',[],'ytick',[])
title('Gaussian Graphical Model of Indicators on Amherst Temperature Diff at lopt = 0.15')