%% DISCLAIMER AND CONDITIONS FOR USE:
%     This software is distributed under the terms of the GNU General Public
%     License v3, dated 2007/06/29 (see http://www.gnu.org/licenses/gpl.html).
%     Use of this software is at the user's OWN RISK. Functionality is not
%     guaranteed by creator nor modifier(s), if any. This software may be freely
%     copied and distributed. The original header MUST stay part of the file and
%     modifications MUST be reported in the 'MODIFICATION HISTORY'-section,
%     including the modification date and the name of the modifier.

%%
% Detrend_Filter.m
% Remove the lower order trends and low frequency components (noise) from
% the fMRI time-series data

% Inputs
% fMRI: time-series data (num voxels x num time points)
% Fs: sampling frequency

% Outputs
% Normalized_fMRI: data after detrend/filter/normalization steps

%%
function Normalized_fMRI = Detrend_Filter(fMRI,Fs)

% remove the nan values from fMRI because cannot have nan values for
% detrending/filtering
fMRI(isnan(fMRI(:,1)),:) = [];

% Initialize output vector
Normalized_fMRI = zeros(size(fMRI));

% Iterate over the number of voxels
% Pass in all time points for that voxel to detrend/filter functions
for kk=1:size(fMRI,1)
    
    % Detrend step
    Normalized_fMRI(kk,:) = amri_sig_detrend(fMRI(kk,:),3);
    
    % 0.01-0.1 bandpass filtering
    Normalized_fMRI(kk,:) = amri_sig_filtfft(Normalized_fMRI(kk,:), Fs, 0.01, 0.1);
end

% Normalize the data to have mean 0, stdev 1
Normalized_fMRI = amri_sig_std(Normalized_fMRI')';

end

%% amri_sig_detrend
% remove polynormial functions from the input time series
%
% Version 0.01

%%
function ots = amri_sig_detrend(its, polyorder)
if nargin<1
    eval('help amri_sig_detrend');
    return
end

if nargin<2
    polyorder=1;
end

polyorder = round(polyorder);
if polyorder<0
    polyorder=0;
end

[nr,nc]=size(its);
its=its(:);
its=its-mean(its);

if polyorder>0
    nt=length(its);
    
    poly=zeros(nt,polyorder+1);
    for i=1:polyorder+1
        poly(:,i)=(1:nt).^(i-1);
        poly(:,i)=poly(:,i)./norm(poly(:,i));
    end
    p=double(poly)\double(its);
    trend=double(poly)*p;
    ots=its-trend;
    
%     poly=zeros(nt,polyorder);
%     for i=1:polyorder
%         poly(:,i)=(1:nt).^i;
%         poly(:,i)=poly(:,i)./norm(poly(:,i));
%     end
%     ots=amri_sig_nvr(its,poly);
 
    ots=reshape(ots,nr,nc);
else
    ots=its;
end

end

%% 
% amri_sig_filtfft() - lowpass, highpass or bandpass filtering using a pair of forward 
%             and inverse fourier transform. 
%
% Usage
%   [ts_new]=amri_sig_filtfft(ts, fs, lowcut, highcut, revfilt, trans)
%
% Inputs
%   ts:      a discrete time series vector
%   fs:      sampling frequency of the time series
%   lowcut:  lowcutoff frequency (in Hz)
%   highcut: highcutoff frequency (in Hz)
%   revfilt: 0:band-pass; 1:band-stop {default: 0}
%   trans:   relative transition zone {default: 0.15}
%
% Output:
%   ts_new:  the filtered time series vector
%
% See also:
%  fft(),ifft()
%

function ts_new = amri_sig_filtfft(ts, fs, lowcut, highcut, revfilt, trans)

if nargin<1
    eval('help amri_sig_filtfft');
    return
end

if ~isvector(ts)
    printf('amri_sig_filtfft(): input data has to be a vector');
end

if nargin<2,fs=1;end                % default sampling frequency is 1 Hz, if not specified
if nargin<3,lowcut=NaN;end          % default lowcut is NaN, if not specified
if nargin<4,highcut=NaN;end         % default highcut is NaN, if not specified
if nargin<5,revfilt=0;end           % default revfilt=0: bandpass filter
if nargin<6,trans=0.15;end          % default relative trans of 0.15

[ts_size1, ts_size2] = size(ts);    % save the original dimension of the input signal
ts=ts(:);                           % convert the input into a column vector
npts = length(ts);                  % number of time points 
nfft = 2^nextpow2(npts);            % number of frequency points 

fv=fs/2*linspace(0,1,nfft/2+1);     % even-sized frequency vector from 0 to nyguist frequency
fres=(fv(end)-fv(1))/(nfft/2);      % frequency domain resolution
% fv=fs/2*linspace(0,1,nfft/2);     % even-sized frequency vector from 0 to nyguist frequency
% fres=(fv(end)-fv(1))/(nfft/2-1);  % frequency domain resolution


filter=ones(nfft,1);                % desired frequency response

% remove the linear trend 
ts_old = ts;                        
ts = detrend(ts_old,'linear');
trend  = ts_old - ts;

% design frequency domain filter
if (~isnan(lowcut)&&lowcut>0)&&...          % highpass
   (isnan(highcut)||highcut<=0)
   
    %          lowcut
    %              ----------- 
    %             /
    %            /
    %           /
    %-----------
    %    lowcut*(1-trans)
    
    idxl = round(lowcut/fres)+1;
    idxlmt = round(lowcut*(1-trans)/fres)+1;
    idxlmt = max([idxlmt,1]);
    filter(1:idxlmt)=0;
    filter(idxlmt:idxl)=0.5*(1+sin(-pi/2+linspace(0,pi,idxl-idxlmt+1)'));
    filter(nfft-idxl+1:nfft)=filter(idxl:-1:1);    

elseif (isnan(lowcut)||lowcut<=0)&&...      % lowpass
       (~isnan(highcut)&&highcut>0)
    
    %        highcut
    % ----------
    %           \
    %            \
    %             \
    %              -----------
    %              highcut*(1+trans)
    
    idxh=round(highcut/fres)+1;                                                                         
    idxhpt = round(highcut*(1+trans)/fres)+1;                                   
    filter(idxh:idxhpt)=0.5*(1+sin(pi/2+linspace(0,pi,idxhpt-idxh+1)'));
    filter(idxhpt:nfft/2)=0;
    filter(nfft/2+1:nfft-idxh+1)=filter(nfft/2:-1:idxh);
    
elseif lowcut>0&&highcut>0&&highcut>lowcut  
    if revfilt==0                           % bandpass (revfilt==0)
        
    %         lowcut   highcut
    %             -------
    %            /       \     transition = (highcut-lowcut)/2*trans
    %           /         \    center = (lowcut+highcut)/2;
    %          /           \
    %   -------             -----------
    % lowcut-transition  highcut+transition
    transition = (highcut-lowcut)/2*trans;
    idxl   = round(lowcut/fres)+1;
    idxlmt = round((lowcut-transition)/fres)+1;
    idxh   = round(highcut/fres)+1;
    idxhpt = round((highcut+transition)/fres)+1;
    idxl = max([idxlmt,1]);
    idxlmt = max([idxlmt,1]);
    idxh = min([nfft/2 idxh]);
    idxhpt = min([nfft/2 idxhpt]);
    filter(1:idxlmt)=0;
    filter(idxlmt:idxl)=0.5*(1+sin(-pi/2+linspace(0,pi,idxl-idxlmt+1)'));
    filter(idxh:idxhpt)=0.5*(1+sin(pi/2+linspace(0,pi,idxhpt-idxh+1)'));
    filter(idxhpt:nfft/2)=0;
    filter(nfft-idxl+1:nfft)=filter(idxl:-1:1);
    filter(nfft/2+1:nfft-idxh+1)=filter(nfft/2:-1:idxh);
    
    else                                    % bandstop (revfilt==1)
        
    % lowcut-transition  highcut+transition
    %   -------             -----------
    %          \           /  
    %           \         /    transition = (highcut-lowcut)/2*trans
    %            \       /     center = (lowcut+highcut)/2;
    %             -------
    %         lowcut   highcut
    
    
    transition = (highcut-lowcut)/2*trans;
    idxl   = round(lowcut/fres)+1;
    idxlmt = round((lowcut-transition)/fres)+1;
    idxh   = round(highcut/fres)+1;
    idxhpt = round((highcut+transition)/fres)+1;
    idxlmt = max([idxlmt,1]);
    idxlmt = max([idxlmt,1]);
    idxh = min([nfft/2 idxh]);
    idxhpt = min([nfft/2 idxhpt]);
    filter(idxlmt:idxl)=0.5*(1+sin(pi/2+linspace(0,pi,idxl-idxlmt+1)'));
    filter(idxl:idxh)=0;
    filter(idxh:idxhpt)=0.5*(1+sin(-pi/2+linspace(0,pi,idxl-idxlmt+1)'));
    filter(nfft-idxhpt+1:nfft-idxlmt+1)=filter(idxhpt:-1:idxlmt);
    
    end
    
else
    printf('amri_sig_filtfft(): error in lowcut and highcut setting');
end

X=fft(ts,nfft);                         % fft
ts_new = real(ifft(X.*filter,nfft));    % ifft
ts_new = ts_new(1:npts);                % tranc

% add back the linear trend
ts_new = ts_new + trend;

ts_new = reshape(ts_new,ts_size1,ts_size2);

end

%%
% amri_sig_std: standarize signal with zeros mean and standard variance
%
% Usage
%   Y = amri_sig_std(X)
%
% Inputs
%   X: vector or matrix. if X is a matrix, then standarize each column
%
% Outputs
%   Y: standarized signal
%
% Version
%   1.0

%%
function [X, m_x, std_x]= amri_sig_std(X,varargin)

if isvector(X)
   X = X(:); 
end

% defaults
standard = 1;% 1 means standarize,  0 means percentage, 2 means norm(X)=1, else just demean
epsilon = 1e-4;
std_x = [];
trans_flag = 0;
% Keywords
if size(varargin,2) >= 1 && ~isempty(varargin{1})
    standard = varargin{1};
end
if size(varargin,2) >= 2 && ~isempty(varargin{2})
    epsilon = varargin{2};
end
if size(varargin,2) >= 3 && ~isempty(varargin{3})
    trans_flag = varargin{3};
end


m_x = mean(X);
X = bsxfun(@minus,X,m_x);

if standard == 1
    std_x = std(X,0,1);
    std_x(std_x <= epsilon) = 1;
    X = bsxfun(@rdivide, X, std_x);
elseif standard==0
    m_x(m_x==0) = 1;
    X = bsxfun(@rdivide, X, m_x);
elseif standard == 2
    normy = sqrt(sum(X.^2,1));
    normy(normy< 1e-20) = 1;
    X = bsxfun(@rdivide, X, normy);
end
 
  % tempory
  if trans_flag == 1
      X = X';
  end
end



