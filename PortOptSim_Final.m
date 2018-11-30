
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Portfolio Optimization and Simulaion                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
 Briefs:
    The purpose of the program is to demonstrate portfolio optimization 
and simulation. Using Harry Markowitz portfolio theory, the program 
computes the optimal weight for a given risk/return. Using Monte Carlo 
simulation with Cholesky decomposition, the program simulates portfolio 
value in the next T period. Finally, Value-at-Risk is computed

 Programming Steps:
    There are 3 phases in this program. First, for comparison purpose, 
the optimum weight of investment for each stock is calculated for 
(1) equally weighted portfolio, (2) minimum variance portfolio, and 
(3) required rate of return portfolio. Given an initial sum of investment
 and the latest stock prices, the number of units for each stock can be 
calculated for each portfolio investment style. The second phase of 
the program is to simulate the price paths of each stock by using Monte 
Carlo simulation and Cholesky decomposition technique. Given that we have 
units of investment from phase 1 and each stock price path from phase 2, 
the final phase is to calculate portfolio value and its value-at-risk at 
the end of period.

%}


%Clear workspace and load return data from mat file.
clc
clear all
load findata

% Setup all control parameters and necessary matrixs and vectors
sRQ=0.05;
nAsset=5;
nSim=10000;
nPeriod=100;
%252/12 for daily/monthly data. to ignore scaling system, set scale=1
nScale=1; 
dt=1/nScale; 
sRQ=sRQ*nScale;
nInvestment=1000000;
mReturn=[ADVANC	BBL	AOT	BIGC PTT];
vInitPrice=[209;205;237;240;321];
vMue=nScale*[mean(ADVANC);mean(BBL);mean(AOT);mean(BIGC);mean(PTT)];
vSigma=sqrt(nScale)*[std(ADVANC);std(BBL);std(AOT);std(BIGC);std(PTT)];
mStockPath=zeros(nAsset,nSim,nPeriod);
mCovar=cov(mReturn);
mCorr=corr(mReturn);
mCholsky=chol(mCorr);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Phase 1: Portfolio optimization.Calculate optimum weigths          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[1] equally weigth portfolio
vWeightEQ=zeros(nAsset,1);
vWeightEQ(:,1)=1/nAsset;
vUnitEQ=(nInvestment*vWeightEQ)'./vInitPrice';
sRetEQ=vWeightEQ'*vMue;
sRiskEQ=sqrt(vWeightEQ'*mCovar*vWeightEQ);

%[2] minimum variance portfolio
mBorderMVP=ones(nAsset+1,nAsset+1);
mBorderMVP(1:nAsset,1:nAsset)=2*mCovar;
mBorderMVP(nAsset+1,nAsset+1)=0;
vZero1=zeros(nAsset+1,1);
vZero1(nAsset+1,1)=1;
vWeightMVP=mBorderMVP^(-1)*vZero1;
vUnitMVP=nInvestment*vWeightMVP(1:nAsset)'./vInitPrice';
sRetMVP=(vWeightMVP(1:nAsset)')*vMue;
sRiskMVP=sqrt((vWeightMVP(1:nAsset)')*mCovar*vWeightMVP(1:nAsset));

%[3] required rate of return portfolio
mBorderRQ=ones(nAsset+2,nAsset+2);
mBorderRQ(1:nAsset,1:nAsset)=2*mCovar;
mBorderRQ(1:nAsset,nAsset+1)=vMue;
mBorderRQ(nAsset+1,1:nAsset)=vMue';
mCornerZeros=zeros(2);
mBorderRQ(nAsset+1:nAsset+2,nAsset+1:nAsset+2)=mCornerZeros;
vZeroRQ1=zeros(nAsset+2,1);
vZeroRQ1(nAsset+1)=sRQ;
vZeroRQ1(nAsset+2)=1;
vWeightRQ=mBorderRQ^(-1)*vZeroRQ1;
vUnitRQ=nInvestment*vWeightRQ(1:nAsset)'./vInitPrice';
sRetRQ=(vWeightRQ(1:nAsset)')*vMue;
sRiskRQ=sqrt((vWeightRQ(1:nAsset)')*mCovar*vWeightRQ(1:nAsset));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Phase 2: Portfolio simulation, simulate price path of each stock   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Note                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     i for Asset i=1...nAsset     %
%     p for Period p=1...nPeriod   %
%     s for Sim s=1...nSim         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for p=1:nPeriod
    mZi=randn(nSim,nAsset);
    mXi=mZi*mCholsky;
    for i=1:nAsset
        mStockPath(i,:,p)=mXi(:,i);
    end
end

%Generat stock price path for each stock. Assume that stock prices follow
%GBM S=S0*Exp((mue-(sigma^2)/2)*dt+sigma*X*sqrt(dt)) where X is from X=Z*CholeskyMatrix
for i=1:nAsset
    mStockPath(i,:,:)=vInitPrice(i)*exp((vMue(i)-(vSigma(i)^2)/2)*dt +vSigma(i)*mStockPath(i,:,:)*sqrt(dt));
end

%This is to check price path of each stock,for debugging purpose only.
%mPath=reshape(mStockPath(1,:,:),nSim,nPeriod);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Phase 3: Portfolio valuation, calculate portfolio value and VaR    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Note: Not that we need portfolio values in between period=1...(nPeriod-1)
%since we only need portfolio values the last period to calculate VaR
%But we will calculate it anyway so that the reader may learn from it.

%[1] equally weigthed portfolio values
mAssetValueEQ=zeros(nAsset,nSim,nPeriod);
for i=1:nAsset
    for s=1:nSim
        mAssetValueEQ(i,s,:)=vUnitEQ(i).*mStockPath(i,s,:);
    end
end
mPortValueEQ=zeros(nSim,nPeriod);
for s=1:nSim
    %this inner loop p=1..(nPeriod) is not nescessary though..
    for p=1:nPeriod
        mPortValueEQ(s,p)=sum(mAssetValueEQ(:,s,p));
    end
end
sVaR_EQ=prctile(mPortValueEQ(:,nPeriod),1);

%[2] minimum variance portfolio values
mAssetValueMVP=zeros(nAsset,nSim,nPeriod);
for i=1:nAsset
    for s=1:nSim
        mAssetValueMVP(i,s,:)=vUnitMVP(i).*mStockPath(i,s,:);
    end
end
mPortValueMVP=zeros(nSim,nPeriod);
for s=1:nSim
    for p=1:nPeriod
        mPortValueMVP(s,p)=sum(mAssetValueMVP(:,s,p));
    end
end
sVaR_MVP=prctile(mPortValueMVP(:,nPeriod),1);

%[3] required rate of return portfolio values
mAssetValueRQ=zeros(nAsset,nSim,nPeriod);
for i=1:nAsset
    for s=1:nSim
        mAssetValueRQ(i,s,:)=vUnitRQ(i).*mStockPath(i,s,:);
    end
end
mPortValueRQ=zeros(nSim,nPeriod);
for s=1:nSim
    for p=1:nPeriod
        mPortValueRQ(s,p)=sum(mAssetValueRQ(:,s,p));
    end
end
sVaR_RQ=prctile(mPortValueRQ(:,nPeriod),1);

mAns=[sRetEQ sRiskEQ sVaR_EQ;sRetMVP sRiskMVP sVaR_MVP;sRetRQ sRiskRQ sVaR_RQ];

close all;
figure

subplot(3,1,1)
hist(mPortValueEQ(:,nPeriod))

subplot(3,1,2)
hist(mPortValueMVP(:,nPeriod))

subplot(3,1,3)
hist(mPortValueRQ(:,nPeriod))
