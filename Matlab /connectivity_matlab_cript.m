load data

%%
cfg = [];
cfg.channel    = {'MRC21' 'MLC21' 'EMGlft' 'EMGrgt'};
data = ft_preprocessing(cfg, data);

cfg = [];
cfg.resamplefs  = 300;
data = ft_resampledata(cfg, data);

number_trials = length(data.trial);
for trial = 1:number_trials
    data.trial{trial} = zscore(data.trial{trial}')';
end

time = data.time{1};
reduced_indices = 1:length(time);
reduced_time = time(reduced_indices);
[T_x, T_y] = meshgrid(reduced_time,reduced_time); % For constructing the kernel

%%
cfg            = [];
cfg.output     = 'powandcsd';
cfg.method     = 'mtmfft';
cfg.foilim     = [0 60];
cfg.tapsmofrq  = 1;
%cfg.keeptrials = 'yes';
cfg.channel    = {'MRC21' 'MLC21' 'EMGlft' 'EMGrgt'};
%cfg.channelcmb = {'MRC21' 'EMGlft'; 'MRC21' 'EMGrgt'};
freq           = ft_freqanalysis(cfg, data);
%%
figure,
plot(freq.freq, freq.powspctrm(2,:))
hold on
plot(freq.freq, freq.powspctrm(3,:))

%% %% GP decomposition %% %%

%% rearrange data 
MEG_signal = cell(number_trials,1);
EMG_signal = cell(number_trials,1);
for trial = 1:number_trials
    MEG_signal{trial} = data.trial{trial}(1:4,:);
    EMG_signal{trial} = data.trial{trial}(3:4,:);
end
    
%% Signal covariance function (MEG)
c_1 = 0;
for trial = 1:number_trials
    demeaned_signal = MEG_signal{trial}(:,reduced_indices); %detrend(spatiotemporal_signal{trial}','constant')';
    c_1 = c_1 + demeaned_signal'*demeaned_signal/number_trials;
end

% stationarization
c_stat_1 = 0;
for j = -length(reduced_time):length(reduced_time)
    c_stat_1 = c_stat_1 + diag(mean(diag(c_1,j))*ones(length(diag(c_1,j)),1),j);
end

max_c_1 = max(max(c_stat_1));
MEG_covariance = c_stat_1/max_c_1;

figure,
plot(diag(fliplr(MEG_covariance)))

%% Signal covariance function (EMG)
c_1 = 0;
for trial = 1:number_trials
    demeaned_signal = EMG_signal{trial}(:,reduced_indices); %detrend(spatiotemporal_signal{trial}','constant')';
    c_1 = c_1 + demeaned_signal'*demeaned_signal/number_trials;
end

% stationarization
c_stat_1 = 0;
for j = -length(reduced_time):length(reduced_time)
    c_stat_1 = c_stat_1 + diag(mean(diag(c_1,j))*ones(length(diag(c_1,j)),1),j);
end

max_c_1 = max(max(c_stat_1));
EMG_covariance = c_stat_1/max_c_1;

figure,
plot(diag(fliplr(EMG_covariance)))

%% MEG GP parameters

% Priors parameters

% Kernel configuration
num_kernels = 4; % Number of kernels
kernels = {'exponential','damped_harmonic_oscillator_alpha', 'damped_harmonic_oscillator_beta', 'squared_exponential'}; % Type of kernels

% Prior configuration
cfg_param = [];
num_param = [ 2, 3, 3, 2];
% Each hyper-parameter can have its own (hyper-)prior distribution
prior_pdf{1} = {'logNorm', 'unif'}; %Exponential
prior_pdf{2} = {'logNorm', 'logNorm', 'beta'};
prior_pdf{3} = {'logNorm', 'logNorm', 'beta'};
prior_pdf{4} = {'logNorm','logNorm'};


% Hyper-prior parameters
prior_hyp_parameters{1} = { [ 1, 2], [0, 5]}; %Exponential
prior_hyp_parameters{2} = { [ 1, 2], [0.01, 40], [2*pi*9, 2*pi*11, 1, 1]}; %Theta
prior_hyp_parameters{3} = { [ 1, 2], [0.01, 40], [2*pi*14, 2*pi*16, 1, 1]}; %Alpha 
prior_hyp_parameters{4} = { [ 1, 2], [ 0, 1]}; 

% Variable transformations
var_transform{1} = {'log', 'logit'};
var_transform{2} = {'log','log', 'logit' };
var_transform{3} = {'log','log', 'logit' };
var_transform{4} = {'log','logit'};

transform_param{1} = {[], [0, 5]};
transform_param{2} = {[], [], [2*pi*8, 2*pi*11] };
transform_param{3} = {[], [], [2*pi*14, 2*pi*16] };
transform_param{4} = {[], [0.000001,0.0001]};

for j = 1:num_kernels
    % Parameters structure
    cfg = [];
    cfg.numparam = num_param(j); % number of (hyper-)parameters
    cfg.pdf = prior_pdf{j};
    cfg.hyparam = prior_hyp_parameters{j};
    cfg.tran = var_transform{j};
    cfg.tranparam = transform_param{j}; 
    cfg_param = setfield(cfg_param, kernels{j}, cfg);
end

% Kernel structure
cfg_k = [];
cfg_k.kernels = kernels;
cfg_k.num = num_kernels;
cfg_k.num_param = max(num_param);

% Parameters initialization
in_param = [log(1/4), log(1), log(1/4), 0, 0, log(1/4), 0, 0, log(1/4), 0  ]'  + 0.1*randn(sum(num_param),1);

%% MEG Simulated annealing 
% Finding the global mode that will be also used as a starting point for
% the monte carlo sampling
sigma = 0.1;

l = @(x) ct_least_squares4(cfg_k, cfg_param, x, MEG_covariance, T_x, T_y);
g = @(x) x + randn(1)*sigma*mnrnd(1,ones(size(x))/length(x)).*trnd(2);

cfg = [];
cfg.Verbosity = 2;
cfg.InitTemp = 1;
cfg.StopTemp = 10^-4;cfg.Generator = g;
cfg.CoolSched = @(T) (.95*T);
mode = anneal(l,in_param',cfg);
lambda_n = 0.1;

%% MEG Covariance function plot
theta = ct_transform(cfg_k, cfg_param, mode);
[K_tot K] = ct_kernel(cfg_k, theta, T_x, T_y, 0);
K_tot = max_c_1*K_tot;
K = max_c_1*K;

tau = (T_x - T_y);
k_plot = [ fliplr(K_tot(2:end,1)'), K_tot(1,1:end)];
c_plot = [ fliplr(MEG_covariance(2:end,1)'), MEG_covariance(1,1:end)];
t_plot = [ fliplr(tau(2:end,1)'), tau(1,1:end)];

figure, 
plot(t_plot, max_c_1*c_plot, t_plot, k_plot)
legend('Data autocovariance','Parametric fitting')

%% MEG Posterior mean
MEG_mean_process = cell(number_trials,1);

[K_tot, K] = ct_kernel(cfg_k, ct_transform(cfg_k, cfg_param, mode), T_x, T_y, 0);
kernel = 3;
mean_pos = cell(number_trials,1);
for trial = 1:number_trials
    inv_K = squeeze(K(kernel,:,:))/K_tot;
    mean_pos{trial} = MEG_signal{trial}*inv_K';
    MEG_mean_process{trial} = mean_pos{trial};
end

figure,
plot(time, MEG_mean_process{3})


%% MEG GP parameters

% Priors parameters

% Kernel configuration
num_kernels = 3; % Number of kernels
kernels = {'exponential', 'damped_harmonic_oscillator_beta', 'squared_exponential'}; % Type of kernels

% Prior configuration
cfg_param = [];
num_param = [ 2, 3, 2];
% Each hyper-parameter can have its own (hyper-)prior distribution
prior_pdf{1} = {'logNorm', 'unif'}; %Exponential
prior_pdf{2} = {'logNorm', 'logNorm', 'beta'};
prior_pdf{3} = {'logNorm','logNorm'};


% Hyper-prior parameters
prior_hyp_parameters{1} = { [ 1, 2], [0, 5]}; %Exponential
prior_hyp_parameters{2} = { [ 1, 2], [0.01, 40], [2*pi*7, 2*pi*20, 1, 1]}; 
prior_hyp_parameters{3} = { [ 1, 2], [ 0, 1]}; 

% Variable transformations
var_transform{1} = {'log', 'logit'};
var_transform{2} = {'log','log', 'logit' };
var_transform{3} = {'log','logit'};

transform_param{1} = {[], [0, 5]};
transform_param{2} = {[], [], [2*pi*7, 2*pi*20] };
transform_param{3} = {[], [0.000001,0.0001]};

for j = 1:num_kernels
    % Parameters structure
    cfg = [];
    cfg.numparam = num_param(j); % number of (hyper-)parameters
    cfg.pdf = prior_pdf{j};
    cfg.hyparam = prior_hyp_parameters{j};
    cfg.tran = var_transform{j};
    cfg.tranparam = transform_param{j}; 
    cfg_param = setfield(cfg_param, kernels{j}, cfg);
end

% Kernel structure
cfg_k = [];
cfg_k.kernels = kernels;
cfg_k.num = num_kernels;
cfg_k.num_param = max(num_param);

% Parameters initialization
in_param = [log(1/4), log(1), log(1/4), 0, 0, log(1/4), 0  ]'  + 0.1*randn(sum(num_param),1);

%% MEG Simulated annealing 
% Finding the global mode that will be also used as a starting point for
% the monte carlo sampling
sigma = 0.1;

l = @(x) ct_least_squares4(cfg_k, cfg_param, x, EMG_covariance, T_x, T_y);
g = @(x) x + randn(1)*sigma*mnrnd(1,ones(size(x))/length(x)).*trnd(2);

cfg = [];
cfg.Verbosity = 2;
cfg.InitTemp = 1;
cfg.StopTemp = 10^-3;cfg.Generator = g;
cfg.CoolSched = @(T) (.95*T);
mode = anneal(l,in_param',cfg);
lambda_n = 0.1;

%% MEG Covariance function plot
theta = ct_transform(cfg_k, cfg_param, mode);
[K_tot K] = ct_kernel(cfg_k, theta, T_x, T_y, 0);
K_tot = max_c_1*K_tot;
K = max_c_1*K;

tau = (T_x - T_y);
k_plot = [ fliplr(K_tot(2:end,1)'), K_tot(1,1:end)];
c_plot = [ fliplr(EMG_covariance(2:end,1)'), EMG_covariance(1,1:end)];
t_plot = [ fliplr(tau(2:end,1)'), tau(1,1:end)];

figure, 
plot(t_plot, max_c_1*c_plot, t_plot, k_plot)
legend('Data autocovariance','Parametric fitting')

%% Posterior mean
EMG_mean_process = cell(number_trials,1);

[K_tot, K] = ct_kernel(cfg_k, ct_transform(cfg_k, cfg_param, mode), T_x, T_y, 0);
kernel = 2;
mean_pos = cell(number_trials,1);
for trial = 1:number_trials
    inv_K = squeeze(K(kernel,:,:))/K_tot;
    mean_pos{trial} = EMG_signal{trial}*inv_K';
    EMG_mean_process{trial} = mean_pos{trial};
end

%%
trial = 5;
figure,
plot(time, EMG_mean_process{trial}(1,:), time, MEG_mean_process{trial}(2,:))







