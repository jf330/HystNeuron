function [out, elig_sum_array] = sim_matlab(stop_time, N, path)

% Load SymBiology neuron model
m2_load = sbioloadproject('/Users/jf330/Dropbox/JuliaResearch/Paper1/Matlab_project/hyst_neuron.sbproj');
% m2_load = sbioloadproject('/home/jf330/HystNeuron/matlab/hyst_neuron.sbproj');
% m2_load = sbioloadproject('/Users/jf330/Dropbox/JuliaResearch/Paper1/Matlab_project/hyst_neuron_test.sbproj');

m2 = m2_load.m1;

% Readjust neural dynamics parameters
params = load(path + "N_" + N + "_params.mat");

m2.Parameters(1).Value = params.a; % a
m2.Parameters(2).Value = params.b; % b
m2.Parameters(3).Value = params.g; % g
m2.Parameters(4).Value = params.d; % d
m2.Parameters(5).Value = params.eta; % eta
m2.Parameters(6).Value = params.K; % threshold
m2.Parameters(7).Value = params.h; % steepness

% Prepare weighted dose inputs
for neur_idx = 0:N-1
    dose_new{neur_idx+1} = sbiodose("dose_"+ neur_idx, "schedule");
% "/matlab_data/N_"
    in_time = load(path + "N_"+ N +"_time_inputs_"+ neur_idx +".mat");
    in_amount = load(path + "N_"+ N +"_weight_inputs_"+ neur_idx +".mat");

    Time = transpose(in_time.data);
    Amount = transpose(in_amount.data);

    tbl = table(Time, Amount);
    setTable(dose_new{neur_idx+1}, tbl);

    adddose(m2, dose_new{neur_idx+1});
    dose_new{neur_idx+1}.targetName = "V";
    
    if neur_idx == 0
        dose_array = dose_new{neur_idx+1};
    else
        dose_array = [dose_array; dose_new{neur_idx+1}];
    end
end

csObj = getconfigset(m2,'active');
set(csObj, 'Stoptime', stop_time);
 
[time, x, names] = sbiosimulate(m2, dose_array);

% Calculate spike estimation
% spike_integral = trapz(time,x(:,4));

% Choose neuron post_synaptic trace
post_syn = x(:,1); %V
% post_syn = x(:,3); %O

% Calculate eligibility for each pre-synaptic neuron
elig_sum_array = [];
for neur_idx = 1:N
    dose_times = dose_array(neur_idx).Time;
    elig = 0;
    if not(isempty(dose_times))
        for dose_t = dose_times
            elig = elig + max(post_syn(find(time>=dose_t & time<dose_t+1)));
        end
    end
    if not(isempty(elig))
        elig_sum_array(neur_idx) = elig;
    else
        elig_sum_array(neur_idx) = 0;
    end
end

out_x = x;
out_times = time;
out = transpose([out_x out_times]);
