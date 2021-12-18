% Project Graphs
N = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000];
times = importdata("results_time.txt");
values = importdata("results_intgrl.txt");

figure(1); clf(1);
loglog(N, times,'.-')
legend({'parallelQuad()','richardsonQuad()','serialQuad()'},...
    'location','southeast')
xlabel('N'); ylabel('time (μs)')
grid on
title('Algorithm time for approximating $\int_0^1 \exp(\cos(x))dx$',...
    'interpreter','latex')

times_f2 = importdata("results_time_f2.txt");
values_f2 = importdata("results_intgrl_f2.txt");

figure(2); clf(2);
loglog(N, times_f2,'.-')
legend({'parallelQuad()','richardsonQuad()','serialQuad()'},...
    'location','southeast')
xlabel('N'); ylabel('time (μs)')
grid on
title('Algorithm time for approximating $\int_0^1 \sqrt{\exp(\cos(x^{x^x}))}dx$',...
    'interpreter','latex')