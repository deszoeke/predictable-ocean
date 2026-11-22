# Reading list: predictability, ensembles, TLM/adjoint, targeting, ML emulators (2026-11-19)

Ordered for reading. ★ = start here. "checked" = details confirmed by web search on 2026-11-19; the rest are from memory and still need a DOI check before citing.

## 1. Predictability
- ★ Kalnay, E., 2003: *Atmospheric Modeling, Data Assimilation and Predictability.* Cambridge Univ. Press. Ch. 5 (data assimilation), Ch. 6 (predictability, ensembles, TLM and adjoint). The single best textbook entry.
- Lorenz, E. N., 1969: The predictability of a flow which possesses many scales of motion. *Tellus*, 21, 289–307. Upscale error growth; the reason emulators that damp small scales get predictability wrong.
- Palmer, T., and R. Hagedorn (eds.), 2006: *Predictability of Weather and Climate.* Cambridge Univ. Press. Chapters by leading authors on each topic below.
- DelSole, T., and M. K. Tippett, 2007: Predictability: Recent insights from information theory. *Rev. Geophys.*, 45, RG4002. Predictable components, the statistics side.

## 2. Ensemble forecasting
- ★ Leutbecher, M., and T. N. Palmer, 2008: Ensemble forecasting. *J. Comput. Phys.*, 227, 3515–3539. (checked) Initial perturbations, singular vectors, model error, verification.
- Gneiting, T., and A. E. Raftery, 2007: Strictly proper scoring rules, prediction, and estimation. *J. Amer. Stat. Assoc.*, 102, 359–378. CRPS and why proper scores matter; also the basis of probabilistic emulator training.

## 3. Tangent-linear and adjoint models, singular vectors
- ★ Errico, R. M., 1997: What is an adjoint model? *Bull. Amer. Meteor. Soc.*, 78, 2577–2591. Short, clear primer.
- Buizza, R., and T. N. Palmer, 1995: The singular-vector structure of the atmospheric global circulation. *J. Atmos. Sci.*, 52, 1434–1456.
- Farrell, B. F., and P. J. Ioannou, 1996: Generalized stability theory. Part I: Autonomous operators. *J. Atmos. Sci.*, 53, 2025–2040. Non-normal transient growth.

## 4. Data assimilation and forecast error covariance
- ★ Carrassi, A., M. Bocquet, L. Bertino, and G. Evensen, 2018: Data assimilation in the geosciences: An overview of methods, issues, and perspectives. *WIREs Clim. Change*, 9, e535. Variational and ensemble methods, and how the error covariance is propagated.
- Evensen, G., 2009: *Data Assimilation: The Ensemble Kalman Filter*, 2nd ed. Springer.
- Asch, M., M. Bocquet, and M. Nodet, 2016: *Data Assimilation: Methods, Algorithms, and Applications.* SIAM. More mathematical.

## 5. Sensitivity and targeted observations
- ★ Majumdar, S. J., 2016: A review of targeted observations. *Bull. Amer. Meteor. Soc.*, 97, 2287–2303.
- Bishop, C. H., B. J. Etherton, and S. J. Majumdar, 2001: Adaptive sampling with the ensemble transform Kalman filter. Part I: Theoretical aspects. *Mon. Wea. Rev.*, 129, 420–436. Variance reduction from an ensemble (the ΔVar formula).
- Ancell, B., and G. J. Hakim, 2007: Comparing adjoint- and ensemble-sensitivity analysis with applications to observation targeting. *Mon. Wea. Rev.*, 135, 4117–4134.
- Torn, R. D., and G. J. Hakim, 2008: Ensemble-based sensitivity analysis. *Mon. Wea. Rev.*, 136, 663–677.

## 6. ML emulators: what they get wrong about error growth
- ★ Selz, T., and G. C. Craig, 2023: Can artificial intelligence-based weather prediction models simulate the butterfly effect? *Geophys. Res. Lett.*, 50, e2023GL105747. (checked) Pangu-Weather: small initial errors do not grow; it overestimates predictability.
- Bonavita, M., 2024: On some limitations of current machine learning weather prediction models. *Geophys. Res. Lett.*, 51, e2023GL107377. (checked) Pangu, FourCastNet, GraphCast lack sub-synoptic and mesoscale realism.
- Hakim, G. J., and S. Masanam, 2024: Dynamical tests of a deep-learning weather prediction model. arXiv:2309.10867 (checked; journal venue not confirmed). Linear responses of Pangu-Weather (Gill response, cyclone growth, geostrophic adjustment): a model for testing emulator dynamics.

## 7. Training emulators, and what to train them to do
- Bonev, B., et al., 2023: Spherical Fourier neural operators: Learning stable dynamics on the sphere. *ICML 2023*; arXiv:2306.03838. The SFNO.
- Price, I., et al., 2025: Probabilistic weather forecasting with machine learning. *Nature*, 637, 84–90. GenCast, a diffusion model trained to sample the forecast distribution.
- Lang, S., et al., 2024: AIFS-CRPS: Ensemble forecasting using a model trained with a loss function based on the continuous ranked probability score. arXiv:2412.15832. (checked) Trained on CRPS; learns model uncertainty by shaping noise.
- Kochkov, D., et al., 2024: Neural general circulation models for weather and climate. *Nature*, 632, 1060–1066. Hybrid dynamical core plus learned physics; deterministic and stochastic versions.
- ★ Hatfield, S., M. Chantry, P. Dueben, P. Lopez, A. Geer, and T. Palmer, 2021: Building tangent-linear and adjoint models for data assimilation with neural networks. *J. Adv. Model. Earth Syst.*, 13, e2021MS002521. (checked) NN emulator of a parametrization; its TLM and adjoint pass the standard tests and work in 4D-Var. Closest to this project's aim.
- Czarnecki, W. M., et al., 2017: Sobolev training for neural networks. *NeurIPS 2017*. Training on derivatives (Jacobian matching) as well as values.
- Pathak, J., B. Hunt, M. Girvan, Z. Lu, and E. Ott, 2018: Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach. *Phys. Rev. Lett.*, 120, 024102. An emulator judged by whether it reproduces the Lyapunov exponents, not only forecast error.
