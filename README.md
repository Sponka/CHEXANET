
<img src="Figures/logo_chexanet.png" width="80" align="left" style="margin-right: 30px; vertical-align: middle; margin-right: 20px;"/>

  <h2> <p align="center"> CHEXANET: A Novel Approach to Fast-Tracking Disequilibrium Chemistry Calculations for Exoplanets Using Neural Networks </p>  </h2> 
<br />
<br />
<p align="justify"> In the rapidly evolving field of exoplanetary science with missions like JWST and Ariel, there is a pressing need for fast and accurate simulations of disequilibrium chemistry in exoplanet atmospheres. Traditional methods, which assume chemical equilibrium, simplify calculations but fail to capture the more complex chemical dynamics observed in actual exoplanets. Accurate estimations require complex kinetic codes that are time-intensive due to the need to solve ordinary differential equations. Given an extensive parameter space that needs to be explored to calculate forward models, kinetic codes will become a significant bottleneck. We introduce CHEXANET, a novel U-Net-based neural network architecture designed to efficiently simulate disequilibrium chemistry in exoplanetary atmospheres. Utilising an equilibrium state of the hot-Jupiters atmosphere, which computes in seconds, alongside a set of initial parameters, the network effectively learns to predict the atmosphere in disequilibrium. It significantly enhances computational efficiency, reducing the prediction time for atmospheric disequilibrium states to just one second per atmosphere on a standard personal computer—over a hundred times faster than traditional methods. </p> 


 # Prerequisites 
- numpy
- pandas
- matplotlib
- tabulate
- itertools
- os
- glob
- logging
- datetime
- random
- string
- copy
- scipy
- <a href='https://pypi.org/project/astro-forecaster/'>forecaster</a> (ensure you have the correct installation source)
- <a href ='https://taurex3-public.readthedocs.io/en/latest/'>taurex</a> (ensure you have the correct installation source) 
- tensorflow (Tensorflow-gpu)
- keras
- scikit-learn
- joblib
- seaborn
- statsmodels
- plotly
- argparse
- <a href ='https://arxiv.org/pdf/2209.11203'>pychegp</a> (ensure you have the correct installation source)


## 📖 Cite this work

If you use **CHEXANET** in your research, please cite:

> **Vojtekova, A., Waldmann, I., Yip, K. H., Merín, B., Al-Refaie, A. F., & Venot, O. (2025).**  
> *CHEXANET: a novel approach to fast-tracking disequilibrium chemistry calculations for exoplanets using neural networks.*  
> **Monthly Notices of the Royal Astronomical Society, 538(3), 1690–1719.**  
> [https://doi.org/10.1093/mnras/staf297](https://doi.org/10.1093/mnras/staf297)

BibTeX:
```bibtex
@ARTICLE{2025MNRAS.538.1690V,
       author = {{Vojtekova}, Antonia and {Waldmann}, Ingo and {Yip}, Kai Hou and {Mer{\'\i}n}, Bruno and {Al-Refaie}, Ahmed Faris and {Venot}, Olivia},
        title = "{CHEXANET: a novel approach to fast-tracking disequilibrium chemistry calculations for exoplanets using neural networks}",
      journal = {\mnras},
         year = 2025,
        month = apr,
       volume = {538},
       number = {3},
        pages = {1690-1719},
          doi = {10.1093/mnras/staf297},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.538.1690V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
