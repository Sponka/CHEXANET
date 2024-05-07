
<img src="Figures/logo_chexanet.png" width="80" align="left" style="margin-right: 30px; vertical-align: middle; margin-right: 20px;"/>

  <h2> <p align="center"> CHEXANET: A Novel Approach to Fast-Tracking Disequilibrium Chemistry Calculations for Exoplanets Using Neural Networks </p>  </h2> 
<br />
<br />
<p style="border: 1px solid #ccc; padding: 10px; text-align: justify;"> In the rapidly evolving field of exoplanetary science with missions like JWST and Ariel, there is a pressing need for fast and accurate simulations of disequilibrium chemistry in exoplanet atmospheres. Traditional methods, which assume chemical equilibrium, simplify calculations but fail to capture the more complex chemical dynamics observed in actual exoplanets. Accurate estimations require complex kinetic codes that are time-intensive due to the need to solve ordinary differential equations. Given an extensive parameter space that needs to be explored to calculate forward models, kinetic codes will become a significant bottleneck. We introduce CHEXANET, a novel U-Net-based neural network architecture designed to efficiently simulate disequilibrium chemistry in exoplanetary atmospheres. Utilising an equilibrium state of the hot-Jupiters atmosphere, which computes in seconds, alongside a set of initial parameters, the network effectively learns to predict the atmosphere in disequilibrium. It significantly enhances computational efficiency, reducing the prediction time for atmospheric disequilibrium states to just one second per atmosphere on a standard personal computer—over a hundred times faster than traditional methods. Our results position CHEXANET as a promising alternative for large-scale studies of exoplanetary atmospheres. </p> 


 # Prerequisites 
 • Tensorflow (Tensorflow-gpu) \
 

