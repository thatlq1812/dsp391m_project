# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Section 13: References

## Academic Papers

### Traffic Forecasting - Deep Learning

1. **Yu, B., Yin, H., & Zhu, Z. (2018).** "Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting." _IJCAI 2018_.

   - First application of ST-GCN to traffic forecasting
   - ChebNet + temporal convolution architecture

2. **Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019).** "Graph WaveNet for Deep Spatial-Temporal Graph Modeling." _IJCAI 2019_.

   - Adaptive adjacency matrix learning
   - Temporal Convolutional Networks (TCN)
   - SOTA performance on METR-LA (MAE 2.69 mph)

3. **Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020).** "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." _KDD 2020_.

   - MTGNN: Multi-faceted graph learning
   - Uni-directional adaptive graphs
   - Mix-hop propagation

4. **Guo, S., Lin, Y., Feng, N., Song, C., & Wan, H. (2019).** "Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting." _AAAI 2019_.

   - ASTGCN architecture
   - Spatial and temporal attention mechanisms
   - Multi-component modeling (recent/daily/weekly)

5. **Zheng, C., Fan, X., Wang, C., & Qi, J. (2020).** "GMAN: A Graph Multi-Attention Network for Traffic Prediction." _AAAI 2020_.

   - Parallel spatial-temporal attention
   - Encoder-decoder transformer architecture
   - Gated fusion mechanism

6. **Li, M., & Zhu, Z. (2021).** "Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting." _AAAI 2021_.
   - Dynamic graph construction
   - DGCRN: Current SOTA on METR-LA (MAE 2.59 mph)

### Graph Neural Networks

7. **Kipf, T. N., & Welling, M. (2017).** "Semi-Supervised Classification with Graph Convolutional Networks." _ICLR 2017_.

   - Foundational GCN paper
   - Spectral graph convolution

8. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).** "Graph Attention Networks." _ICLR 2018_.

   - GAT with attention mechanism
   - Dynamic neighbor weighting

9. **Brody, S., Alon, U., & Yahav, E. (2022).** "How Attentive are Graph Attention Networks?" _ICLR 2022_.

   - GATv2: Fixed expressiveness limitation
   - Improved dynamic attention

10. **Defferrard, M., Bresson, X., & Vandergheynst, P. (2016).** "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering." _NeurIPS 2016_.
    - ChebNet: Chebyshev polynomial filters
    - Efficient spectral convolution

### Recurrent Neural Networks

11. **Hochreiter, S., & Schmidhuber, J. (1997).** "Long Short-Term Memory." _Neural Computation 9(8)_.

    - LSTM architecture
    - Gating mechanisms for long-term dependencies

12. **Ma, X., Tao, Z., Wang, Y., Yu, H., & Wang, Y. (2015).** "Long Short-Term Memory Neural Network for Traffic Speed Prediction Using Remote Microwave Sensor Data." _Transportation Research Part C_.
    - LSTM for traffic forecasting
    - Single-location prediction

### Transformers and Attention

13. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** "Attention is All You Need." _NeurIPS 2017_.

    - Original Transformer architecture
    - Self-attention mechanism

14. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021).** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." _ICLR 2021_.
    - Vision Transformer (ViT)
    - Positional encodings

### Uncertainty Quantification

15. **Bishop, C. M. (1994).** "Mixture Density Networks." Technical Report NCRG/94/004, Aston University.

    - Mixture Density Networks (MDN)
    - Gaussian mixture outputs for regression

16. **Gneiting, T., & Raftery, A. E. (2007).** "Strictly Proper Scoring Rules, Prediction, and Estimation." _Journal of the American Statistical Association_.

    - CRPS (Continuous Ranked Probability Score)
    - Proper scoring rules for probabilistic forecasts

17. **Gal, Y., & Ghahramani, Z. (2016).** "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." _ICML 2016_.
    - MC Dropout for uncertainty estimation
    - Bayesian neural networks

### Multi-Modal Fusion

18. **Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A. (2018).** "FiLM: Visual Reasoning with a General Conditioning Layer." _AAAI 2018_.

    - Feature-wise Linear Modulation
    - Conditional neural networks

19. **Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2019).** "Multimodal Machine Learning: A Survey and Taxonomy." _IEEE Transactions on Pattern Analysis and Machine Intelligence_.
    - Multi-modal fusion strategies
    - Cross-modal attention

### Weather Impact on Traffic

20. **Zhang, Y., Haghani, A., & Zeng, X. (2019).** "Probabilistic Analysis of Traffic Congestion Due to Weather and Incidents." _Transportation Research Record_.

    - Rain impact on traffic speed
    - Empirical analysis

21. **Datla, S., & Sharma, S. (2008).** "Impact of Cold and Snow on Temporal and Spatial Variations of Highway Traffic Volumes." _Journal of Transport Geography_.
    - Weather effects on traffic patterns
    - Seasonal variations

---

## Datasets and Benchmarks

22. **Jagadish, H. V., Gehrke, J., Labrinidis, A., Papakonstantinou, Y., Patel, J. M., Ramakrishnan, R., & Shahabi, C. (2014).** "Big Data and Its Technical Challenges." _Communications of the ACM_.

    - METR-LA, PeMS-BAY datasets
    - Traffic forecasting benchmarks

23. **Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018).** "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting." _ICLR 2018_.
    - DCRNN baseline
    - Dataset preprocessing methodology

---

## Machine Learning Fundamentals

24. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** _Deep Learning_. MIT Press.

    - Comprehensive ML textbook
    - Neural network foundations

25. **Kingma, D. P., & Ba, J. (2015).** "Adam: A Method for Stochastic Optimization." _ICLR 2015_.

    - Adam optimizer
    - Adaptive learning rates

26. **Ioffe, S., & Szegedy, C. (2015).** "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." _ICML 2015_.

    - Batch normalization
    - Training stabilization

27. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." _JMLR 15(1)_.
    - Dropout regularization
    - Overfitting prevention

---

## Software and Tools

28. **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).** "PyTorch: An Imperative Style, High-Performance Deep Learning Library." _NeurIPS 2019_.

    - PyTorch framework
    - https://pytorch.org/

29. **Fey, M., & Lenssen, J. E. (2019).** "Fast Graph Representation Learning with PyTorch Geometric." _ICLR Workshop 2019_.

    - PyTorch Geometric library
    - https://pytorch-geometric.readthedocs.io/

30. **McKinney, W. (2010).** "Data Structures for Statistical Computing in Python." _Proceedings of the 9th Python in Science Conference_.

    - Pandas library
    - https://pandas.pydata.org/

31. **Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020).** "Array Programming with NumPy." _Nature 585_.
    - NumPy library
    - https://numpy.org/

---

## APIs and Data Sources

32. **Google Directions API** - https://developers.google.com/maps/documentation/directions

    - Real-time traffic data source
    - Route duration in traffic

33. **OpenWeatherMap API** - https://openweathermap.org/api

    - Weather data provider
    - Historical and forecast data

34. **OpenStreetMap / Overpass API** - https://overpass-api.de/
    - Road network topology
    - Geographic data

---

## Project-Specific References

35. **STMGT Project Repository** - https://github.com/thatlq1812/dsp391m_project

    - Source code
    - Training scripts and configurations
    - API implementation

36. **Project Documentation** - `docs/INDEX.md`

    - Architecture documentation
    - Training guides
    - API documentation

37. **Research Consolidated Report** - `docs/research/STMGT_RESEARCH_CONSOLIDATED.md`
    - Literature review synthesis
    - Design decisions rationale
    - Ablation study plans

---

## Vietnamese Context References

38. **Ho Chi Minh City Department of Transport** - http://www.sggp.org.vn/

    - Traffic statistics
    - Urban planning documents

39. **Vietnam Institute for Economic and Policy Research (VEPR)** - http://vepr.org.vn/
    - Economic impact of traffic congestion
    - Policy recommendations

**[PLACEHOLDER: Add more Vietnam-specific references if available]**

---

## Additional Resources

### Online Courses and Tutorials

40. **Stanford CS224W: Machine Learning with Graphs** - http://web.stanford.edu/class/cs224w/

    - Graph neural networks course
    - Jure Leskovec

41. **Deep Learning Specialization (Coursera)** - Andrew Ng
    - Neural network foundations
    - Optimization techniques

### Conference Proceedings

- **ICLR (International Conference on Learning Representations)**
- **NeurIPS (Neural Information Processing Systems)**
- **ICML (International Conference on Machine Learning)**
- **AAAI (Association for the Advancement of Artificial Intelligence)**
- **KDD (Knowledge Discovery and Data Mining)**
- **IJCAI (International Joint Conference on Artificial Intelligence)**

---

## Citation Style

This report uses **IEEE citation style** for consistency with computer science publications.

**Format:**

- Journal: `[#] Author(s), "Title," *Journal*, vol. X, no. Y, pp. Z-W, Year.`
- Conference: `[#] Author(s), "Title," in *Proc. Conference*, Year, pp. Z-W.`
- Book: `[#] Author(s), *Book Title*, Publisher, Year.`
- Website: `[#] "Title," URL, accessed: Month Day, Year.`

---

**Total References:** 41+ (academic papers, datasets, tools, APIs)

**Next:** [Appendices →](12_appendices.md)
