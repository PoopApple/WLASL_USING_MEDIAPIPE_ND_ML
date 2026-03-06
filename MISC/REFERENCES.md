# References

## Dataset

1. **WLASL: A Large-Scale Dataset for Word-Level American Sign Language**
   - Li, D., Rodriguez, C., Yu, X., & Li, H. (2020)
   - Paper: https://arxiv.org/abs/1910.11006
   - Dataset: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
   - Description: Word-Level American Sign Language dataset with 2,000+ signs and 21,000+ video instances

## Pose and Hand Landmark Detection

2. **MediaPipe Holistic**
   - Google Research, 2020
   - Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
   - Hand Landmarks: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
   - Pose Landmarks: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md
   - Description: Real-time human pose and hand tracking using machine learning models

3. **BlazePose: On-device Real-time Body Pose tracking**
   - Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020)
   - Paper: https://arxiv.org/abs/2006.10204
   - Description: Lightweight pose estimation model optimized for mobile devices

4. **MediaPipe Hands: On-device Real-time Hand Tracking**
   - Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C. L., & Grundmann, M. (2020)
   - Paper: https://arxiv.org/abs/2006.10214
   - Description: 21-point hand landmark detection model

## Deep Learning Models

5. **Long Short-Term Memory Networks (LSTM)**
   - Hochreiter, S., & Schmidhuber, J. (1997)
   - Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
   - Description: Foundational recurrent neural network architecture for sequence modeling

6. **Bidirectional LSTM**
   - Schuster, M., & Paliwal, K. K. (1997)
   - Paper: IEEE Transactions on Signal Processing, 45(11), 2673-2681
   - Description: Processes sequences in both forward and backward directions

7. **Gated Recurrent Units (GRU)**
   - Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014)
   - Paper: https://arxiv.org/abs/1406.1078
   - Description: Simplified alternative to LSTM with fewer parameters

8. **Attention Mechanism**
   - Bahdanau, D., Cho, K., & Bengio, Y. (2015)
   - Paper: https://arxiv.org/abs/1409.0473
   - Description: Attention mechanism for neural machine translation, applicable to sequence modeling

## Optimization Techniques

9. **Adam: A Method for Stochastic Optimization**
   - Kingma, D. P., & Ba, J. (2014)
   - Paper: https://arxiv.org/abs/1412.6980
   - Description: Adaptive learning rate optimization algorithm

10. **AdamW: Decoupled Weight Decay Regularization**
    - Loshchilov, I., & Hutter, F. (2017)
    - Paper: https://arxiv.org/abs/1711.05101
    - Description: Adam with improved weight decay for better generalization

11. **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**
    - Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014)
    - Paper: Journal of Machine Learning Research, 15(1), 1929-1958
    - Description: Regularization technique by randomly dropping units during training

12. **Batch Normalization: Accelerating Deep Network Training**
    - Ioffe, S., & Szegedy, C. (2015)
    - Paper: https://arxiv.org/abs/1502.03167
    - Description: Normalizes layer inputs to stabilize and accelerate training

13. **Label Smoothing**
    - Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016)
    - Paper: https://arxiv.org/abs/1512.00567
    - Description: Regularization technique that prevents overconfident predictions

## Data Augmentation

14. **Data Augmentation for Deep Learning**
    - Shorten, C., & Khoshgoftaar, T. M. (2019)
    - Paper: Journal of Big Data, 6(1), 60
    - Description: Survey of data augmentation techniques for improving model generalization

## Sign Language Recognition

15. **Survey on Sign Language Recognition**
    - Rastgoo, R., Kiani, K., & Escalera, S. (2021)
    - Paper: https://arxiv.org/abs/2008.09918
    - Description: Comprehensive survey of sign language recognition methods

16. **Continuous Sign Language Recognition**
    - Koller, O., Zargaran, S., Ney, H., & Bowden, R. (2018)
    - Paper: IEEE Transactions on Pattern Analysis and Machine Intelligence
    - Description: Deep learning approaches for continuous sign language recognition

## Frameworks and Libraries

17. **TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems**
    - Abadi, M., et al. (2015)
    - Website: https://www.tensorflow.org/
    - Description: Open-source machine learning framework

18. **Keras: Deep Learning for Humans**
    - Chollet, F., et al. (2015)
    - Website: https://keras.io/
    - Documentation: https://www.tensorflow.org/api_docs/python/tf/keras
    - Description: High-level neural networks API

19. **NumPy: The Fundamental Package for Scientific Computing with Python**
    - Harris, C. R., et al. (2020)
    - Paper: Nature, 585(7825), 357-362
    - Website: https://numpy.org/
    - Description: Array computing library for Python

20. **scikit-learn: Machine Learning in Python**
    - Pedregosa, F., et al. (2011)
    - Paper: Journal of Machine Learning Research, 12, 2825-2830
    - Website: https://scikit-learn.org/
    - Description: Machine learning library for Python

## Evaluation Metrics

21. **Top-k Accuracy Metric**
    - Russakovsky, O., et al. (2015)
    - Paper: ImageNet Large Scale Visual Recognition Challenge
    - Description: Measures if correct class is in top-k predictions

22. **F1-Score and Precision-Recall**
    - Powers, D. M. (2011)
    - Paper: Evaluation: From Precision, Recall and F-Measure to ROC
    - Description: Standard metrics for classification evaluation

## Additional Resources

23. **Mixed Precision Training**
    - Micikevicius, P., et al. (2017)
    - Paper: https://arxiv.org/abs/1710.03740
    - Description: Training neural networks with lower precision to improve speed and memory

24. **Early Stopping**
    - Prechelt, L. (1998)
    - Paper: Neural Networks: Tricks of the Trade
    - Description: Regularization technique to prevent overfitting

25. **Learning Rate Scheduling**
    - Smith, L. N. (2017)
    - Paper: https://arxiv.org/abs/1506.01186
    - Description: Cyclical learning rates for training neural networks

---

## Dataset Citation

If you use the WLASL dataset, please cite:

```bibtex
@inproceedings{li2020word,
  title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={1459--1469},
  year={2020}
}
```

## MediaPipe Citation

```bibtex
@misc{mediapipe,
  title={MediaPipe},
  author={Google Research},
  year={2020},
  howpublished={\url{https://mediapipe.dev}}
}
```

---

*Last updated: November 27, 2025*
