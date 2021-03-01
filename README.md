Code is based on the code for the paper: ["Offline Meta Learning of Exploration"](https://arxiv.org/abs/2008.02598) - Ron Dorfman and Aviv Tamar.

```
@article{dorfman2020offline,
  title={Offline Meta Reinforcement Learning},
  author={Dorfman, Ron and Tamar, Aviv},
  journal={arXiv preprint arXiv:2008.02598},
  year={2020}
}
```

In `metalearner_sac.py` and `metalearner_sac_mer.py` there is the implementation of attempting to perform meta-rl without the VAE.  
The respective data collection file for these implementations is `online_training_only_sac.py`.  
In `metalearner_momentum.py` there is the implementation of standard off-polcy VariBAD with the SparsePointEnv environment with sliding ("momentum").  
In `metalearner_momentum_mer.py` there is the implementation of MER incorporated into off-policy VariBAD, tested on the SparsePointEnv environment with sliding ("momentum").  
The respective data collection file for these implementations is `online_training_momentum.py`.
`online_config/args_point_robot_sparse.py` is the configuration file for all my runs.
In `utils/plot_learning_curves.py` there is the code for plotting results.  
The rest of the files include code from the original repository: https://github.com/Rondorf/BOReL.



