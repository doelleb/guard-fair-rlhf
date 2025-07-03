# is our reward model from train_reward_model_base.py actually unbiased? 


# TODO: 
# 1. load the reward model 
# 2. load the dataset 
# 3. test the reward model on the dataset 
# take the hh-rlhf dataset, take 100 helpful examples, 100 harmless examples, 
# run reward model on each of the examples and plot the helpful distribution and plot the harmless distribution of rewards 
# ideally, for base model, these are going to be very different because model is biased 
# then, once we add our fairness, we should see that the distributions are much closer 
