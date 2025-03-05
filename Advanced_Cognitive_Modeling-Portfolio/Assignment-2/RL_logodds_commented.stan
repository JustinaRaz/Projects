/*
The data block below specifies the input data.

- k: number of possible hands (min. 1, integer)
- n: total number of trials (min. 1, integer)
- oh: hand choices of the other participant 
    Expects an array of integer values between 0 and 1, the size of the array depends on trial number.
    Values: 0 and 1.
- h: modeled participant's (RL) hand choices
    Expects an array of integer values between 0 and 1, the size of the array depends on trial number.
    Values: 0 and 1.
*/

data{
  int<lower=1> k, n;
  array[n] int<lower=0, upper=1> oh, h;
}


/*
The transformed data block below processes raw input data for modelling.

- outcome: array of integers (len = n) stores values of whether the RL agent's hand choices match the opponent's in each trial.

If the chosen RL's hand [h] on trial [i] matches the opponent's hand with a coin, outcome = 1.
If there is no match on trial [i], outcome = 0.

*/

transformed data{
  array[n] int outcome;
  for (i in 1:n) {
    if (h[i]==oh[i]){
      outcome[i] = 1;
    }
    else if (h[i]!=oh[i]){
      outcome[i] = 0;
    }
  }
}


/*
The parameters block below defines the parameters of the Stan model that will be estimated.

- alpha & theta: defined as real unbounded values (as we are working on continuous scale, -inf to +inf).

*/

parameters{
  real alpha, theta;
}


/*
The tranformed parameters block

- Q: expected value of choosing right (1) or left (0) hand.
- p: the probability (after the softmax function) of choice right or left.
- exp_p: intermediate step in the softmax function.
- alpha_p: alpha (learning rate) on probability scale.
- theta_l: 0 to inf - inverse temperature parameter in a softmax function (controls exploration versus exploitation). Low value - more exploration, high value - more exploitation.

*/

transformed parameters {
    matrix[k, n] Q, exp_p, p;                         // Matrix that stores the values for both hands across n trials for Q, p and exp_p.
    real<lower=0,upper=1> alpha_p = inv_logit(alpha); // Using inverse logistic function to bound alpha between 0 and 1. This makes large negative values become closer to 0, and large positive values closer to 1.
    real<lower=0> theta_l = exp(theta);               // Ensuring that theta is always positive number (temperature cannot be negative).
  
    // Q for 1st trial:
    for (j in 1:k) {
        Q[j, 1] = 0.5;                               // Set initial Q for both hands [k].
        exp_p[j,1]= exp(theta_l*Q[j,1]);             // Computing exponentiated Q value, weighted by inverse temperature [theta_l]. Part of softmax function.
    }
    
    // p for first trial:
    for (j in 1:k){                                 
      p[j,1] = exp_p[j,1]/sum(exp_p[,1]);            // Applying softmax f. to calculate probability of choosing each hand at 1st trial [n].
      
    }

    // Computing Qs and probabilities for the rest of trials:
    for (i in 2:n) {                                // Looping trials [i] from 2 to n.
        for (j in 1:k) {                            // Looping for each hand [j] at trial [i].
            Q[j, i] = Q[j, i - 1];                  // Carry forward previous Qs to be able to adjust them from previous to current trial.
            
          if(h[i-1]+1==j){                          // Update Q-value for the chosen hand. If the RL agent's previous choice is the same as the current choice, the Q for that hand is updated
            Q[j,i] = Q[j, i - 1] + alpha_p * (outcome[i - 1] - Q[j, i - 1]);   // Delta rule (Rescorla-Wagner) to update Q.
          
          }
          else if (h[i-1]+1!=j){                    // If choice is not the same, the Q value for that hand remains unchanged.
            Q[j,i] = Q[j,i];
          
          }
        
          exp_p[j,i]= exp(theta_l*Q[j,i]);         // Intermediate step for softmax.
          
        }
    
    for (j in 1:k){
      p[j,i] = exp_p[j,i]/sum(exp_p[,i]);          // Calculating the probability of choosing each hand at trial [i].
    }
  }
}


/*
The model block - defines how the parameters are estimated using Bayesian inference.

Priors are drawn from a normal distribution with a mean of 0 and SD of 1.
Alpha: In logit space, -3 to 3 covers the probability space roughly between 0 and 1.

Theta: In log space, it allows for values between 0 and above 1, but theta doesn't become too large.

The choice is distributed by categorical distribution with probabilities calculated in "transformed parameters" code block. 
The choice from the data is drawn from this distribution every turn. Our data consists of 0s and 1s, however, the categorical distribution excpects values above 0 - that
is why we add +1 to h.
*/

model {
  
  //priors
  target += normal_lpdf(alpha|0,1);         // follows standard normal distribution, mean = 0, SD = 1.
  target += normal_lpdf(theta|0,1);         // follows standard normal distribution, mean = 0, SD = 1.
  
  // Turn-by-turn estimation
  for (i in 1:n){                          // loops over trials
  vector[k] x = [p[1,i],p[2,i]]';          // takes softmax probabilities of choosing each hand at trial i and creates a vector of those probabilities.
  target += categorical_lpmf(h[i]+1| x);   // computes log-likelihood - the probability of the choice h[i] givent the probability vector [x]. "+1" - to transform 0s and 1s to 1st and 2s, respectively.
  }
}


/*
The generated quantities block - for prior predictive checks and posterior predictive checks for model validation.

We are drawing priors from the same distributions as in the "model" block. 

For prior predictions, we are simulating the opponent's choices as well, as the RL agent cannot "play" by itself.

For posterior predictions, we are using the same posteriors as in the model block, but we are not estimating the choice from categorical distribution, but we are
generating the choice based on the model estimates.

*/ 

generated quantities {
  
  // Prior predictive checks:
  
  real<lower=0, upper=1> alpha_prior;        // Defines prior for alpha [learning rate], bounds between 0 and 1.
  alpha_prior = inv_logit(normal_rng(0,1));  // alpha_prior can be any random number drawn from normal distribution with mean=0 and SD=1.
                                             // inv_logit ensures alpha_prior values between 0 and 1.
  
  real<lower=0> theta_prior;                 // The same as for alpha_prior, except that in this case, theta_prior values are between 0 and +inf.
  theta_prior = exp(normal_rng(0,1));
  
  real<lower=0, upper=1> bias_prior;         // Bias for opponent, bounded between 0 and 1.
  bias_prior = beta_rng(1,1);                // drawn from uniform beta distribution, can take random value between 0 and 1. However, beta_rng(1,1) ensures that 
                                             // any value of bias between 0 and 1 is equally likely. 
  
  // Make prior predictions:
  
  // prior_preds[1,] : biased opponent's choice predictions
  // prior_preds[2,] : RL agent's prior-based choice predictions
  // prior_preds[3,] : outcome calculated from [1,] and [2,] - whether biased opponent's and RL agent's choices match [1] or not[0].
  
  matrix[3,n] prior_preds;                  // initiates a matrix with 3 rows (to store prior predictions explained above) 
  matrix[k,n] pp_Q, pp_exp_p, pp_p;         // pp_Q - expected Q values, pp_exp_p - exponentiated Q values, pp_p - probabilities from softmax.
  
  // First trial, biased opponent - prior_preds[1,] :

  prior_preds[1,1] = bernoulli_rng(bias_prior);      // Biased opponent's hand choice. Assigns a random binary outcome [either 0 or 1] for the first trial, based on bias_prior
  
  // First trial, RL agent - prior_preds[2,] :
  
  for (j in 1:k) {
      pp_Q[j, 1] = 0.5;                            // Set initial Q for both arms
      pp_exp_p[j,1]= exp(theta_prior*pp_Q[j,1]);   // calculation is the same as in transformed parameters block, just with prior prediction values.
      
  }
  for (j in 1:k){
    pp_p[j,1] = pp_exp_p[j,1]/sum(pp_exp_p[,1]);   // calculation is the same as in transformed parameters block, just with prior prediction values.
    
  }
  
  vector[k] pp_p_turnwise_1 = [pp_p[1,1],pp_p[2,1]]';      // a vector of probabilities for RL agent to choose the left pp_p[1,1] or right pp_p[2,1] hand at the first trial.
  prior_preds[2,1]= categorical_rng(pp_p_turnwise_1) - 1;  // Evaluates RL agent's hand choice based on probabilities and adds it to the RL agent's prior-based choice prediction vector.
                                                           // If pp_p_turnwise_1 = [0.6, 0.4], the categorical_rng(pp_p_turnwise_1) will return 1 with 60% probability,
                                                           // and 2 with 40% probability. However, -1 transforms 1 to 0, and 2 to 1, to align with the model's expectations.
  
  // Calculating the outcome - prior_preds[3,] :
  
  if (prior_preds[1,1] == prior_preds[2,1]){             // If the biased opponent's choice at trial 1 is the same as RL agent's choice
    prior_preds[3,1] = 1;                                // The outcome is 1 (win)
  } else {
     prior_preds[3,1] = 0;                               // If choices do not match, the outcome is 0.
  }

  
  // Calculating values for the rest of the rounds:
  
  for (i in 2:n){
    
    // Biased opponent (as previously, for the rest of trials):
    
    prior_preds[1,i] = bernoulli_rng(bias_prior);
    
    // RL agent (as previously, for the rest of trials):
    
    for (j in 1:k) {
      
      if(prior_preds[2,i-1]+1==j){
        pp_Q[j,i] = pp_Q[j, i - 1] + alpha_prior * (prior_preds[3,i - 1] - pp_Q[j, i - 1]);
      }
      else {
        pp_Q[j,i] = pp_Q[j,i-1];
      }
        pp_exp_p[j,i]= exp(theta_prior*pp_Q[j,i]);
    }
    
    for (j in 1:k){
    pp_p[j,i] = pp_exp_p[j,i]/sum(pp_exp_p[,i]);
    }
    
    // Determining RL agent's choice:
    
    vector[k] pp_p_turnwise = [pp_p[1,i],pp_p[2,i]]'; 
    prior_preds[2,i]= categorical_rng(pp_p_turnwise) - 1;
    
   // Calculating the outcome:
   
   if(prior_preds[1,i] == prior_preds[2,i]){
    prior_preds[3,i] = 1;
    } else {
     prior_preds[3,i] = 0;
    }
  }
  
  // Posterior predictive checks: 

  vector[n] posterior_preds;                                // Vector for storing predicted outcomes for all trials based on the model's posterior distribution.
  
  for (i in 1:n){
  vector[k] posterior_p = [p[1,i],p[2,i]]';                 // Creates a vector of posterior probabilities of choices
  posterior_preds[i] = categorical_rng(posterior_p) - 1;    // Contains agent's choices based on posterior probabilities
  }
  
}