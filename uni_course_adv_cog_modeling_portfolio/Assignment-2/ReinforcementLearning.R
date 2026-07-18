pacman::p_load(cmdstanr, LaplacesDemon, stringr, tidyverse, dplyr, gridExtra, scales, posterior)

#---------------------------- Agents ----------------------------

#------ Random agent with bias ------

favourite_hand_agent <- function(bias,noise){
  
  noise_or_not <- rbinom(1,1,noise)
  
  guess <- ifelse(noise_or_not == 0,
                  rbinom(1,1,bias),
                  rbinom(1,1,0.5)
  )
  
  return(guess)
}

#------ Reinforcement learning (RL) agent (as in assignment 1) ------

RL_guessr <- function(prev_outcome, Q_prev, choice_prev, alpha, theta){
  
  prev_outcome <- prev_outcome + 1                      # rename for "arm" space: 1 is left, 2 is right
  
  # Estimating only 1 arm - right
  # They should be re-coded to loss and win in the reward sense
  
  outcome  <- ifelse(prev_outcome == choice_prev, 1, 0)
  Qt <- c(NA,NA)
  Q <- c(NA,NA)
  exp_p <- c(NA,NA)
  p <- c(NA,NA)
  #rW: Qt = Qt-1 + a(rt-1 - Qt-1)
  
  # Rescorla-Wagner learning for 2 hands
  
  for (i in 1:2){
    Qt[i] <- Q_prev[i] + alpha*(outcome - Q_prev[i])
    
    Q[i] <- ifelse(i==choice_prev,                     #only previous choice is updated
                   Qt[i],
                   Q_prev[i])
    
    exp_p[i] <- exp(theta*Q[i])
  }
  
  for (i in 1:2){
    p[i] <- exp_p[i]/sum(exp_p)
  }
  
  guess <- rcat(1,p)-1                                 # -1 to return from [1,2] "arm" space to [0,1] left right space
  return(c(guess,Q))                                   # order guess, previous Q
  
}


#---------------------------- Creating data ---------------------------- 

set.seed(123)

turns <- 120   # 120 trials
bias <- 0.9    # Bias of the opponent
noise <- 0     # No noise
theta <- 1     # Temperature
alpha <- 0.8   # Learning rate

# ------ Initiating the game ------

r <- array(NA,c(2,turns))

# Hider - random biased agent

r[1,1] <- favourite_hand_agent(bias,noise)

# Guesser

Q <- array(NA, c(2,turns))
Q[1,1] <- 0.5
Q[2,1] <- 0.5
r[2,1] <- favourite_hand_agent(0.5, 0.5)

# Game

for (t in 2:turns){
  
  # Hider - biased agent
  r[1,t] <- favourite_hand_agent(bias,noise)
  
  # Guesser - RL agent
  g_res <- RL_guessr(prev_outcome = r[1,t-1],
                     Q_prev = Q[,t-1],
                     choice_prev = r[2,t-1] + 1,
                     alpha = alpha,
                     theta = theta)
  
  # Guessr Q
  r[2,t] <- g_res[1]
  Q[1,t] <- g_res[2]
  Q[2,t] <- g_res[3]
  
}

# Save  the data
data <- list(
  n = 120,
  h = r[2,],
  oh =r[1,],
  k = 2
)

#---------------------------- RL stan logodds model ---------------------------- 

file <- file.path("RL_logodds_commented.stan")

#------ Compile the model ------

mod <- cmdstan_model(file, 
                     cpp_options = list(stan_threads = TRUE), # So we can parallelize the gradient estimations on multiple cores
                     stanc_options = list("O1"))              # A trick to make it faster

#------ Calling Stan with specific options ------

samples <- mod$sample(
  data = data,           # Data
  seed = 123,            # For reproducibility
  chains = 4,            # Number of chains to fit (to check whether they give the same results)
  parallel_chains = 4,   # The number of the chains to be run in parallel
  threads_per_chain = 1, # Distribute gradient estimations within chain across multiple cores
  iter_warmup = 1000,    # Warm-up iterations through which hyperparameters (steps and step length) are adjusted
  iter_sampling = 2000,  # Total number of iterations
  refresh = 200,         # How often to show that iterations have been run
  #output_dir = "simmodels", # saves the samples as csv so it can be later loaded
  max_treedepth = 20,    # Number of steps in the future to check to avoid u-turns
  adapt_delta = 0.99,    # How high a learning rate to adjust hyperparameters during warm-up
)


#---------------------------- Markov Chains ---------------------------- 

# Extract posterior samples and include sampling of the prior

draws_df <- as_draws_df(samples$draws())

# Checking the model's chains
t_theta <- ggplot(draws_df, aes(.iteration, theta, group = factor(.chain), color = factor(.chain))) +
  geom_line() +
  theme_classic() +
  xlab("Iterations") +
  labs(color = "Chain") +
  scale_color_manual(values = c("darkred", "darkgoldenrod1", "#3399FF", "darkolivegreen3")) +
  labs(y = "Theta")

t_alpha <- ggplot(draws_df, aes(.iteration, alpha, group = factor(.chain), color = factor(.chain))) +
  geom_line() +
  theme_classic() +
  xlab("Iterations") +
  labs(color = "Chain") +
  scale_color_manual(values = c("darkred", "darkgoldenrod1", "#3399FF", "darkolivegreen3")) +
  labs(y = "Alpha")


grid.arrange(t_theta, t_alpha, ncol = 1, nrow = 2)

#---------------------------- Prior visualization ---------------------------- 

priors <- samples$draws(
  variables = c("alpha_prior","theta_prior"),
  inc_warmup = FALSE,
  format = "df"
)

# Adjusting the format for visualization:

priors <- priors %>% 
  pivot_longer(cols = c(1,2))

# Subsetting alpha and theta

alpha_priors <- priors %>%
  filter(name == "alpha_prior")

theta_priors <- priors %>%
  filter(name == "theta_prior")

p_alpha_priors <- ggplot(alpha_priors, aes(x = value)) +
  geom_density(fill = "darkgoldenrod1", alpha = 0.6) +
  theme_classic() +
  labs(
    title = "Alpha",
    x = "Prior values",                       
    y = "Density"                                   
  ) +
  theme(
    plot.title = element_text(hjust = 0.5))

p_theta_priors <- ggplot(theta_priors, aes(x = value)) +
  geom_density(fill = "#3399FF", alpha = 0.6) +
  theme_classic() +
  labs(
    title = "Theta",
    x = "Prior values",                       
    y = "Density"                                   
  ) +
  theme(
    plot.title = element_text(hjust = 0.5))

grid.arrange(p_alpha_priors, p_theta_priors, ncol = 2)

#---------------------------- Prior-predictive checks ---------------------------- 

# Get prior predictions for RL agent

# Get varnames that start with prior_preds[2,] 

samples_varnames <- samples$summary()$variable                      # Getting all varnames
prior_pred_varnames <- na.omit(str_extract(samples_varnames,
                                           "^prior_preds\\[2,.*"))
# Extract prior_predictions

prior_predictions <- samples$draws(
  variables = prior_pred_varnames,
  inc_warmup = FALSE,
  format = "df"
)

# Adjusting the format + adding the "turn" value as a separate column

pp_vis <- prior_predictions %>% 
  pivot_longer(cols = seq(1,120,by = 1))

pp_vis <- pp_vis %>% 
  mutate(turn = as.integer(str_extract(name,"(\\d+)(?!.*\\d)"))) #turn is derived from index so "..[1,12]" is the value at turn 12

# The distribution of prior-based hand choices for RL 

pp_vis %>% 
  ggplot(aes(x = value, group = name))+
  geom_density(color="steelblue") +
  theme_classic()

# The same data plotted, this time - geom_bar

pp_vis %>%
  ggplot(aes(x = as.factor(value), fill = as.factor(value))) +  
  geom_bar(stat = "count", show.legend = FALSE, width = 0.3, alpha = 0.6) +  
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +  # Add count labels on bars
  theme_classic() +
  labs(
    title = "Distribution of prior-based choices (0 and 1)",
    x = "Choice",
    y = "Count"
  ) +
  scale_x_discrete(labels = c("0", "1")) +  
  scale_y_continuous(labels = scales::comma) +
  scale_fill_manual(values = c("0" = "darkolivegreen3", "1" = "darkgoldenrod1")) +  # Custom colors for 0 and 1
  theme(
    plot.title = element_text(hjust = 0.5)
  )

# The plot below shows how the mean choices change across all chains as turns increase

RL_mean_prior_choices <- pp_vis %>% 
  group_by(name) %>%
  reframe(mu = mean(value)) %>% 
  mutate(turn = as.integer(str_extract(name,"(\\d+)(?!.*\\d)")))

RL_mean_prior_choices_plot <- ggplot(RL_mean_prior_choices, aes(x=turn, y=mu)) +
  geom_line(size = 0.6, color = "steelblue") +
  geom_hline(yintercept = 0.5, 
             linetype = 2,
             alpha = 0.8, 
             col = "darkorange") +
  theme_classic() +
  labs(
    x = "Turn",
    y = "Mean"
  )

RL_mean_prior_choices_plot

#---------------------------- Prior-posterior updates ---------------------------- 

prior_posterior_updates <- samples$draws(
  variables = c("alpha_prior","theta_prior",
                "alpha_p","theta_l"),
  inc_warmup = FALSE,
  format = "df"
)

# Reformatting to fit the ggplot

prior_posterior_updates <- prior_posterior_updates %>% 
  pivot_longer(cols = c(1,2,3,4))


prior_posterior_updates_df <- prior_posterior_updates %>% 
  mutate(varname = gsub("\\_(.*)","",name),               #first regex removes everything befor "_"
         type = gsub("^[^_]*.","",name),                  #second removes everyhing after "_"
         type = ifelse( type == "prior",type,"posterior"))

# Subsetting alpha and theta

alpha_update <- prior_posterior_updates_df %>%
  filter(name %in% c("alpha_prior", "alpha_p"))

theta_update <- prior_posterior_updates_df %>%
  filter(name %in% c("theta_prior", "theta_l"))

prior_posterior_updates_alpha <- ggplot(alpha_update, aes(x = value, group = type, fill = type))+
  geom_density(alpha= 0.6)+
  scale_fill_manual(values = c("darkolivegreen3","darkgoldenrod1"))+
  theme_classic() +
  labs(
    title = "Alpha",
    x = "Value",
    y = "Density",
    fill = "Type"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

prior_posterior_updates_theta <- ggplot(theta_update, aes(x = value, group = type, fill = type))+
  geom_density(alpha= 0.6)+
  scale_fill_manual(values = c("darkolivegreen3","darkgoldenrod1"))+
  theme_classic() +
  labs(
    title = "Theta",
    x = "Value",
    y = "Density",
    fill = "Type"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

grid.arrange(prior_posterior_updates_alpha, prior_posterior_updates_theta, ncol = 2)

#---------------------------- Posterior-predictive checks ---------------------------- 

# Get posterior_predictions for RL agent
# Get varnames that start with posterior_preds 

posterior_pred_varnames <- na.omit(str_extract(samples_varnames,
                                               "^posterior_preds.*"))
# Extract posterior_predictions

posterior_predictions <- samples$draws(
  variables = posterior_pred_varnames,
  inc_warmup = FALSE,
  format = "df"
)

# Reformatting for plotting

pp2_vis <- posterior_predictions %>% 
  pivot_longer(cols = seq(1,120,by = 1))

pp2_vis <- pp2_vis %>% 
  mutate(turn = as.integer(str_extract(name,"(\\d+)(?!.*\\d)")))

# The distribution of posterior-based hand choices for RL 

pp2_vis %>% 
  ggplot(aes(x = value, group = name))+
  geom_density(color="steelblue") +
  #geom_density(aes(x=r[2,]), colour = "black")+
  theme_classic()

# The same data plotted, this time - geom_bar

pp2_vis %>%
  ggplot(aes(x = as.factor(value), fill = as.factor(value))) +  
  geom_bar(stat = "count", show.legend = FALSE, width = 0.3, alpha = 0.6) +  
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +  # Add count labels on bars
  theme_classic() +
  labs(
    title = "Distribution of posterior-based choices (0 and 1)",
    x = "Choice",
    y = "Count"
  ) +
  scale_x_discrete(labels = c("0", "1")) +  
  scale_y_continuous(labels = scales::comma) +
  scale_fill_manual(values = c("0" = "darkolivegreen3", "1" = "darkgoldenrod1")) +  # Custom colors for 0 and 1
  theme(
    plot.title = element_text(hjust = 0.5)
  )

# The plot below shows how the mean choices change across all chains as turns increase

RL_mean_posterior_choices <- pp2_vis %>% 
  group_by(name) %>%
  reframe(mu = mean(value)) %>% 
  mutate(turn = as.integer(str_extract(name,"(\\d+)(?!.*\\d)")))

RL_mean_posterior_choices_plot <- ggplot(RL_mean_posterior_choices, aes(x=turn, y=mu)) +
  geom_line(size = 0.6, color = "steelblue") +
  geom_hline(yintercept = 0.5, 
             linetype = 2,
             alpha = 0.8, 
             col = "darkorange") +
  theme_classic() +
  labs(
    x = "Turn",
    y = "Mean"
  )

RL_mean_posterior_choices_plot
