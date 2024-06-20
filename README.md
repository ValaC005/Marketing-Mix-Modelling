# Robyn MMM Quick Guide

This repository contains a quick guide for setting up and running the Robyn Marketing Mix Modeling (MMM) tool, version 3.8.1, developed by Meta Platforms, Inc. The guide includes steps for setting up the environment, loading data, configuring hyperparameters, running the model, and analyzing the results.

## Table of Contents

1. [Setup Environment](#setup-environment)
2. [Load Data](#load-data)
3. [Configure Hyperparameters](#configure-hyperparameters)
4. [Run Model](#run-model)
5. [Select and Save Model](#select-and-save-model)
6. [Budget Allocation](#budget-allocation)

## Setup Environment

First, ensure you have the necessary libraries and packages installed. Follow these steps to set up your environment:

```r
library(usethis)
usethis::edit_r_environ()

# Install the stable version from CRAN
# install.packages("Robyn")

# Install the dev version from GitHub
# install.packages("remotes") # Install remotes first if you haven't already
remotes::install_github("facebookexperimental/Robyn/R")
library(Robyn)

# Check if you have installed the latest version
packageVersion("Robyn")

# Force multicore when using RStudio
Sys.setenv(R_PARALLELLY_FORK_ENABLE = "true")
options(parallelly.fork.enable = TRUE)

# Install and load the python library Nevergrad
# install.packages("reticulate") # Install reticulate first if you haven't already
library("reticulate")
use_python("~/Library/r-miniconda-arm64/envs/r-reticulate/bin/python", required = TRUE)
py_install("nevergrad", pip = TRUE)
py_config() # Check your python version and configurations

# Load additional libraries
library(stringr)
library(lubridate, warn.conflicts = FALSE)
library(foreach)
library(future)
library(doFuture)
library(rngtools)
library(doRNG)
library(glmnet)
library(car)
library(StanHeaders)
library(prophet, warn.conflicts = FALSE)
library(rstan)
library(ggplot2)
library(gridExtra)
library(grid)
library(ggpubr)
library(see)
library(PerformanceAnalytics, warn.conflicts = FALSE)
library(nloptr)
library(minpack.lm)
library(rPref, warn.conflicts = FALSE)
library(reticulate)
library(rstudioapi)
library(readr)
library(remote)
library(prophet)
```

## Load Data

Load the data required for the model:

```r
data("dt_simulated_weekly")
head(dt_simulated_weekly)

dt_input <- read_csv("data_mmm.csv")
head(dt_input)
View(dt_input)

# Check holidays from Prophet
data("dt_prophet_holidays")
head(dt_prophet_holidays)
```

## Configure Hyperparameters

Set up the hyperparameters for the model:

```r
InputCollect_Geo <- robyn_inputs(
  dt_input = dt_input,
  dt_holidays = dt_prophet_holidays,
  date_var = "date",
  dep_var = "sales",
  dep_var_type = "revenue",
  prophet_vars = c("trend", "season", "weekday", "holiday"),
  prophet_country = "US",
  context_vars = c("unemployment", "temperature"),
  paid_media_spends = c("newspaper_spend", "tv_spend"),
  paid_media_vars = c("newspaper_readership", "tv_gross_rating_points"),
  adstock = "geometric"
)

print(InputCollect_Geo)

plot_adstock(plot = TRUE)
plot_saturation(plot = TRUE)

hyper_names(adstock = InputCollect_Geo$adstock, all_media = InputCollect_Geo$all_media)

hyperparameters_geo <- list(
  newspaper_spend_alphas = c(0.5, 3),
  newspaper_spend_gammas = c(0.3, 1),
  newspaper_spend_thetas = c(0.1, 0.4),
  tv_spend_alphas = c(0.5, 3),
  tv_spend_gammas = c(0.3, 1),
  tv_spend_thetas = c(0.3, 0.8),
  train_size = c(0.5, 0.8)
)

InputCollect_Geo <- robyn_inputs(
  dt_input = dt_input,
  dt_holidays = dt_prophet_holidays,
  date_var = "date",
  dep_var = "sales",
  dep_var_type = "revenue",
  prophet_vars = c("trend", "season", "weekday", "holiday"),
  prophet_country = "US",
  context_vars = c("unemployment", "temperature"),
  paid_media_spends = c("newspaper_spend", "tv_spend"),
  paid_media_vars = c("newspaper_readership", "tv_gross_rating_points"),
  window_start = "2019-07-01",
  window_end = "2022-07-01",
  adstock = "geometric"
)

InputCollect_Geo <- robyn_inputs(InputCollect = InputCollect_Geo, hyperparameters = hyperparameters_geo)
print(InputCollect_Geo)

# Check spend exposure fit if available
if (length(InputCollect_Geo$exposure_vars) > 0) {
  lapply(InputCollect_Geo$modNLS$plots, plot)
}

# Manually save and import InputCollect as JSON file
robyn_write(InputCollect_Geo, dir = "/Users/chavi/Desktop/MMM/MMM_Meta/Geometric")
InputCollect_Geo <- robyn_inputs(dt_input = dt_input,
                             dt_holidays = dt_prophet_holidays,
                             json_file = "/Users/chavi/Desktop/MMM/MMM_Meta/Geometric/RobynModel-inputs.json")
```

## Run Model

Run the model with the specified inputs and hyperparameters:

```r
OutputModels <- robyn_run(
  InputCollect = InputCollect_Geo,
  iterations = 5000, 
  trials = 8, 
  ts_validation = FALSE,
  outputs = FALSE
)
print(OutputModels)

ts_validation(OutputModels)

# Check MOO (multi-objective optimization) convergence plots
OutputModels$convergence$moo_distrb_plot
OutputModels$convergence$moo_cloud_plot

# Check time-series validation plot (when ts_validation == TRUE)
if (OutputModels$ts_validation) OutputModels$ts_validation_plot

robyn_object2 <- "/Users/chavi/Desktop/MMM/MMM_Meta/Geometric/MyRobyn2.RDS"

# Calculate Pareto optimality, cluster and export results and plots
OutputCollect_geo <- robyn_outputs(
  InputCollect_Geo, OutputModels,
  pareto_fronts = "auto",
  calibration_constraint = c(0.01, 0.1),
  csv_out = "pareto",
  clusters = TRUE,
  plot_pareto = TRUE,
  plot_folder = robyn_object2,
  export = TRUE
)
print(OutputCollect_geo)
```

## Select and Save Model

Select and save the model that best reflects your business reality:

```r
print(OutputCollect_geo)
select_model <- "1_357_4" # Pick one of the models from OutputCollect to proceed

# JSON export and import
ExportedModel <- robyn_write(InputCollect_Geo, OutputCollect_geo, select_model)
print(ExportedModel)

# Deprecated method
ExportedModelOld <- robyn_save(
   robyn_object = robyn_object,
   select_model = select_model,
   InputCollect = InputCollect_Geo,
   OutputCollect = OutputCollect_geo)
print(ExportedModelOld)
plot(ExportedModelOld)
```

## Budget Allocation

Get budget allocation based on the selected model:

```r
# Check media summary for selected model
print(ExportedModel)

# Run the "max_historical_response" scenario
AllocatorCollect1 <- robyn_allocator(
  InputCollect = InputCollect_Geo,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  scenario = "max_historical_response",
  channel_constr_low = 0.7,
  channel_constr_up = c(1.5, 1.5),
  export = TRUE,
  date_min = "2019-07-01",
  date_max = "2022-07-01"
)
print(AllocatorCollect1)
plot(AllocatorCollect1)

# Run the "max_response_expected_spend" scenario
AllocatorCollect2 <- robyn_allocator(
  InputCollect = InputCollect_Geo,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  scenario = "max_response_expected_spend",
  channel_constr_low = c(0.7, 0.7),
  channel_constr_up = c(1.5, 1.5),
  expected_spend = 300000,
  expected_spend_days = 9,
  export = TRUE
)
print(AllocatorCollect2)
AllocatorCollect2$dt_optimOut
plot(AllocatorCollect2)

# QA optimal response
select_media <- "newspaper_spend"
metric_value <- AllocatorCollect1$dt_optimOut$optmSpendUnit[
  AllocatorCollect1$dt_optimOut$channels == select_media
]; metric_value

if (TRUE) {
  optimal_response_allocator <- AllocatorCollect1$dt_optimOut$optmResponseUnit[
    AllocatorCollect1$dt_optimOut$channels == select_media
  ]
  optimal_response <- robyn_response(
    InputCollect = InputCollect_Geo,
    OutputCollect = OutputCollect_geo,
    select_model = select_model,
    select_build = 0,
    media_metric = select_media,
    metric_value = metric_value
  )
  plot(optimal_response$plot)
  if (length(optimal_response_allocator) > 0) {
    cat("QA if results from robyn_allocator and robyn_response agree: ")
    cat(round(optimal_response_allocator) == round(optimal_response$response), "( ")
    cat(optimal_response$response, "==", optimal_response_allocator, ")\n")
  }
}

AllocatorCollect3 <- robyn_allocator(
  InputCollect = InputCollect_Geo,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  scenario = "max_response_expected_spend",
  channel_constr_low = c(0.7, 0.7),
  channel_constr_up = c(1.5, 1.5),
  expected_spend = 300000,
  expected_spend_days = 9
)
print(AllocatorCollect3)
plot(AllocatorCollect3)

# Get response for 50k from result saved in robyn_object
Spend1 <- 50000
Response1 <- robyn_response(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  media_metric = "newspaper_spend",
  metric_value = Spend1
)
Response1$response / Spend1
Response1$plot

# Get response for +10%
Spend2 <- Spend1 * 1.1
Response2 <- robyn_response(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  media_metric = "newspaper_spend",
  metric_value = Spend2
)
Response2$response / Spend2
Response2$plot

# Marginal ROI of next 1000$ from 80k spend level for search
(Response2$response - Response1$response) / (Spend2 - Spend1)

# Example of getting paid media exposure response curves
imps <- 100000
response_imps <- robyn_response(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  media_metric = "newspaper_readership",
  metric_value = imps
)
response_imps$response / imps * 1000
response_imps$plot
```

## License

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
