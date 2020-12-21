################################################################################
# Movielens Project - HarvardX - Data Science Capstone
################################################################################

# This code is one of the delivered items of the first
# project in the Data Science Capstone course by Harvardx.
#
# By using part of the Movielens dataset, the goal is to
# build a model that will predict the ratings users will
# give to movies.
#
# The code presented in Part 1 of the experiment was put
# together by the course's staff and it creates the dataset
# that has to be used, which is split into edx (for training)
# and validation (for calculating the performance of the
# chosen model). In Part 2, the edx dataset was further
# divided into train and test, with the latter allowing
# the evaluation of the constructed models without the 
# usage of the validation set.
#
# Parts 3-12 build and test various models, from a simple
# one that only uses the total average as the predicted
# rating, to models that try to capture the influence users,
# movies, genres, rating week, and movie year exert over the 
# ratings. Concepts like regularization and linear regression 
# are used, and multiple values for the regularization parameter are 
# tested.
#
# Parts 13 and 14 close out the experiment. The first by
# obtaining the best model, according to the test dataset,
# out of the more than 100 that are built, and the second
# by using that best model on the validation dataset to
# obtain the final result, which is reported in RMSE.


################################################################################
# Part 1 - Downloading and Defining the Dataset
################################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(caret)
library(lubridate)
library(stringr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()

download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



################################################################################
# Part 2 - Experiment Setup
################################################################################

# Likewise, the training dataset (edx) must be split into
# train and test. This is done so we can evaluate all models on the test set
# to pick the best one before finally using it on the validation dataset.
# Before doing the split, though, a couple of columns of the dataset are
# transformed (the week in which the rating happened is obtained out of the
# timestamp column and the year is extracted out of the movie title).
# These columns will be used in the construction of the models.
edx <- edx %>%
  mutate(rating_week = round_date(as_datetime(timestamp), unit = "week")) %>%
  extract(title, "movie_year", regex = "\\(([0-9 \\-]*)\\)$", remove = F)

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_ds <- edx[-test_index,]
test_ds <- edx[test_index,]

# The following rows are the setups steps for the tests that will be done.
# For the models being built in this code, the total average of the ratings
# is necessary. The average is taken from the train dataset.
total_avg <- mean(train_ds$rating)

# As it is done with the edx and validation sets, we want to make sure all
# entities that will be used in the creation of the models are in both datasets.
# So these joins are done to eliminate rows from the test set that do not meet
# the criteria.
test_ds <- test_ds %>% 
  semi_join(train_ds, by = "movieId") %>%
  semi_join(train_ds, by = "userId") %>%
  semi_join(train_ds, by = "genres") %>%
  semi_join(train_ds, by = "rating_week") %>%
  semi_join(train_ds, by = "movie_year")



################################################################################
# Part 3 - Experiment - The Average Model
################################################################################

# The average model is the simplest one we could build. It rates all movies 
# according to the current total average. As it will be done throughout the 
# experiment, the results will be measured with RMSE and
# then appended to a data frame that will store the experiment results.

# Note that the experiment is done with test_ds.
avg_model_predictions <- rep(total_avg, times=nrow(test_ds))

rmse <- RMSE(total_avg, test_ds$rating)

final_results <- data.frame(Model='Average', Alpha='N/A', RMSE=rmse)

rm(avg_model_predictions)



################################################################################
# Part 4 - Experiment - The User Model
################################################################################

# In this experiment, the total average is also used. It is, however,
# combined with the user bias. It is assumed every user has their own
# rating tendencies and we want to take it into account when predicting the
# given ratings.

# This function will be used repeatedly, with only alpha (the regularization
# parameter) being altered for every experiment.
build_test_user_model <- function(alpha) {
  
  # Here the user dataframe is created. The goal here is to calculate how the
  # ratings given by the users are distant from the total average. In such
  # a way, we intend to capture the bias every user has (user_effect).
  # The calculation is done by taking the average of each user's ratings, which
  # are centered around zero according to the total dataset average. To avoid
  # users with just a few ratings to influence the predictions too much,
  # regularization is used via the parameter alpha and n_user, which indicates
  # how many ratings the user has given. The more ratings give by the user,
  # the less his average will be affected by regularization.
  user_data_frame <- train_ds %>% 
    group_by(userId) %>% 
    summarize(user_avg = mean(sum(rating - total_avg)), n_user=n(), user_effect = user_avg/(n_user + alpha)) %>%
    select(userId, user_effect)
  
  # The prediction is done by taking the user_effect calculated previously via
  # the train dataset and applying it to the test dataset. The predictions, then
  # will be given by the total average of the train dataset plus the effect 
  #(bias) of each user.
  predicted_ratings <- test_ds %>% 
    inner_join(user_data_frame) %>%
    mutate(predicted_ratings = total_avg + user_effect) %>%
    pull(predicted_ratings)
  
  # Results are calculated and returned.
  rmse <- RMSE(predicted_ratings, test_ds$rating)
  
  c(Model='User Effect Model', Alpha=as.character(alpha), RMSE=rmse)
}

# The function is executed for regularizarion parameter (alpha)
# between 0 and 30 (in increments of 0.25). The higher the parameter,
# the stronger the regularization. Returned results are appended to
# dataset.
new_results <- sapply(seq(0, 30, 0.25), build_test_user_model)
new_results <- as.data.frame(t(new_results))
                   
final_results <- rbind(final_results, new_results)



################################################################################
# Part 5 - Experiment - The Movie Model
################################################################################

# Now, it's time to calculate effect movies have on ratings. The calculations
# as well as the logic behind them are the same ones that apply to the
# user model: the idea each movie has its own capacity to affect its ratings
# (its quality). Here we look to measuer it and apply it to predictions.

# This function will be used repeatedly, with only alpha (the regularization
# parameter) being altered for every experiment.
build_test_movie_model <- function(alpha) {
  
  # Here the movie dataframe is created. The goal is to calculate how the
  # ratings gotten by the movies are distant from the total average. In such
  # a way, we intend to capture the influence every movie has (movie_effect).
  # The calculation is done by taking the average of each movie's ratings, which
  # are centered around zero according to the total dataset average. To avoid
  # movies with just a few ratings to influence the predictions too much,
  # regularization is used via the parameter alpha and n_movie, which indicates
  # how many ratings the movie has gotten The more ratings,
  # the less its average will be affected by regularization.
  movie_data_frame <- train_ds %>% 
    group_by(movieId) %>% 
    summarize(movie_avg = mean(sum(rating - total_avg)), n_movie=n(), movie_effect = movie_avg/(n_movie + alpha)) %>%
    select(movieId, movie_effect)
  
  # The prediction is done by taking the movie_effect calculated previously via
  # the train dataset and applying it to the test dataset. The predictions, then
  # will be given by the total average of the train dataset plus the effect 
  # (bias) of each movie.
  predicted_ratings <- test_ds %>% 
    inner_join(movie_data_frame) %>%
    mutate(predicted_ratings = total_avg + movie_effect) %>%
    pull(predicted_ratings)
  
  # Results are calculated and returned.
  rmse <- RMSE(predicted_ratings, test_ds$rating)
  
  c(Model='Movie Effect Model', Alpha=as.character(alpha), RMSE=rmse)
}

# The function is executed for regularizarion parameter (alpha)
# between 0 and 30 (in increments of 0.25). The higher the parameter,
# the stronger the regularization. Returned results are appended to
# dataset.
new_results <- sapply(seq(0, 30, 0.25), build_test_movie_model)
new_results <- as.data.frame(t(new_results))

final_results <- rbind(final_results, new_results)



################################################################################
# Part 6 - Experiment - The Genre Model
################################################################################

# The genre model works following the same idea as the one used in the user and
# movies ones. The point is to extract, if there is any, the effect genres have
# on movie ratings, and use that knowledge to make predictions on the test set.

# This function will be used repeatedly, with only alpha (the regularization
# parameter) being altered for every experiment.
build_test_genre_model <- function(alpha) {
  
  # Here the genre dataframe is created. The goal is to calculate how the
  # ratings for the movies of each genre are distant from the total average. 
  # We intend to capture the influence every genre has (genre_effect).
  # The calculation is done by taking the average of each genre's ratings, which
  # are centered around zero according to the total dataset average. To avoid
  # movie genres with just a few ratings to influence the predictions too much,
  # regularization is used via the parameter alpha and n_genre, which indicates
  # how many ratings movies of the genre have received The more ratings,
  # the less its average will be affected by regularization.
  genre_data_frame <- train_ds %>% 
    group_by(genres) %>% 
    summarize(genre_avg = mean(sum(rating - total_avg)), n_genre=n(), genre_effect = genre_avg/(n_genre + alpha)) %>%
    select(genres, genre_effect)
  
  # The prediction is done by taking the genre_effect calculated previously via
  # the train dataset and applying it to the test dataset. The predictions, then
  # will be given by the total average of the train dataset plus the effect 
  # (bias) of each genre.
  predicted_ratings <- test_ds %>% 
    inner_join(genre_data_frame) %>%
    mutate(predicted_ratings = total_avg + genre_effect) %>%
    pull(predicted_ratings)
  
  # Results are calculated and returned.
  rmse <- RMSE(predicted_ratings, test_ds$rating)
  
  c(Model='Genre Effect Model', Alpha=as.character(alpha), RMSE=rmse)
}

# The function is executed for regularizarion parameter (alpha)
# between 0 and 30 (in increments of 0.25). The higher the parameter,
# the stronger the regularization. Returned results are appended to
# dataset.
new_results <-  sapply(seq(0, 30, 0.25), build_test_genre_model)
new_results <- as.data.frame(t(new_results))

final_results <- rbind(final_results, new_results)



################################################################################
# Part 7 - Experiment - The Movie Year Model
################################################################################

# The movie year model's point is to extract, if there is any, the effect the release
# year of movies has on the ratings they receive, and use that knowledge to 
# make predictions on the test set.

# This function will be used repeatedly, with only alpha (the regularization
# parameter) being altered for every experiment.
build_test_movie_year_model <- function(alpha) {
  
  # Here the movie year dataframe is created. The goal is to calculate how the
  # averages of movies released in a certain year are distant from the total average. 
  # We intend to capture the influence every release year has (movie_year_effect).
  # The calculation is done by taking the average of each year's ratings, which
  # are centered around zero according to the total dataset average. To avoid
  # release years with just a few ratings to influence the predictions too much,
  # regularization is used via the parameter alpha and n_movie_year, which indicates
  # how many ratings movies of that year have received. The more ratings,
  # the less its average will be affected by regularization.
  movie_year_data_frame <- train_ds %>%
    group_by(movie_year) %>% 
    summarize(movie_year_avg = mean(sum(rating - total_avg)), n_movie_year=n(), movie_year_effect = movie_year_avg/(n_movie_year + alpha)) %>%
    select(movie_year, movie_year_effect)
  
  # The prediction is done by taking the movie_year_effect calculated previously via
  # the train dataset and applying it to the test dataset. The predictions, then
  # will be given by the total average of the train dataset plus the effect 
  # (bias) of each release year
  predicted_ratings <- test_ds %>% 
    inner_join(movie_year_data_frame) %>%
    mutate(predicted_ratings = total_avg + movie_year_effect) %>%
    pull(predicted_ratings)
  
  # Results are calculated and returned.
  rmse <- RMSE(predicted_ratings, test_ds$rating)
  
  c(Model='Movie Year Effect Model', Alpha=as.character(alpha), RMSE=rmse)
}

# The function is executed for regularizarion parameter (alpha)
# between 0 and 30 (in increments of 0.25). The higher the parameter,
# the stronger the regularization. Returned results are appended to
# dataset.
new_results <- sapply(seq(0, 30, 0.25), build_test_movie_year_model)
new_results <- as.data.frame(t(new_results))

final_results <- rbind(final_results, new_results)



################################################################################
# Part 8 - Experiment - The Rating Week Model
################################################################################

# The rating week model's point is to extract, if there is any, the effect the
# week in which the rating was given has on the ratings, and use 
# that knowledge to make predictions on the test set.

# This function will be used repeatedly, with only alpha (the regularization
# parameter) being altered for every experiment.
build_test_rating_week_model <- function(alpha) {
  
  # Here the rating week dataframe is created. The goal is to calculate how the
  # averages of movies rated in a certain week are distant from the total average. 
  # We intend to capture the influence every week has (rating_week_effect).
  # The calculation is done by taking the average of each week's ratings, which
  # are centered around zero according to the total dataset average. To avoid
  # rating weeks with fewer ratings to influence the predictions too much,
  # regularization is used via the parameter alpha and n_rating_week, which indicates
  # how many ratings were given to movies in that week. The more ratings,
  # the less its average will be affected by regularization.
  week_data_frame <- train_ds %>%
    group_by(rating_week) %>% 
    summarize(rating_week_avg = mean(sum(rating - total_avg)), n_rating_week=n(), rating_week_effect = rating_week_avg/(n_rating_week + alpha)) %>%
    select(rating_week, rating_week_effect)
  
  # The prediction is done by taking the rating_week_effect calculated previously via
  # the train dataset and applying it to the test dataset. The predictions, then
  # will be given by the total average of the train dataset plus the effect 
  # (bias) of each release week.
  predicted_ratings <- test_ds %>% 
    inner_join(week_data_frame) %>%
    mutate(predicted_ratings = total_avg + rating_week_effect) %>%
    pull(predicted_ratings)
  
  # Results are calculated and returned.
  rmse <- RMSE(predicted_ratings, test_ds$rating)
  
  c(Model='Rating Week Effect Model', Alpha=as.character(alpha), RMSE=rmse)
}

# The function is executed for regularizarion parameter (alpha)
# between 0 and 30 (in increments of 0.25). The higher the parameter,
# the stronger the regularization. Returned results are appended to
# dataset.
new_results <- sapply(seq(0, 30, 0.25), build_test_rating_week_model)
new_results <- as.data.frame(t(new_results))

final_results <- rbind(final_results, new_results)



################################################################################
# Part 9 - Experiment - The Movie + User Model
################################################################################

# This model attempts to simultaneously capture the effect the
# movie being rated and the user doing the rating will have over each score,
# and then summing those effects to the total average to estimate the final
# score.
build_test_movie_user_model <- function(alpha) {
  
  # The Movie + User model starts out in the same way observed in the previous
  # experiments. That is, we begin by grouping movies, calculating their
  # average rating (centered around the total train_ds average), and then
  # applying regularization with alpha and n_movie (the total number or
  # ratings received by that movie.)
  movie_data_frame <- train_ds %>% 
    group_by(movieId) %>% 
    summarize(movie_avg = mean(sum(rating - total_avg)), n_movie=n(), movie_effect = movie_avg/(n_movie + alpha)) %>%
    select(movieId, movie_effect)
  
  # The second step is doing the same, but for the user. The twist, in this
  # case, is that this time around we are not merely grouping by ID and
  # averaging ratings. Since we already have, in the previous step, calculated
  # the movie effect, when trying to do the same for the user we actually
  # subtract the movie effect from each movie rating before grouping by user.
  # By doing so, theoretically, we are eliminating the effect each movie
  # has over the rating and extracting only the effect the user has over the
  # score. It is for that reason that a join is made between train_ds and
  # movie_data_frame before doing the calculations of average and 
  # regularization.
  
  # Note that the user the order of the operations could have been inverted,
  # with the user effect being calculated first. But since, via testing, the
  # movie effect showed to be larger, a choice was made to take it out first.
  # And, indeed, tests proved that yielded the best result.
  user_data_frame <- train_ds %>% 
    inner_join(movie_data_frame) %>%
    mutate(rating = rating - movie_effect - total_avg) %>%
    group_by(userId) %>% 
    summarize(user_avg = mean(sum(rating)), n_user=n(), user_effect = user_avg/(n_user + alpha)) %>%
    select(userId, user_effect)
  
  # Scores are predicted on the test set. Each given score is a combination of
  # the total average observed in train_ds (total_avg), the movie effect, and
  # the user effect.
  predicted_ratings <- test_ds %>% 
    inner_join(user_data_frame) %>%
    inner_join(movie_data_frame) %>%
    mutate(predicted_ratings = total_avg + movie_effect + user_effect) %>%
    pull(predicted_ratings)
  
  # Results are calculated and returned.
  rmse <- RMSE(predicted_ratings, test_ds$rating)
  
  c(Model='Movie + User Effect Model', Alpha=as.character(alpha), RMSE=rmse)
}

# The function is executed for regularizarion parameter (alpha)
# between 0 and 30 (in increments of 0.25). The higher the parameter,
# the stronger the regularization. Returned results are appended to
# dataset.
new_results <- sapply(seq(0, 30, 0.25), build_test_movie_user_model)
new_results <- as.data.frame(t(new_results))

final_results <- rbind(final_results, new_results)



################################################################################
# Part 10 - Experiment - The Movie + User + Genre Model
################################################################################

# This model attempts to simultaneously capture the effect of the
# movie being rated, the user doing the rating, and the movie's genre will have 
# over each score, and then summing those effects to the total average to 
# estimate the final score.
build_test_movie_user_genre_model <- function(alpha) {
  
  # We start by calculating, once more, the average (centered around zero) each
  # movie has on the rating that's given.
  movie_data_frame <- train_ds %>% 
    group_by(movieId) %>% 
    summarize(movie_avg = mean(sum(rating - total_avg)), n_movie=n(), movie_effect = movie_avg/(n_movie + alpha)) %>%
    select(movieId, movie_effect)
  
  # Now the process is repeated for the users, but to discount the effect movies
  # have on ratings, the previously calculated movie_effect is subtracted from
  # the rating.
  user_data_frame <- train_ds %>% 
    inner_join(movie_data_frame) %>%
    mutate(rating = rating - movie_effect - total_avg) %>%
    group_by(userId) %>% 
    summarize(user_avg = mean(sum(rating)), n_user=n(), user_effect = user_avg/(n_user + alpha)) %>%
    select(userId, user_effect)
  
  # The same is done with the genre, with both the movie and user effect being
  # subtracted before the genre effect is extracted.
  genre_data_frame <- train_ds %>% 
    inner_join(movie_data_frame) %>%
    inner_join(user_data_frame) %>%
    mutate(rating = rating - movie_effect - user_effect - total_avg) %>%
    group_by(genres) %>% 
    summarize(genre_avg = mean(sum(rating)), n_genre=n(), genre_effect = genre_avg/(n_genre + alpha)) %>%
    select(genres, genre_effect)
  
  # Scores are predicted on the test set. Each given score is a combination of
  # the total average observed in train_ds (total_avg), the movie effect,
  # the user effect, and the genre effect.
  predicted_ratings <- test_ds %>% 
    inner_join(user_data_frame) %>%
    inner_join(movie_data_frame) %>%
    inner_join(genre_data_frame) %>%
    mutate(predicted_ratings = total_avg + movie_effect + user_effect + genre_effect) %>%
    pull(predicted_ratings)
  
  # Results are calculated and returned.
  rmse <- RMSE(predicted_ratings, test_ds$rating)
  
  c(Model='Movie + User + Genre Effect Model', Alpha=as.character(alpha), RMSE=rmse)
}

# The function is executed for regularizarion parameter (alpha)
# between 0 and 30 (in increments of 0.25). The higher the parameter,
# the stronger the regularization. Returned results are appended to
# dataset.
new_results <- sapply(seq(0, 30, 0.25), build_test_movie_user_genre_model)
new_results <- as.data.frame(t(new_results))

final_results <- rbind(final_results, new_results)



################################################################################
# Part 11 - Experiment - The Movie + User + Genre + Year Model
################################################################################

# This model attempts to simultaneously capture the effect of the
# movie being rated, the user doing the rating, the movie's genre, and movie's
# release year will have over each score, and then summing those effects to the 
# total average to estimate the final score.
build_test_movie_user_genre_movie_year_model <- function(alpha) {
  
  # We start by calculating, once more, the average (centered around zero) each
  # movie has on the rating that's given.
  movie_data_frame <- train_ds %>% 
    group_by(movieId) %>% 
    summarize(movie_avg = mean(sum(rating - total_avg)), n_movie=n(), movie_effect = movie_avg/(n_movie + alpha)) %>%
    select(movieId, movie_effect)
  
  # Now the process is repeated for the users, but to discount the effect movies
  # have on ratings, the previously calculated movie_effect is subtracted from
  # the rating.
  user_data_frame <- train_ds %>% 
    inner_join(movie_data_frame) %>%
    mutate(rating = rating - movie_effect - total_avg) %>%
    group_by(userId) %>% 
    summarize(user_avg = mean(sum(rating)), n_user=n(), user_effect = user_avg/(n_user + alpha)) %>%
    select(userId, user_effect)
  
  # The same is done with the genre, with both the movie and user effect being
  # subtracted.
  genre_data_frame <- train_ds %>% 
    inner_join(movie_data_frame) %>%
    inner_join(user_data_frame) %>%
    mutate(rating = rating - movie_effect - user_effect - total_avg) %>%
    group_by(genres) %>% 
    summarize(genre_avg = mean(sum(rating)), n_genre=n(), genre_effect = genre_avg/(n_genre + alpha)) %>%
    select(genres, genre_effect)
  
  # Now it's time for the year effect to be extracted. Note that the user, movie,
  # and genre effect are subtracted before the year effect is obtained.
  movie_year_data_frame <- train_ds %>% 
    inner_join(movie_data_frame) %>%
    inner_join(user_data_frame) %>%
    inner_join(genre_data_frame) %>%
    mutate(rating = rating - movie_effect - user_effect - genre_effect - total_avg) %>%
    group_by(movie_year) %>% 
    summarize(movie_year_avg = mean(sum(rating)), n_movie_year=n(), movie_year_effect = movie_year_avg/(n_movie_year + alpha)) %>%
    select(movie_year, movie_year_effect)
  
  # Scores are predicted on the test set. Each given score is a combination of
  # the total average observed in train_ds (total_avg), the movie effect,
  # the user effect, the genre effect, and the release year effect.
  predicted_ratings <- test_ds %>% 
    inner_join(user_data_frame) %>%
    inner_join(movie_data_frame) %>%
    inner_join(genre_data_frame) %>%
    inner_join(movie_year_data_frame) %>%
    mutate(predicted_ratings = total_avg + movie_effect + user_effect + genre_effect + movie_year_effect) %>%
    pull(predicted_ratings)
  
  # Results are calculated and returned.
  rmse <- RMSE(predicted_ratings, test_ds$rating)
  
  c(Model='Movie + User + Genre + Movie Year Effect Model', Alpha=as.character(alpha), RMSE=rmse)
}

# The function is executed for regularizarion parameter (alpha)
# between 0 and 30 (in increments of 0.25). The higher the parameter,
# the stronger the regularization. Returned results are appended to
# dataset.
new_results <- sapply(seq(0, 30, 0.25), build_test_movie_user_genre_movie_year_model)
new_results <- as.data.frame(t(new_results))

final_results <- rbind(final_results, new_results)



################################################################################
# Part 12 - Experiment - The Movie + User + Genre + Year + Rating Week Model
################################################################################

# This model attempts to simultaneously capture the effect of the
# movie being rated, the user doing the rating, the movie's genre, the movie's
# release year, and the rating week will have over each score, and then summing 
# those effects to the total average to estimate the final score.
build_test_movie_user_genre_movie_year_rating_week_model <- function(alpha) {
  
  # We start by calculating, once more, the average (centered around zero) each
  # movie has on the rating that's given.
  movie_data_frame <- train_ds %>% 
    group_by(movieId) %>% 
    summarize(movie_avg = mean(sum(rating - total_avg)), n_movie=n(), movie_effect = movie_avg/(n_movie + alpha)) %>%
    select(movieId, movie_effect)
  
  # Now the process is repeated for the users, but to discount the effect movies
  # have on ratings, the previously calculated movie_effect is subtracted from
  # the rating.
  user_data_frame <- train_ds %>% 
    inner_join(movie_data_frame) %>%
    mutate(rating = rating - movie_effect - total_avg) %>%
    group_by(userId) %>% 
    summarize(user_avg = mean(sum(rating)), n_user=n(), user_effect = user_avg/(n_user + alpha)) %>%
    select(userId, user_effect)
  
  # The same is done with the genre, with both the movie and user effect being
  # subtracted.
  genre_data_frame <- train_ds %>% 
    inner_join(movie_data_frame) %>%
    inner_join(user_data_frame) %>%
    mutate(rating = rating - movie_effect - user_effect - total_avg) %>%
    group_by(genres) %>% 
    summarize(genre_avg = mean(sum(rating)), n_genre=n(), genre_effect = genre_avg/(n_genre + alpha)) %>%
    select(genres, genre_effect)
  
  # Now it's time for the year effect to be extracted. Note that the user, movie,
  # and genre effect are subtracted before the year effect is obtained.
  movie_year_data_frame <- train_ds %>% 
    inner_join(movie_data_frame) %>%
    inner_join(user_data_frame) %>%
    inner_join(genre_data_frame) %>%
    mutate(rating = rating - movie_effect - user_effect - genre_effect - total_avg) %>%
    group_by(movie_year) %>% 
    summarize(movie_year_avg = mean(sum(rating)), n_movie_year=n(), movie_year_effect = movie_year_avg/(n_movie_year + alpha)) %>%
    select(movie_year, movie_year_effect)
  
  # Finally, the rating week effect is obtained.
  rating_week_data_frame <- train_ds %>% 
    inner_join(movie_data_frame) %>%
    inner_join(user_data_frame) %>%
    inner_join(genre_data_frame) %>%
    inner_join(movie_year_data_frame) %>%
    mutate(rating = rating - movie_effect - user_effect - genre_effect - movie_year_effect - total_avg) %>%
    group_by(rating_week) %>% 
    summarize(rating_week_avg = mean(sum(rating)), n_rating_week=n(), rating_week_effect = rating_week_avg/(n_rating_week + alpha)) %>%
    select(rating_week, rating_week_effect)
  
  # Scores are predicted on the test set. Each given score is a combination of
  # the total average observed in train_ds (total_avg), the movie effect,
  # the user effect, the genre effect, the release year effect, and the
  # rating week effect.
  predicted_ratings <- test_ds %>% 
    inner_join(user_data_frame) %>%
    inner_join(movie_data_frame) %>%
    inner_join(genre_data_frame) %>%
    inner_join(movie_year_data_frame) %>%
    inner_join(rating_week_data_frame) %>%
    mutate(predicted_ratings = total_avg + movie_effect + user_effect + genre_effect + movie_year_effect + rating_week_effect) %>%
    pull(predicted_ratings)
  
  # Results are calculated and returned.
  rmse <- RMSE(predicted_ratings, test_ds$rating)
  
  c(Model='Movie + User + Genre + Movie Year + Rating Week Effect Model', Alpha=as.character(alpha), RMSE=rmse)
}

# The function is executed for regularizarion parameter (alpha)
# between 0 and 30 (in increments of 0.25). The higher the parameter,
# the stronger the regularization. Returned results are appended to
# dataset.
new_results <- sapply(seq(0, 30, 0.25), build_test_movie_user_genre_movie_year_rating_week_model)
new_results <- as.data.frame(t(new_results))

final_results <- rbind(final_results, new_results)



################################################################################
# Part 13 - Experiment - Finding the Best Model
################################################################################

# Before obtaining the final results, the model that will be applied on
# the validation dataset is picked. It will, naturally, be the one with the
# lowest RMSE. The result of the best model is printed on the screen.

best_result_index <- which.min(final_results$RMSE)

model_name <- final_results[best_result_index,]$Model
alpha <- as.numeric(final_results[best_result_index,]$Alpha)
training_rmse <- final_results[best_result_index,]$RMSE

cat(paste('Best Model - Training Dataset:\n',
          'Model:', model_name, '\n',
          'Alpha:', alpha, '\n',
          'Training RMSE:', as.character(training_rmse)))



################################################################################
# Part 14 - Experiment - Final Results
################################################################################

# The best model is built with the full training dataset (edx) and applied to 
# the validation dataset. The final result is printed on the screen.

movie_data_frame <- edx %>% 
  group_by(movieId) %>% 
  summarize(movie_avg = mean(sum(rating - total_avg)), n_movie=n(), movie_effect = movie_avg/(n_movie + alpha)) %>%
  select(movieId, movie_effect)

user_data_frame <- edx %>% 
  inner_join(movie_data_frame) %>%
  mutate(rating = rating - movie_effect - total_avg) %>%
  group_by(userId) %>% 
  summarize(user_avg = mean(sum(rating)), n_user=n(), user_effect = user_avg/(n_user + alpha)) %>%
  select(userId, user_effect)

genre_data_frame <- edx %>% 
  inner_join(movie_data_frame) %>%
  inner_join(user_data_frame) %>%
  mutate(rating = rating - movie_effect - user_effect - total_avg) %>%
  group_by(genres) %>% 
  summarize(genre_avg = mean(sum(rating)), n_genre=n(), genre_effect = genre_avg/(n_genre + alpha)) %>%
  select(genres, genre_effect)

movie_year_data_frame <- edx %>% 
  inner_join(movie_data_frame) %>%
  inner_join(user_data_frame) %>%
  inner_join(genre_data_frame) %>%
  mutate(rating = rating - movie_effect - user_effect - genre_effect - total_avg) %>%
  group_by(movie_year) %>% 
  summarize(movie_year_avg = mean(sum(rating)), n_movie_year=n(), movie_year_effect = movie_year_avg/(n_movie_year + alpha)) %>%
  select(movie_year, movie_year_effect)

rating_week_data_frame <- edx %>% 
  inner_join(movie_data_frame) %>%
  inner_join(user_data_frame) %>%
  inner_join(genre_data_frame) %>%
  inner_join(movie_year_data_frame) %>%
  mutate(rating = rating - movie_effect - user_effect - genre_effect - movie_year_effect - total_avg) %>%
  group_by(rating_week) %>% 
  summarize(rating_week_avg = mean(sum(rating)), n_rating_week=n(), rating_week_effect = rating_week_avg/(n_rating_week + alpha)) %>%
  select(rating_week, rating_week_effect)

# When predicting on the validation dataset, a few transformations are done so
# it also has the rating week and movie year columns. Plus, it is joined
# with the other data frames via a left join. This is done in case the
# edx dataset lacks a genre, movie year, or rating week that the
# validation set contains. If that's the case, the effects of those
# items are zero.
predicted_ratings <- validation %>% 
  mutate(rating_week = round_date(as_datetime(timestamp), unit = "week")) %>%
  extract(title, "movie_year", regex = "\\(([0-9 \\-]*)\\)$", remove = F) %>%
  left_join(user_data_frame) %>%
  left_join(movie_data_frame) %>%
  left_join(genre_data_frame) %>%
  left_join(movie_year_data_frame) %>%
  left_join(rating_week_data_frame) %>%
  mutate(predicted_ratings = total_avg + movie_effect + user_effect + genre_effect + movie_year_effect + rating_week_effect) %>%
  pull(predicted_ratings)

# Results are calculated and returned.
final_rmse <- RMSE(predicted_ratings, validation$rating)

cat(paste('Best Model - Final Result:\n',
          'Model:', model_name, '\n',
          'Alpha:', alpha, '\n',
          'Validation RMSE:', as.character(final_rmse)))