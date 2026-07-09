# Dataset Description


Dataset Description
IIT-H Campus Life Prediction Dataset
1 Intro
This document describes the How the IIT-H Campus Life Prediction Dataset created . The goal
of the dataset is to predict the mess duration, i.e., the amount of time (in minutes) a student
spends in the mess during a particular meal.
The dataset tries simulates approximately realistic student behavior at IIT Hyderabad by in-
corporating factors such as:
 student lifestyle patterns
 academic schedule
 project pressure
 weather conditions
 sleep cycle
The dataset is designed to be used for training and evaluating neural network models while
studying the evolution of neural network architectures.
2 Dataset Overview
The dataset is generated in two stages:
 Base Dataset: 500 samples representing simulated observations collected over approximately
15 days.
 Extended Dataset: 5000 samples created by augmenting the base dataset using Gaussian
noise.
Each row corresponds to a single students mess visit for a specific meal.
The target variable is:
Mess Duration (minutes)
which represents how long the student spends eating or socializing in the mess.
1


3 Feature Description
The dataset contains multiple features capturing student behavior, academic workload, and envi-
ronmental conditions.
Feature Type Description
student id Integer Unique identifier for each student
day of week Categorical Day index (06 where Monday = 0)
meal time Categorical 0: breakfast, 1: lunch, 2: dinner
meal type Binary 0: vegetarian, 1: non-vegetarian
is weekend Binary Indicates if the day is Saturday or Sunday
class day Binary 0: classes scheduled, 1: no classes
project deadline Binary 0: project due that day, 1: none
temperature Continuous Temperature in degrees Celsius
humidity Continuous Relative humidity (%)
wind speed Continuous Wind speed (km/h)
rain Binary Indicates rainfall occurrence
air quality Continuous Air quality index value
rising time Continuous Wake-up time in minutes from midnight
sleeping time Continuous Sleep time in minutes from midnight
mess duration Continuous Target variable: time spent in mess (minutes)
4 Data Generation Methodology
Since real student data is not available, the dataset is simulated using probabilistic models that
mimic realistic student behavior.
4.1 Student Information
Each sample is assigned a randomly generated student ID. The day of the week and meal time are
selected randomly to simulate different student visits to the mess.
The variable is weekend is derived from the day of the week.
4.2 Academic Schedule Variables
Two variables simulate academic pressure:
 class day: Indicates whether academic classes are scheduled.
 deadline flag: Indicates whether a time-sensitive deliverable is scheduled for that day.
These factors affect how much time a student spends eating in the mess.
For example:
 Students tend to eat faster when projects are due.
 On days without classes, students may spend more time socializing in the mess.
2


4.3 Weather Simulation
Weather features are generated using Gaussian distributions centered around typical climatic con-
ditions in Hyderabad.
 Temperature: N (30, 3)
 Humidity: N (60, 10)
 Wind Speed: N (8, 2)
Rain is modeled using a Bernoulli distribution with probability 0.2.
Air quality values are generated around a mean AQI of 80 with moderate variance.
4.4 Sleep Cycle Simulation
Student sleep patterns are simulated as follows:
 Rising time is centered around 7:00 AM.
 Sleeping time is centered around 11:00 PM.
5 Target Variable Generation
The target variable mess duration is generated using a heuristic rule that reflects common student
behavior.
Several factors influence the duration:
 Meal time (dinner generally takes longer)
 Weekend effect
 Project deadlines
 Weather conditions
For example:
 Students spend more time during dinner compared to breakfast.
 Weekends increase social interaction, increasing mess duration.
 Project deadlines reduce the time spent eating.
A small Gaussian noise term is added to introduce randomness and avoid deterministic rela-
tionships.
3


6 Dataset Augmentation
To create a larger dataset suitable for training neural networks, the base dataset is expanded to
5000 samples.
This is done by sampling rows from the original dataset and adding Gaussian noise to selected
continuous features.
Noise is applied to the following variables:
 temperature
 humidity
 wind speed
 rising time
 sleeping time
This approach preserves the structure of the original data while increasing dataset size and
variability.
7 Final Thoughts
The generated dataset tries to captures realistic aspects of student life at IIT Hyderabad, including
lifestyle patterns, academic workload, and environmental conditions.
The dataset is designed to be sufficiently complex for training neural networks while remaining
interpretable for experimentation with classical neural network models such as:
 Perceptron
 Multi-layer Perceptrons
 Modern deep neural networks
The use of dataset augmentation allows models to be trained on a larger dataset while main-
taining realistic statistical properties.
4
