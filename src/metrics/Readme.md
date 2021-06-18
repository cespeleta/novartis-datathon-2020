# Datathon Evaluation Metric

The datathon metric contains two parts: point accuracy and confidence intervals.

## Accuracy metric

To compute the prediction error we will evaluate the difference between the predicted values and the actual volume in four different ways weighted as follows:

1. Absolute __monthly__ error of all 24 months (50%)

2. Absolute __accumulated__ error of months 0 to 5 (30%)

3. Absolute __accumulated__ error of months 6 to 11 (10%)

4. Absolute __accumulated__ error of months 12 to 23 (10%)

All the four items will be normalized by the average monthly volume of the last 12 months before the generic entry in order to take into account the magnitude of the brand.

## Confidence intervals

Given the prediction intervals $(L_j, U_j)$ for a particular example we will measure 2 things with the following weights:

1. Whether the actual values fall inside the intervals (15%):
$$L_{j,i}\le Y^{act}_{j,i} \le U_{j,i}$$

2. How wide are the predictions intervals. Wider prediction intervals will have higher penalization (85%).
$$|U_{j, i} - L_{j, i}|$$

For business reasons, the confidence error for the first 6 months will be weighted more than the rest of the months (60% and 40% respectively). The error will be also normalized by the average monthly volume of the brand in the 12 months prior to the generic entry.