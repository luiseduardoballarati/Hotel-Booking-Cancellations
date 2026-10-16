# Hotel Booking Cancellations

Predicting which hotel bookings will be cancelled, and estimating how much money that could save the hotel.

## The problem

About 1 in 3 hotel bookings get cancelled. Late cancellations mean empty rooms and lost money. If the hotel knows which bookings are likely to cancel, it can act: ask for a deposit, send a reminder, or overbook safely.

**Main question:** Can we predict cancellations well enough to save the hotel money, and how much?

## Data

- Source: [TidyTuesday Hotels dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11) (two hotels in Portugal, one City Hotel and one Resort Hotel).
- 119,390 bookings, 32 columns, with arrival dates from 2015 to 2017.
- Target: `is_canceled` (1 = cancelled, 0 = showed up). About 37% of bookings are cancelled.
- Prices (`adr`, the average price per night) are in euros.

## What the notebook does

1. **Data check.** Looks at the shape, types, missing values and duplicates.
2. **Outliers.** Looks at `lead_time`, nights, children, babies and `adr`. Most outliers are kept because they look like real bookings. Two rows are removed: one with a negative `adr` and one with `adr` = 5,400.
3. **Missing values.**
   - `agent` and `company`: blanks become `"individual"`, and IDs are turned into text so the model doesn't treat them as numbers.
   - `children` and `country` have only a few blanks.
4. **Exploration.** Compares cancelled vs. not cancelled by lead time, dates, agents, companies and countries.
5. **Feature engineering.** Adds party size, booking date (arrival date minus lead time), and booking month, weekday and year.
6. **Feature selection.** Keeps features with a correlation above |0.1| with the target, plus selected one-hot columns (deposit type, country, market segment, hotel, customer type and others).
7. **Leakage checks.** Removes the target, a helper column that copied the target, and `reservation_status`, which is the same as the target.
8. **Train/test split by time.** Trains on 2015–2016 and tests on 2017, so the model is tested on "the future".
9. **Models.** Logistic Regression (with and without scaling, and with a 0.9 threshold), Random Forest and XGBoost.
10. **Business estimate.** Turns the XGBoost confusion matrix into a rough money estimate.

## Results (test set: 2017, 40,686 bookings, 15,745 cancellations)

Baseline: a model that always says "not cancelled" gets **61.3%** accuracy on 2017.

| Model | Accuracy | Precision (cancel) | Recall (cancel) | F1 (cancel) |
|---|---|---|---|---|
| Logistic Regression (no scaling) | 0.75 | 0.81 | 0.45 | 0.58 |
| Logistic Regression (scaled) | 0.75 | 0.69 | 0.62 | 0.65 |
| Logistic Regression (scaled, threshold 0.9) | 0.72 | 0.99 | 0.27 | 0.42 |
| Random Forest | 0.74 | 0.69 | 0.57 | 0.63 |
| **XGBoost** | **0.77** | **0.78** | **0.58** | **0.67** |

- **XGBoost** has the best accuracy, precision and F1.
- **Scaled Logistic Regression** catches the most cancellations (highest recall).
- **The 0.9 threshold** is almost never wrong when it says "will cancel", but it only catches about a quarter of cancellations. It's useful for overbooking, where a false alarm is expensive.

**Strongest signals** (from the scaled Logistic Regression coefficients):
- More likely to cancel: previous cancellations, a non-refundable deposit, guests from Portugal, and long lead times.
- Less likely to cancel: car parking requests and special requests.

## Business estimate (rough)

These are the XGBoost results on 2017:
- 9,146 cancellations correctly flagged, out of 15,745.
- 2,583 false alarms (flagged as "will cancel" but showed up).
- 11,729 bookings flagged in total.

The current estimate in the notebook assumes:
- each cancellation costs about €480 (average price per night × average nights);
- each reminder costs €100;
- every reminder sent to a real canceller saves the full booking.

Under these assumptions, the model saves about **€3.2M** in 2017. This is an upper bound, because not every reminder will stop a cancellation (see next steps).

## Known limitations

- Some features may not be known at booking time: `assigned_room_type`, `booking_changes` and possibly `country`. Might be a leak problem. 
- 2017 only covers arrivals up to August.

## Suggested next steps

**Improve the model**

1. Add ROC AUC to compare models across all thresholds.
2. Tune XGBoost and Random Forest settings (depth, number of trees, learning rate), using a validation set from the training years.
3. Look at XGBoost feature importance, and compare it with the Logistic Regression coefficients.
4. Try separate models for the City Hotel and the Resort Hotel.


## How to run

```
pip install pandas plotly scikit-learn xgboost
jupyter notebook HotelBookings.ipynb
```

The notebook loads the data directly from GitHub, so no download is needed.
