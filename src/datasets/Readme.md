# Data

The following files have been provided by the organization. For a detailed exploration of the files refers to `notebooks/eda-gx.ipynb`

## Volume

Historical (pre-generics) volume for __1078 country-brands__ that went generic in the past. Notice that volume can be in different units depending on the country and brand (miligrams, packs, pills, etc)

The data frame is grouped by _country_ and _brand_, that means that each entry represents an specific combination of _country_ and _brand_ over the months observed. Volume is provided at monthly level.

_month_num_ is related to the timing of the data, with respect to the month of the generic entry. That is, when a generic entry to the market `month_num = 0`, when this variable is positive corresponds to a time after the generic entry and viceversa, when is negative is for months prior to the generic. In addition, _month_name_ corresponds to the actual month of the year.

## Number of generics

For each _country_ and _brand_ the numbers of generics drugs with the same molecule that entry to the market.

## Packaging

This can give us an insight into the presentation of the drug for an specific _country_ and _brand_

## Therapeutic Area

A threapeutic area represents a group of similar diseases or conditions under a generalized headache. For example, cardiovascular relates to heart diseases, dermatology to skin diseases, etc.

## Panel

This dataset contains the channels of distribution, that it the channels throught the drug is sold. For instance, drugs can be distributed by hospitals, retails, etc. For each _country_ and _brand_ the distribution rate is provided and add to 100.

## Submission template

Example of the submission file