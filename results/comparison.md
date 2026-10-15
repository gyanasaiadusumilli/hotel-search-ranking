| subset                   | model                           |   ndcg@5 |   ndcg@10 |   ndcg@38 |   booked_in_top5 |
|:-------------------------|:--------------------------------|---------:|----------:|----------:|-----------------:|
| all_test                 | Random order                    |   0.1589 |    0.2295 |    0.3511 |           0.2661 |
| all_test                 | Cheapest first                  |   0.2053 |    0.2767 |    0.3844 |           0.3308 |
| all_test                 | Highest stars first             |   0.1903 |    0.2658 |    0.3736 |           0.3202 |
| all_test                 | Best reviews first              |   0.1627 |    0.2369 |    0.3539 |           0.2835 |
| all_test                 | Best location first             |   0.2287 |    0.2988 |    0.4002 |           0.3748 |
| all_test                 | Most booked hotels first        |   0.2621 |    0.3314 |    0.4225 |           0.4238 |
| all_test                 | Logistic regression (pointwise) |   0.3415 |    0.4088 |    0.4782 |           0.5415 |
| all_test                 | LightGBM classifier (pointwise) |   0.3773 |    0.4428 |    0.5036 |           0.5865 |
| all_test                 | LambdaMART (LightGBM ranker)    |   0.3807 |    0.4463 |    0.5064 |           0.5878 |
| random_order_searches    | Random order                    |   0.1606 |    0.2327 |    0.3554 |           0.2762 |
| random_order_searches    | Cheapest first                  |   0.208  |    0.2786 |    0.3888 |           0.3296 |
| random_order_searches    | Highest stars first             |   0.1936 |    0.27   |    0.3786 |           0.3315 |
| random_order_searches    | Best reviews first              |   0.1636 |    0.2403 |    0.3586 |           0.2812 |
| random_order_searches    | Best location first             |   0.2128 |    0.2825 |    0.3912 |           0.4072 |
| random_order_searches    | Most booked hotels first        |   0.2382 |    0.3089 |    0.4089 |           0.4264 |
| random_order_searches    | Logistic regression (pointwise) |   0.2949 |    0.3659 |    0.4485 |           0.5487 |
| random_order_searches    | LightGBM classifier (pointwise) |   0.3262 |    0.3966 |    0.4704 |           0.5971 |
| random_order_searches    | LambdaMART (LightGBM ranker)    |   0.3318 |    0.4016 |    0.475  |           0.5978 |
| expedia_ordered_searches | Random order                    |   0.1581 |    0.2281 |    0.3493 |           0.2655 |
| expedia_ordered_searches | Cheapest first                  |   0.2039 |    0.2753 |    0.3821 |           0.3313 |
| expedia_ordered_searches | Highest stars first             |   0.1903 |    0.2634 |    0.3713 |           0.3188 |
| expedia_ordered_searches | Best reviews first              |   0.1628 |    0.2357 |    0.3525 |           0.2819 |
| expedia_ordered_searches | Best location first             |   0.2358 |    0.3055 |    0.404  |           0.3744 |
| expedia_ordered_searches | Most booked hotels first        |   0.2717 |    0.3407 |    0.428  |           0.4244 |
| expedia_ordered_searches | Logistic regression (pointwise) |   0.3613 |    0.4271 |    0.4908 |           0.5411 |
| expedia_ordered_searches | LightGBM classifier (pointwise) |   0.399  |    0.4625 |    0.5177 |           0.5859 |
| expedia_ordered_searches | LambdaMART (LightGBM ranker)    |   0.4015 |    0.4654 |    0.5198 |           0.5871 |
| expedia_ordered_searches | Expedia's own ranking           |   0.4025 |    0.4658 |    0.519  |           0.591  |