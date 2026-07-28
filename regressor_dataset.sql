SELECT
    user_id,
    order_number,
    order_dow,
    order_hour_of_day,
    days_since_prior_order
FROM orders
WHERE days_since_prior_order IS NOT NULL;