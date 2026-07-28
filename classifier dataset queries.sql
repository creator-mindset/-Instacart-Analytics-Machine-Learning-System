SELECT
    o.user_id,
    opp.product_id,
    opp.add_to_cart_order,
    o.order_number,
    o.order_dow,
    o.order_hour_of_day,
    opp.reordered
FROM order_products_prior opp
JOIN orders o
ON opp.order_id = o.order_id;
