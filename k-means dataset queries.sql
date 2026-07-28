SELECT
    o.user_id,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(opp.product_id) AS total_products,
    ROUND(AVG(opp.add_to_cart_order)::numeric, 2) AS avg_cart_position,
    ROUND(AVG(o.days_since_prior_order)::numeric, 2) AS avg_days_between_orders,
    SUM(CASE WHEN opp.reordered THEN 1 ELSE 0 END) AS total_reorders
FROM orders o
JOIN order_products_prior opp
ON o.order_id = opp.order_id
GROUP BY o.user_id;