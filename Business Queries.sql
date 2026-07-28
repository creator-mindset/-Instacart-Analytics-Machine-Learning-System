CREATE DATABASE instacart_db

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    user_id INT NOT NULL,
    eval_set VARCHAR(10) NOT NULL,
    order_number INT NOT NULL,
    order_dow SMALLINT NOT NULL CHECK (order_dow BETWEEN 0 AND 6),
    order_hour_of_day SMALLINT NOT NULL CHECK (order_hour_of_day BETWEEN 0 AND 23),
    days_since_prior_order REAL
);

SELECT * FROM orders;

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    aisle_id INT NOT NULL,
    department_id INT NOT NULL
);

SELECT * FROM products;

CREATE TABLE order_products_prior (
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    add_to_cart_order INT NOT NULL,
    reordered BOOLEAN NOT NULL,

    PRIMARY KEY (order_id, product_id)
);

SELECT * FROM order_products_prior;

CREATE TABLE order_products_train (
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    add_to_cart_order INT NOT NULL,
    reordered BOOLEAN NOT NULL,

    PRIMARY KEY (order_id, product_id)
);

SELECT * FROM order_products_train;

CREATE TABLE aisles (
    aisle_id INT PRIMARY KEY,
    aisle VARCHAR(100) NOT NULL
);

SELECT * FROM aisles;

CREATE TABLE departments (
    department_id INT PRIMARY KEY,
    department VARCHAR(100) NOT NULL
);

SELECT * FROM departments;

ALTER TABLE products
ADD CONSTRAINT fk_products_aisles
FOREIGN KEY (aisle_id)
REFERENCES aisles(aisle_id);

ALTER TABLE products
ADD CONSTRAINT fk_products_departments
FOREIGN KEY (department_id)
REFERENCES departments(department_id);

ALTER TABLE order_products_prior
ADD CONSTRAINT fk_prior_orders
FOREIGN KEY (order_id)
REFERENCES orders(order_id);

ALTER TABLE order_products_prior
ADD CONSTRAINT fk_prior_products
FOREIGN KEY (product_id)
REFERENCES products(product_id);

ALTER TABLE order_products_train
ADD CONSTRAINT fk_train_orders
FOREIGN KEY (order_id)
REFERENCES orders(order_id);

ALTER TABLE order_products_train
ADD CONSTRAINT fk_train_products
FOREIGN KEY (product_id)
REFERENCES products(product_id);

-- How many total orders has the company received?

SELECT COUNT(*) AS Total_orders
FROM orders;

-- Total orders received by company are 3421083

-- How many unique customers do we have?

SELECT COUNT(DISTINCT(user_id)) AS Total_customers
FROM orders;
-- Unique Customers are 206209

-- What is the average number of order placed by customers?

SELECT
AVG(orders_count) AS avg_orders_per_customer
FROM(
SELECT 
user_id,
COUNT(*) AS orders_count
FROM orders
GROUP BY user_id
)t;

-- Average orders placed by customers is around 17

-- Which customers placed the highest number of orders?
SELECT user_id,
COUNT(*) AS Total_Orders
FROM orders
GROUP BY user_id
ORDER BY Total_Orders DESC
LIMIT 3;

-- Customers with user_id 310,210 and 313 placed highest number of orders

-- Which hour of the day receives the maximum orders?

SELECT order_hour_of_day,
COUNT(*) AS Total_Orders
FROM orders
GROUP BY order_hour_of_day
ORDER BY Total_Orders DESC;

-- At 10 a.m.,company receives maximum orders

-- Which days of the week receives the maximum orders?
SELECT
order_dow,
COUNT(*) AS Total_Orders
FROM orders
GROUP BY order_dow
ORDER BY Total_Orders DESC;

-- Sunday receives the maximum orders

-- Most Ordered Products

SELECT p.product_name,
COUNT(*) AS total_orders
FROM order_products_prior opp
JOIN products p
ON opp.product_id = p.product_id
GROUP BY p.product_name
ORDER BY total_orders DESC
LIMIT 3;

-- Banana,Bag of Organic Bananas and Organic Strawberries are most ordered products

-- Most Reordered Products

SELECT
p.product_name,
COUNT(*) AS reorder_count
FROM order_products_prior opp
JOIN products p
ON opp.product_id = p.product_id
WHERE reordered = TRUE
GROUP BY p.product_name
ORDER BY reorder_count DESC
LIMIT 3;

--  Banana,Bag of Organic Bananas and Organic Strawberries are most reordered products

-- Which department has the highest number of ordered products?

SELECT 
d.department,
COUNT(*) AS Total_Orders
FROM order_products_prior opp
JOIN products p
ON opp.product_id = p.product_id
JOIN departments d
ON p.department_id = d.department_id
GROUP BY d.department
ORDER BY Total_Orders DESC
LIMIT 1;

-- produce department has the highest number of ordered products with 9479291 orders

-- Average Basket Size

SELECT
ROUND(AVG(product_count),2) AS average_basket_size
FROM (
  SELECT
   order_id,
   COUNT(product_id) AS product_count
   FROM order_products_prior
   GROUP BY order_id
   )t;

-- Average basket size is 10

-- Largest Basket Ever Purchased

SELECT 
    order_id,
	COUNT(product_id) AS basket
	FROM order_products_prior
   GROUP BY order_id
   ORDER BY basket DESC;

-- Largest basket size ever purchased by order_id 1564244 by a basket size of 145 products

-- Reoreder rate of every product

SELECT 
p.product_name,
ROUND(100.0 * SUM(CASE WHEN reordered THEN 1 ELSE 0 END) / COUNT(*),2) AS reorder_rate
FROM products p
JOIN order_products_prior opp
ON p.product_id = opp.product_id
GROUP BY p.product_name
ORDER BY reorder_rate DESC;

-- Percentage of customers having more than 20 orders

SELECT ROUND(COUNT(*) * 100.0/(SELECT COUNT(DISTINCT(user_id)) FROM orders),2) AS percentage_of_customers_with_more_than_20_orders
FROM(SELECT
     user_id
	 FROM orders
	 GROUP BY user_id
	 HAVING COUNT(*) > 20
	 )t;
-- 24.60 % customers have more than 20 orders

-- Top 3 products of each department

WITH product_sales AS
(
SELECT
d.department,
p.product_name,
COUNT(*) total_orders
FROM order_products_prior opp
JOIN products p
ON opp.product_id=p.product_id
JOIN departments d
ON p.department_id=d.department_id
GROUP BY d.department,p.product_name
)

SELECT *
FROM
(
SELECT *,
DENSE_RANK() OVER
(PARTITION BY department ORDER BY total_orders DESC) rnk
FROM product_sales
)x
WHERE rnk<=3;