DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS all_data;

CREATE TABLE categories (
    id INTEGER PRIMARY KEY, -- Unique identifier for each category
    name TEXT NOT NULL      -- Name of the category
);

CREATE TABLE all_data (
    id INTEGER PRIMARY KEY, -- Unique identifier for each entry
    further_remarks TEXT,    -- Additional remarks or descriptions
    assigned BOOLEAN DEFAULT 0,        -- Indicates if the entry is assigned
    category INTEGER DEFAULT NULL,        -- Foreign key to the category table
    ml_assigned BOOLEAN DEFAULT 0,     -- Indicates if the entry is machine learning assigned
    ml_category INTEGER DEFAULT NULL,     -- Foreign key to the category table for machine learning
    ml_probability REAL,     -- Probability score from machine learning
    random_sampled BOOLEAN  DEFAULT 0,  -- Indicates if the entry is randomly sampled
    FOREIGN KEY (category) REFERENCES categories (id),
    FOREIGN KEY (ml_category) REFERENCES categories (id)
);