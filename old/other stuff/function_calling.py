def calculator(operation: str, x: float, y: float) -> float:
    """
    Perform basic arithmetic operations
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        x: First number
        y: Second number
        
    Returns:
        float: Result of the operation
    """
    print(f"Calculating {operation} of {x} and {y}")
    operations = {
        'add': lambda a, b: a + b,
        'subtract': lambda a, b: a - b,
        'multiply': lambda a, b: a * b,
        'divide': lambda a, b: a / b if b != 0 else "Error: Division by zero"
    }
    
    if operation not in operations:
        return f"Error: Invalid operation. Please use one of {list(operations.keys())}"
    
    return operations[operation](x, y)

def number_to_emoji_sum(numbers: list) -> str:
    """
    Convert a list of integers to their emoji representation and append the sum as emojis.
    Args:
        numbers: List of integers
    Returns:
        str: Emoji string with sum
    """
    emoji_map = {
        0: '0️⃣', 1: '1️⃣', 2: '2️⃣', 3: '3️⃣', 4: '4️⃣',
        5: '5️⃣', 6: '6️⃣', 7: '7️⃣', 8: '8️⃣', 9: '9️⃣'
    }
    def int_to_emoji(n):
        return ''.join(emoji_map[int(d)] for d in str(abs(n)))
    emoji_numbers = [int_to_emoji(n) for n in numbers]
    total = sum(numbers)
    emoji_total = int_to_emoji(total)
    return f"{' '.join(emoji_numbers)} = {emoji_total}"

def handle_math_problem(problem: str) -> str:
    """
    Handle a mathematical problem by checking if an answer exists or saving it as pending.
    
    Args:
        problem: The mathematical problem to handle
        
    Returns:
        str: Response indicating if answer exists or problem was saved
    """
    import json
    import os
    
    # File to store problems and answers
    PROBLEMS_FILE = "math_problems.json"
    
    # Create file if it doesn't exist
    if not os.path.exists(PROBLEMS_FILE):
        with open(PROBLEMS_FILE, 'w') as f:
            json.dump({"problems": []}, f)
    
    # Read existing problems
    with open(PROBLEMS_FILE, 'r') as f:
        data = json.load(f)
    
    # Check if problem already exists
    for p in data["problems"]:
        if p["problem"] == problem:
            if p["status"] == "answered":
                return f"Answer exists: {p['answer']}"
            else:
                return "This problem is pending an answer."
    
    # If problem doesn't exist, save it as pending
    data["problems"].append({
        "problem": problem,
        "status": "pending",
        "answer": None
    })
    
    # Save updated data
    with open(PROBLEMS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    
    return "Problem saved as pending. An answer will be provided later."

def get_available_apartments(city: str = None, min_price: float = None, max_price: float = None) -> list:
    """
    Retrieve available apartments from the database based on filters.
    
    Args:
        city: Optional city filter
        min_price: Optional minimum price filter
        max_price: Optional maximum price filter
        
    Returns:
        list: List of available apartments matching the criteria
    """
    import psycopg2
    from psycopg2 import sql
    
    # Database connection parameters
    DB_PARAMS = {
        'dbname': 'apartments_db',
        'user': 'your_username',
        'password': 'your_password',
        'host': 'localhost',
        'port': '5432'
    }
    
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Build the query
        query = """
            SELECT 
                apartment_id,
                title,
                city,
                price_per_night,
                bedrooms,
                bathrooms,
                available_from,
                available_to
            FROM apartments
            WHERE status = 'available'
        """
        params = []
        
        # Add filters if provided
        if city:
            query += " AND city = %s"
            params.append(city)
        if min_price:
            query += " AND price_per_night >= %s"
            params.append(min_price)
        if max_price:
            query += " AND price_per_night <= %s"
            params.append(max_price)
        
        # Execute query
        cur.execute(query, params)
        results = cur.fetchall()
        
        # Format results
        apartments = []
        for row in results:
            apartments.append({
                'id': row[0],
                'title': row[1],
                'city': row[2],
                'price_per_night': row[3],
                'bedrooms': row[4],
                'bathrooms': row[5],
                'available_from': row[6].strftime('%Y-%m-%d'),
                'available_to': row[7].strftime('%Y-%m-%d')
            })
        
        return apartments
        
    except Exception as e:
        return f"Error accessing database: {str(e)}"
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()
