with open('database_manager.py', 'r') as f:
    content = f.read()

# Apply fixes
content = content.replace('self.db_params', 'self.db_config')
content = content.replace('with self.get_connection() as conn:', '''conn = self.get_connection()
        if conn is None:
            print("❌ Could not connect to database")
            return
        with conn:''')

# Write back
with open('database_manager.py', 'w') as f:
    f.write(content)

print("✅ Fixed database_manager.py")